// hailo_re/turbo/turbo.cpp: LD_PRELOAD library for Hailo's DFC `compiler` binary (hailo_tools/build/compiler).
// The allocator's partition search places every candidate context with OR-tools CP-SAT in SEQUENTIAL mode
// (one worker; "*** starting sequential search"), so a multi-hour compile uses 1 of N cores. CP-SAT merges the
// gflags string FLAGS_cp_model_params into the parameters of EVERY solve (cp_model_solver.cc); the binary never
// parses a command line, so we set it through gflags' public API once the binary has registered its flags.
//   HAILO_CP_PARAMS  text SatParameters for cp_model_params (unset: left alone)
//   HAILO_GFLAGS     any other flags, "name=value;name=value", e.g. "cp_model_dump_file=/x/m.pbtxt" (SolveCpModel
//                    writes each model it solves there, text format, overwriting: see cpdump.py to keep them all)
//   HAILO_DUMP_SEQ   path P: every write-open of exactly P (the cp_model_dump_file) goes to P.NNNNN instead, so
//                    concurrent solves (the splitter runs 4 at once) each keep their own model
//   HAILO_THROW_TRACE FILE: every C++ throw appends "type +offset +offset ..." (return addresses as offsets into
//                    the compiler binary) to FILE; name them with cprof/fnmap.py
//   HAILO_TURBO_LOG  1 = report to stderr
// Only acts inside a process whose executable is named `compiler`.
#include <dlfcn.h>
#include <execinfo.h>
#include <link.h>

#include <typeinfo>
#include <fcntl.h>

#include <atomic>
#include <cstdarg>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// gflags API, resolved at run time from whichever libgflags the process loaded (the DFC bundles one with namespace
// `gflags`; distro builds export `google`): no link-time dependency, so the library is inert elsewhere.
using GetFn = bool (*)(const char*, std::string*);
using SetFn = std::string (*)(const char*, const char*);
static GetFn get_fn() {
    for (const char* s : {"_ZN6gflags20GetCommandLineOptionEPKcPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
                          "_ZN6google20GetCommandLineOptionEPKcPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"})
        if (void* f = dlsym(RTLD_DEFAULT, s)) return reinterpret_cast<GetFn>(f);
    return nullptr;
}
static SetFn set_fn() {
    for (const char* s : {"_ZN6gflags20SetCommandLineOptionB5cxx11EPKcS1_", "_ZN6google20SetCommandLineOptionB5cxx11EPKcS1_"})
        if (void* f = dlsym(RTLD_DEFAULT, s)) return reinterpret_cast<SetFn>(f);
    return nullptr;
}

static void apply() {
    std::vector<std::pair<std::string, std::string>> flags;
    if (const char* p = getenv("HAILO_CP_PARAMS")) flags.emplace_back("cp_model_params", p);
    if (const char* g = getenv("HAILO_GFLAGS")) {
        std::string all = g;
        for (size_t at = 0; at < all.size();) {
            size_t end = all.find(';', at);
            if (end == std::string::npos) end = all.size();
            std::string kv = all.substr(at, end - at);
            size_t eq = kv.find('=');
            if (eq != std::string::npos) flags.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
            at = end + 1;
        }
    }
    if (flags.empty()) return;
    bool log = getenv("HAILO_TURBO_LOG") && getenv("HAILO_TURBO_LOG")[0] == '1';
    GetFn get = nullptr;
    SetFn set = nullptr;
    for (int i = 0; i < 20000; ++i) {  // static initialisers register the flags at startup: poll up to ~20 s
        std::string cur;
        if (!get) get = get_fn();
        if (!set) set = set_fn();
        if (get && set && get(flags[0].first.c_str(), &cur)) {
            for (auto& [name, value] : flags) {
                std::string before, now;
                bool known = get(name.c_str(), &before);
                std::string r = known ? set(name.c_str(), value.c_str()) : "unknown flag";
                get(name.c_str(), &now);
                if (log) fprintf(stderr, "[turbo] %s '%s' -> '%s' (%s)\n", name.c_str(), before.c_str(), now.c_str(),
                                 r.c_str());
            }
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (log) fprintf(stderr, "[turbo] flag %s never registered\n", flags[0].first.c_str());
}

// ---- HAILO_DUMP_SEQ: give each write-open of the dump path its own numbered file
static std::atomic<int> g_seq{0};
static std::string seq_path(const char* path, bool writing) {
    static const char* want = getenv("HAILO_DUMP_SEQ");
    if (!want || !path || !writing || strcmp(path, want) != 0) return {};
    char buf[4200];
    snprintf(buf, sizeof(buf), "%s.%05d", path, g_seq.fetch_add(1));
    return buf;
}
static bool mode_writes(const char* m) { return m && (strchr(m, 'w') || strchr(m, 'a') || strchr(m, '+')); }

extern "C" FILE* fopen(const char* path, const char* mode) {
    static auto real = reinterpret_cast<FILE* (*)(const char*, const char*)>(dlsym(RTLD_NEXT, "fopen"));
    std::string p = seq_path(path, mode_writes(mode));
    return real(p.empty() ? path : p.c_str(), mode);
}
extern "C" FILE* fopen64(const char* path, const char* mode) {
    static auto real = reinterpret_cast<FILE* (*)(const char*, const char*)>(dlsym(RTLD_NEXT, "fopen64"));
    std::string p = seq_path(path, mode_writes(mode));
    return real(p.empty() ? path : p.c_str(), mode);
}
extern "C" int open(const char* path, int flags, ...) {
    static auto real = reinterpret_cast<int (*)(const char*, int, ...)>(dlsym(RTLD_NEXT, "open"));
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, mode_t);
        va_end(ap);
    }
    std::string p = seq_path(path, (flags & O_ACCMODE) != O_RDONLY);
    return real(p.empty() ? path : p.c_str(), flags, mode);
}

// ---- HAILO_THROW_TRACE: who throws (the compiler turns most internal errors into one-line messages)
static uintptr_t exe_base() {
    static uintptr_t base = [] {
        uintptr_t b = 0;
        dl_iterate_phdr([](dl_phdr_info* info, size_t, void* out) {
            if (info->dlpi_name && info->dlpi_name[0] == 0) {  // the main executable
                *static_cast<uintptr_t*>(out) = info->dlpi_addr;
                return 1;
            }
            return 0;
        }, &b);
        return b;
    }();
    return base;
}
extern "C" [[noreturn]] void __cxa_throw(void* obj, std::type_info* tinfo, void (*dest)(void*)) {
    static auto real = reinterpret_cast<void (*)(void*, std::type_info*, void (*)(void*))>(dlsym(RTLD_NEXT, "__cxa_throw"));
    static const char* path = getenv("HAILO_THROW_TRACE");
    if (path) {
        void* frames[48];
        int n = backtrace(frames, 48);
        if (FILE* f = fopen(path, "a")) {
            fprintf(f, "%s", tinfo ? tinfo->name() : "?");
            for (int i = 1; i < n; i++) fprintf(f, " +%lx", (unsigned long)((uintptr_t)frames[i] - exe_base()));
            fputc('\n', f);
            fclose(f);
        }
    }
    real(obj, tinfo, dest);
    __builtin_unreachable();
}

// ---- HAILO_CPCACHE DIR: memoize every CP-SAT solve (bit-exact, persistent, shared by concurrent compiles)
// The allocator re-solves identical CP-SAT models many times (router: 111 solves, 17 distinct). CP-SAT in the
// compiler is single-worker with a deterministic time budget, so the same model + the same call site gives the
// same response. We patch the entry of operations_research::sat::SolveCpModel(const CpModelProto&, Model*)
// (stripped; found by its strings, verified by its prologue bytes) to jump to cp_hook:
//   key  = 2 x FNV-1a 64 of the serialized model + model size + call-site offsets (2 frames up)
//   DIR/KEY.req holds the model bytes (a hit compares them exactly), DIR/KEY.resp the serialized response.
//   hit  : build a real response object by solving the EMPTY model (microseconds), then ParseFromString(resp).
//   miss : solve, store (tmp + rename, so concurrent compiles may share DIR).
//   HAILO_CPCACHE_VERIFY=1: on a hit solve anyway and count responses that differ outside the timing fields.
//   HAILO_CPCACHE_STATS=FILE: append one stats line at exit.
#include <cerrno>
#include <sys/mman.h>
#include <sys/stat.h>
#include <mutex>
#include <unordered_map>
static const uintptr_t kSolveOff = 0x2aee3b0;  // DFC 3.33 compiler
static const unsigned char kProlog[19] = {0xf3, 0x0f, 0x1e, 0xfa, 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57,
                                          0x41, 0x56, 0x4c, 0x8d, 0xbd, 0x80, 0xfd, 0xff, 0xff};
using SolveFn = void* (*)(void* sret, const void* proto, void* model);
using SerFn = bool (*)(const void*, std::string*);
using ParseFn = bool (*)(void*, const std::string&);
using NewFn = void* (*)(const void*);
static SolveFn g_orig;
static SerFn g_ser;
static ParseFn g_parse;
static int g_new_idx = -1;
static void* g_empty;
static char g_dirbuf[2048];  // POD: set by the load-time constructor (a std::string here would be re-initialised after it)
#define g_dir std::string(g_dirbuf)
static bool g_verify;
static std::mutex g_mu;  // constexpr constructor: safe
static std::unordered_map<std::string, std::string>& mem() { static auto* m = new std::unordered_map<std::string, std::string>; return *m; }
static std::unordered_map<std::string, double>& costs() { static auto* m = new std::unordered_map<std::string, double>; return *m; }
#define g_mem mem()
#define g_cost costs()
static std::atomic<long> s_calls{0}, s_hits{0}, s_bad{0}, s_store{0};
static std::atomic<long> s_miss_us{0}, s_hit_us{0}, s_saved_us{0};

static uint64_t fnv(const std::string& s, uint64_t h) {
    for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ULL;
    return h;
}
static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static bool read_file(const std::string& p, std::string* out) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    out->clear();
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    fclose(f);
    return true;
}
static void write_file(const std::string& p, const std::string& s) {
    std::string tmp = p + ".tmp" + std::to_string(getpid()) + "_" + std::to_string(gettid());
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { fprintf(stderr, "[cpcache] cannot write %s: %s\n", tmp.c_str(), strerror(errno)); return; }
    bool ok = fwrite(s.data(), 1, s.size(), f) == s.size();
    ok = (fclose(f) == 0) && ok;
    if (ok && rename(tmp.c_str(), p.c_str()) == 0) return;
    fprintf(stderr, "[cpcache] write/rename %s failed: %s\n", p.c_str(), strerror(errno));
    unlink(tmp.c_str());
}
// response without the timing fields (wall_time 15, user_time 16, deterministic_time 17, primal_integral 22)
static std::string strip_timing(const std::string& s) {
    std::string out;
    size_t i = 0;
    auto varint = [&](uint64_t* v) {
        *v = 0;
        for (int sh = 0; i < s.size() && sh < 64; sh += 7) {
            unsigned char b = s[i++];
            *v |= uint64_t(b & 0x7f) << sh;
            if (!(b & 0x80)) return true;
        }
        return false;
    };
    while (i < s.size()) {
        size_t start = i;
        uint64_t tag, v;
        if (!varint(&tag)) return s;
        int wt = tag & 7, fld = int(tag >> 3);
        if (wt == 0) { if (!varint(&v)) return s; }
        else if (wt == 1) i += 8;
        else if (wt == 5) i += 4;
        else if (wt == 2) { if (!varint(&v)) return s; i += v; }
        else return s;
        if (i > s.size()) return s;
        if (fld == 15 || fld == 16 || fld == 17 || fld == 22) continue;
        out.append(s, start, i - start);
    }
    return out;
}
static void* empty_like(const void* proto);
static uintptr_t off_of(void* a) { return (uintptr_t)a - exe_base(); }

extern "C" void* cp_hook(void* sret, const void* proto, void* model) {
    s_calls++;
    std::string req;
    if (!g_ser(proto, &req)) return g_orig(sret, proto, model);
    if (!g_empty) {
        std::lock_guard<std::mutex> l(g_mu);
        if (!g_empty) g_empty = empty_like(proto);
    }
    void* bt[4];
    int n = backtrace(bt, 4);
    char key[160];
    snprintf(key, sizeof key, "%016lx%016lx_%zx_%lx_%lx", (unsigned long)fnv(req, 0xcbf29ce484222325ULL),
             (unsigned long)fnv(req, 0x84222325cbf29ce4ULL), req.size(), n > 1 ? (unsigned long)off_of(bt[1]) : 0UL,
             n > 2 ? (unsigned long)off_of(bt[2]) : 0UL);
    std::string resp;
    bool hit = false;
    double cost = 0;
    {
        std::lock_guard<std::mutex> l(g_mu);
        auto it = g_mem.find(key);
        if (it != g_mem.end()) { resp = it->second; hit = true; cost = g_cost[key]; }
    }
    if (!hit) {
        std::string disk_req, disk_resp, c;
        std::string base = g_dir + "/" + key;
        if (read_file(base + ".resp", &disk_resp) && read_file(base + ".req", &disk_req) && disk_req == req) {
            resp = disk_resp;
            hit = true;
            if (read_file(base + ".us", &c)) cost = atof(c.c_str());
            std::lock_guard<std::mutex> l(g_mu);
            g_mem[key] = resp;
            g_cost[key] = cost;
        }
    }
    double t0 = now_us();
    if (hit && !g_verify) {
        g_orig(sret, g_empty, model);  // constructs a CpSolverResponse in sret (empty model: trivial)
        if (g_parse(sret, resp)) {
            s_hits++;
            s_hit_us += long(now_us() - t0);
            s_saved_us += long(cost);
            return sret;
        }
        // could not parse: fall through to a real solve into a fresh object is impossible (sret is built);
        // the caller gets the empty-model response -> never happens unless the cache is corrupt; log it
        fprintf(stderr, "[cpcache] parse failed for %s\n", key);
        return sret;
    }
    g_orig(sret, proto, model);
    double dt = now_us() - t0;
    std::string out;
    g_ser(sret, &out);
    if (hit) {  // verify mode
        s_hits++;
        if (strip_timing(out) != strip_timing(resp)) {
            s_bad++;
            write_file(g_dir + "/BAD_" + key + ".resp_new", out);
        }
        return sret;
    }
    s_miss_us += long(dt);
    {
        std::lock_guard<std::mutex> l(g_mu);
        g_mem[key] = out;
        g_cost[key] = dt;
    }
    std::string base = g_dir + "/" + key;
    write_file(base + ".req", req);
    write_file(base + ".us", std::to_string(dt));
    write_file(base + ".resp", out);  // last: a .resp implies a complete .req
    s_store++;
    return sret;
}

static void cp_stats() {
    const char* p = getenv("HAILO_CPCACHE_STATS");
    char line[512];
    snprintf(line, sizeof line,
             "pid %d calls %ld hits %ld stored %ld verify_bad %ld solve_s %.1f hit_overhead_s %.2f saved_s %.1f\n",
             getpid(), s_calls.load(), s_hits.load(), s_store.load(), s_bad.load(), s_miss_us.load() / 1e6,
             s_hit_us.load() / 1e6, s_saved_us.load() / 1e6);
    if (p) { if (FILE* f = fopen(p, "a")) { fputs(line, f); fclose(f); } }
    if (getenv("HAILO_TURBO_LOG")) fputs(line, stderr);
}

static void cp_install() {
    const char* d = getenv("HAILO_CPCACHE");
    if (!d || !*d) return;
    snprintf(g_dirbuf, sizeof g_dirbuf, "%s", d);
    mkdir(d, 0755);
    g_verify = getenv("HAILO_CPCACHE_VERIFY") && getenv("HAILO_CPCACHE_VERIFY")[0] == '1';
    void* lib = dlopen("libprotobuf.so.3.7.1.0", RTLD_NOW | RTLD_NOLOAD);
    if (!lib) lib = RTLD_DEFAULT;
    g_ser = (SerFn)dlsym(lib, "_ZNK6google8protobuf11MessageLite17SerializeToStringEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    g_parse = (ParseFn)dlsym(lib, "_ZN6google8protobuf11MessageLite15ParseFromStringERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    void** vt = (void**)dlsym(lib, "_ZTVN6google8protobuf11MessageLiteE");
    if (!g_ser || !g_parse || !vt) { fprintf(stderr, "[cpcache] protobuf symbols missing: off\n"); return; }
    // MessageLite vtable: find New(Arena*) by name; New() is the slot before it
    for (int i = 2; i < 24; i++) {
        Dl_info di;
        if (dladdr(vt[i], &di) && di.dli_sname && strcmp(di.dli_sname, "_ZNK6google8protobuf11MessageLite3NewEPNS0_5ArenaE") == 0) {
            g_new_idx = i - 2 - 1;  // index from the address point
            break;
        }
    }
    if (g_new_idx < 0) { fprintf(stderr, "[cpcache] MessageLite::New slot not found: off\n"); return; }
    unsigned char* f = (unsigned char*)(exe_base() + kSolveOff);
    if (memcmp(f, kProlog, sizeof kProlog) != 0) { fprintf(stderr, "[cpcache] SolveCpModel prologue mismatch (other DFC version?): off\n"); return; }
    // trampoline: the 19 original bytes (no rip-relative operand) + jmp back to f+19
    unsigned char* tr = (unsigned char*)mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memcpy(tr, kProlog, 19);
    unsigned char jmp[13] = {0x49, 0xbb, 0, 0, 0, 0, 0, 0, 0, 0, 0x41, 0xff, 0xe3};  // movabs r11, imm64; jmp r11
    uintptr_t back = (uintptr_t)f + 19;
    memcpy(jmp + 2, &back, 8);
    memcpy(tr + 19, jmp, 13);
    g_orig = (SolveFn)tr;
    uintptr_t hook = (uintptr_t)&cp_hook;
    memcpy(jmp + 2, &hook, 8);
    uintptr_t pg = (uintptr_t)f & ~uintptr_t(4095);
    size_t len = ((uintptr_t)f + 19 - pg + 4095) & ~size_t(4095);
    if (mprotect((void*)pg, len, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) { fprintf(stderr, "[cpcache] mprotect failed: off\n"); return; }
    memcpy(f, jmp, 13);
    for (int i = 13; i < 19; i++) f[i] = 0x90;
    mprotect((void*)pg, len, PROT_READ | PROT_EXEC);
    atexit(cp_stats);
    if (getenv("HAILO_TURBO_LOG")) fprintf(stderr, "[cpcache] on: %s (New slot %d)%s\n", d, g_new_idx, g_verify ? " VERIFY" : "");
}

// the empty model: made lazily from the first real model via its virtual New()
static void* empty_like(const void* proto) {
    void** vptr = *(void***)proto;
    return ((NewFn)vptr[g_new_idx])(proto);
}


// ---- level-max utilization descent (allocator::ResourceUtilizationDecrementer ctor at 0x140ec80, DFC 3.33)
//   (this, const ResourcesParams&, xmm0..xmm4 = five doubles; xmm2 = step, CHECKed > 0)
//   HAILO_UTIL_LOG=1  print the arguments
//   HAILO_UTIL_ONLY=X run exactly ONE level-max step at control utilization X (others min(X,1)), so N processes can
//                     cover the whole 120% -> 75% descent at once instead of one timeout after another
static const uintptr_t kDecOff = 0x140ec80;
static const unsigned char kDecProlog[16] = {0xf3, 0x0f, 0x1e, 0xfa, 0x41, 0x57, 0x66, 0x0f, 0x28, 0xea, 0x41, 0x56, 0x66, 0x0f, 0x14, 0xeb};
using DecFn = void (*)(void*, const void*, double, double, double, double, double);
static DecFn g_dec_orig;
extern "C" void dec_hook(void* self, const void* rp, double a, double b, double c, double d, double e) {
    if (getenv("HAILO_UTIL_LOG")) fprintf(stderr, "[turbo] ResourceUtilizationDecrementer(a=%g b=%g step=%g d=%g e=%g)\n", a, b, c, d, e);
    if (const char* x = getenv("HAILO_UTIL_ONLY")) {  // start the descent at X: every cap = min(X, its own start)
        double X = atof(x);
        fprintf(stderr, "[turbo] HAILO_UTIL_ONLY=%g: control %g->%g memory %g->%g other %g->%g\n", X, a, X < a ? X : a, b,
                X < b ? X : b, e, X < e ? X : e);
        a = X < a ? X : a; b = X < b ? X : b; e = X < e ? X : e;
    }
    g_dec_orig(self, rp, a, b, c, d, e);
}
static void* make_tramp(unsigned char* f, const unsigned char* prolog, size_t n) {
    unsigned char* tr = (unsigned char*)mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memcpy(tr, prolog, n);
    unsigned char jmp[13] = {0x49, 0xbb, 0, 0, 0, 0, 0, 0, 0, 0, 0x41, 0xff, 0xe3};
    uintptr_t back = (uintptr_t)f + n;
    memcpy(jmp + 2, &back, 8);
    memcpy(tr + n, jmp, 13);
    return tr;
}
static bool patch_jmp(unsigned char* f, size_t n, void* to) {
    unsigned char jmp[13] = {0x49, 0xbb, 0, 0, 0, 0, 0, 0, 0, 0, 0x41, 0xff, 0xe3};
    uintptr_t h = (uintptr_t)to;
    memcpy(jmp + 2, &h, 8);
    uintptr_t pg = (uintptr_t)f & ~uintptr_t(4095);
    size_t len = ((uintptr_t)f + n - pg + 4095) & ~size_t(4095);
    if (mprotect((void*)pg, len, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) return false;
    memcpy(f, jmp, 13);
    for (size_t i = 13; i < n; i++) f[i] = 0x90;
    mprotect((void*)pg, len, PROT_READ | PROT_EXEC);
    return true;
}
static void dec_install() {
    if (!getenv("HAILO_UTIL_LOG") && !getenv("HAILO_UTIL_ONLY")) return;
    unsigned char* f = (unsigned char*)(exe_base() + kDecOff);
    if (memcmp(f, kDecProlog, 16) != 0) { fprintf(stderr, "[turbo] decrementer prologue mismatch: off\n"); return; }
    g_dec_orig = (DecFn)make_tramp(f, kDecProlog, 16);
    if (!patch_jmp(f, 16, (void*)&dec_hook)) fprintf(stderr, "[turbo] decrementer patch failed\n");
}

__attribute__((constructor)) static void turbo_init() {
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) return;
    exe[n] = 0;
    const char* base = strrchr(exe, '/');
    if (!base || strcmp(base + 1, "compiler") != 0) return;
    cp_install();
    dec_install();
    std::thread(apply).detach();
}
