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

__attribute__((constructor)) static void turbo_init() {
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) return;
    exe[n] = 0;
    const char* base = strrchr(exe, '/');
    if (!base || strcmp(base + 1, "compiler") != 0) return;
    std::thread(apply).detach();
}
