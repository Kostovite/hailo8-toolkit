/* LD_PRELOAD shim: run Hailo's HSim.so CUDA kernels on Maxwell (sm_52).
 *
 * HSim.so ships its kernels only as relocatable sm_61 cubins + sm_61 PTX (the sm_52 image is an empty device-link
 * stub), so on an M40 the runtime finds no kernel ("GPUassert: named symbol not found src/HConv2D.cu").
 * build.sh retargets the embedded PTX to sm_52 and links one cubin with every kernel; this shim swaps it in when
 * HSim.so registers its fat binaries. HSim.so itself is not modified.
 *
 *
 * Second defect (any GPU): HSim launches every kernel on stream 0 (`xor %r9d,%r9d` before each
 * __cudaPushCallConfiguration) while TensorFlow produces/consumes the tensors on its own non-blocking stream, so
 * HSim can read inputs before TF finished writing them (nondeterministic garbage in the last rows of conv2).
 * cudaLaunchKernel calls coming from HSim.so are fenced with cudaDeviceSynchronize before and after.
 *
 * Env: HSIM_SM52_FATBIN = path to hsim_sm52.fatbin (required), HSIM_SM52_VERBOSE=1 logs each registration.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int magic;
    int version;
    const void *data;
    void *filename_or_fatbins;
} fatbin_wrapper_t;

static void *load_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = aligned_alloc(64, (n + 63) / 64 * 64);
    if (buf && fread(buf, 1, n, f) != (size_t)n) { free(buf); buf = NULL; }
    fclose(f);
    return buf;
}

void **__cudaRegisterFatBinary(void *fat) {
    static void **(*real)(void *) = NULL;
    static void *sm52 = NULL;
    static fatbin_wrapper_t wrap;
    if (!real) real = (void **(*)(void *))dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
    Dl_info info;
    int from_hsim = dladdr(fat, &info) && info.dli_fname && strstr(info.dli_fname, "HSim.so");
    const fatbin_wrapper_t *w = (const fatbin_wrapper_t *)fat;
    if (getenv("HSIM_SM52_VERBOSE"))
        fprintf(stderr, "[hsim_sm52] register fatbin from %s magic=%#x version=%d%s\n",
                info.dli_fname ? info.dli_fname : "?", w->magic, w->version, from_hsim ? " -> sm_52 image" : "");
    if (!from_hsim) return real(fat);
    if (!sm52) {
        const char *p = getenv("HSIM_SM52_FATBIN");
        sm52 = p ? load_file(p) : NULL;
        if (!sm52) {
            fprintf(stderr, "[hsim_sm52] cannot load HSIM_SM52_FATBIN=%s; using the original image\n", p ? p : "(unset)");
            return real(fat);
        }
        wrap.magic = w->magic;
        wrap.version = 1; /* plain (non-relocatable) registration of the fully linked image */
        wrap.data = sm52;
        wrap.filename_or_fatbins = NULL;
    }
    return real(&wrap);
}


typedef struct { unsigned x, y, z; } dim3_t;
typedef int (*launch_fn)(const void *, dim3_t, dim3_t, void **, size_t, void *);

static int from_hsim_addr(void *addr) {
    Dl_info info;
    return dladdr(addr, &info) && info.dli_fname && strstr(info.dli_fname, "HSim.so");
}

int cudaLaunchKernel(const void *func, dim3_t grid, dim3_t block, void **args, size_t shmem, void *stream) {
    static launch_fn real = NULL;
    static int (*sync)(void) = NULL;
    if (!real) {
        real = (launch_fn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
        sync = (int (*)(void))dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    }
    if (!from_hsim_addr(__builtin_return_address(0))) return real(func, grid, block, args, shmem, stream);
    sync();
    int r = real(func, grid, block, args, shmem, stream);
    sync();
    return r;
}
