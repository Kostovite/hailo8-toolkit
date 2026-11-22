// Fused sm_52 kernels for the Hailo-8 integer datapath used by qatfast.QATNet.
//
// Every kernel reproduces intops/qatnet bit for bit (int64 arithmetic in registers, fp32 tensors in memory):
//   expand    : padded layer input -> [hi; lo] nibbles (batch-stacked, one cuDNN call) + residue features A[x mod M]
//   mac_tail  : nibble/residue conv outputs -> exact accumulator (+ conv-add term) -> overflow partials -> 16b
//               accumulator lossy -> bias lossy -> APU; writes y, the backward scale (APU slope * bias mask) and a
//               code byte (accumulator in-range bit + overflow sign) for the custom backward
//   val_tail  : same tail for an already-accumulated value (ew_add / activation / reduce_sum / matmul / ew_mult)
//   rshift_mul: v = lossy(bankers_round_shift(x * k[c], s, 3, signed)), grad scale k/2^s * in-range
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

typedef long long i64;

struct Lossy {
    int bits, sgn, wrap, on;
};

// All datapath values are int32: every Hailo-8 element here is <= 16 bits and the widest intermediate is
// mant (<= 2^10) * input (<= 2^16) or the pre-shift MAC sum (<= 255 * 128 * 9216 < 2^29). Host code rejects
// configs above 24 bits. 64-bit integer ops and F2I.S64 are multi-instruction on sm_52.
// Silicon extends the first/last APU segment past the table ends; SDK_BIT_EXACT zeroes them (-DHQ_SDK_EDGES).
#ifdef HQ_SDK_EDGES
#define HQ_EDGE_VALID(x) (x)
#else
#define HQ_EDGE_VALID(x) true
#endif
__device__ __forceinline__ int lo_of(Lossy c) { return c.sgn ? -(1 << (c.bits - 1)) : 0; }
__device__ __forceinline__ int hi_of(Lossy c) { return c.sgn ? (1 << (c.bits - 1)) - 1 : (1 << c.bits) - 1; }

__device__ __forceinline__ int hw_lossy(int x, Lossy c, bool sym) {
    if (!c.on) return x;
    int lo = lo_of(c), hi = hi_of(c);
    if (c.wrap) {
        x = ((x - lo) & ((1 << c.bits) - 1)) + lo;  // python mod for a power of two
    }
    if (sym && c.sgn) lo += 1;
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ bool inside(int x, Lossy c) { return !c.on || (x >= lo_of(c) && x <= hi_of(c)); }

// acceleras bankers_round_with_shift (intops.apu_round_shift)
__device__ __forceinline__ int round_shift(int v, int sh, int maxb, bool sgn) {
    int br = sh < maxb ? sh : maxb;
    int sign = 1, tb;
    if (sgn) {
        tb = v >> (sh - br);
    } else {
        sign = (v > 0) - (v < 0);
        tb = (v < 0 ? -v : v) >> (sh - br);
    }
    int fl = tb >> br, odd = fl & 1, mod = tb - (fl << br), twice = 2 * mod, base = 1 << br;
    int out = fl + (twice > base) + (twice == base) * odd;
    return sign * out;
}

struct Apu {
    const int *thr;    // C x (P+1), ceil of the float thresholds (clamped to +-2^30)
    const int *mant;   // C x P
    const int *shift;  // C x P
    const int *offs;   // C x P
    int P, fs, mant_signed;
    Lossy c1, c2, out;
};

// Returns y; *slope = mant/2^(shift+fs) of the hit segment, 0 where y saturates (qatnet.APU backward).
// *pk (optional, for trainable APU tables) = segment | unsaturated << 4: unsaturated = valid segment and none of
// clip1 / clip2 / output saturated, i.e. where dy/dmant = a 2^-(sh+fs) and dy/doffs = 2^-fs.
__device__ __forceinline__ int apu_eval(const Apu &A, int c, int a, float *slope, int8_t *pk = nullptr) {
    const int *t = A.thr + c * (A.P + 1);  // block-uniform c: broadcast loads
    int idx = 0;
#pragma unroll 16
    for (int p = 1; p < 16; ++p) idx += (p < A.P) && a >= __ldg(t + p);
    bool valid = HQ_EDGE_VALID(a >= __ldg(t) && a < __ldg(t + A.P));
    int o = c * A.P + idx;
    int m = __ldg(A.mant + o), sh = __ldg(A.shift + o);
    int of = __ldg(A.offs + o);
    int v = valid ? round_shift(m * a, sh, 6, A.mant_signed) : 0;
    int v1 = hw_lossy(v, A.c1, false);
    int v2pre = v1 + (valid ? of : 0);
    int v2 = hw_lossy(v2pre, A.c2, false);
    int v3 = round_shift(v2, A.fs, A.fs, false);
    int vo = hw_lossy(v3, A.out, A.out.sgn);
    if (pk) {
        bool unsat = valid && v1 == v && v2 == v2pre && vo == v3;
        *pk = (int8_t)(idx | (unsat ? 16 : 0));
    }
    v = vo;
    if (slope) {
        int lo = A.out.sgn ? -(1 << (A.out.bits - 1)) + 1 : 0;
        int hi = A.out.sgn ? (1 << (A.out.bits - 1)) - 1 : (1 << A.out.bits) - 1;
        *slope = (v > lo && v < hi) ? ldexpf((float)m, -(sh + A.fs)) : 0.f;
    }
    return v;
}

static Lossy make_lossy(std::vector<int64_t> c) {
    if (c.empty()) return {0, 0, 0, 0};
    TORCH_CHECK(c[0] >= 1 && c[0] <= 24, "lossy element of ", c[0], " bits: int32 datapath supports <= 24");
    return {(int)c[0], (int)c[1], (int)c[2], 1};
}

static Apu make_apu(const torch::Tensor &thr, const torch::Tensor &mant, const torch::Tensor &shift,
                    const torch::Tensor &offs, int fs, bool mant_signed, std::vector<int64_t> c1,
                    std::vector<int64_t> c2, std::vector<int64_t> out) {
    Apu A;
    TORCH_CHECK(thr.dtype() == torch::kInt32 && offs.dtype() == torch::kInt32 && mant.size(1) <= 16);
    A.thr = thr.data_ptr<int>();
    A.mant = mant.data_ptr<int>();
    A.shift = shift.data_ptr<int>();
    A.offs = offs.data_ptr<int>();
    A.P = (int)mant.size(1);
    A.fs = fs;
    A.mant_signed = mant_signed;
    A.c1 = make_lossy(c1);
    A.c2 = make_lossy(c2);
    A.out = make_lossy(out);
    return A;
}


// Block sum with warp shuffles and ONE barrier (on sm_52 every bar.sync also emits a MEMBAR, so the classic
// log2(BS)-barrier smem tree cost more than the whole datapath). Fixed order: deterministic.
template <int BS>
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float w[BS / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
    if (threadIdx.x == 0)
        for (int i = 0; i < BS / 32; ++i) t += w[i];
    return t;
}

constexpr int EPT = 8;  // elements per thread in the tail / backward kernels

// ---------------------------------------------------------------- expand
// grid: (ceil(HW / block), N * C); 32-bit index math only (64-bit div/mod is a software routine on sm_52)
__global__ void expand_k(const float *__restrict__ x, float *__restrict__ hl, float *__restrict__ xa,
                         const float *__restrict__ fa, int r, int m, i64 n_total, int HW, int planes) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= HW) return;
    int plane = blockIdx.y;
    i64 i = (i64)plane * HW + p;
    float v = x[i];
    if (planes == 2) {
        float hi = floorf(v * 0.0625f);
        hl[i] = hi;
        hl[i + n_total] = v - 16.f * hi;
    } else {  // 16-bit inputs: 4 nibble planes, most significant first (top plane keeps the sign)
        int q = __float2int_rn(v);
        int top = q >> 12;
        hl[i] = (float)top;
        int rest = q - (top << 12);
        hl[i + n_total] = (float)(rest >> 8);
        hl[i + 2 * n_total] = (float)((rest >> 4) & 15);
        hl[i + 3 * n_total] = (float)(rest & 15);
    }
    if (xa) {
        int q = (int)v & (m - 1);
        float *dst = xa + (i64)plane * r * HW + p;
        for (int j = 0; j < r; ++j) dst[(i64)j * HW] = fa[q * r + j];
    }
}

// Single-conv input: per channel c, (1 + r) consecutive channels [x_c, A[x_c mod M, 0..r-1]] (group-contiguous, so
// grouped/depthwise convs keep their groups). Sets *flag |= 1 if an input is outside the 0..255 the bound assumes.
__global__ void expand_cat_k(const float *__restrict__ x, float *__restrict__ out, const float *__restrict__ fa,
                             int r, int m, int HW, int *flag) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= HW) return;
    int plane = blockIdx.y;
    float v = x[(i64)plane * HW + p];
    if (!(v >= 0.f && v <= 255.f)) atomicOr(flag, 1);
    int q = (int)v & (m - 1);
    float *dst = out + (i64)plane * (r + 1) * HW + p;
    dst[0] = v;
    for (int j = 0; j < r; ++j) dst[(i64)(j + 1) * HW] = fa[q * r + j];
}

torch::Tensor expand_cat(torch::Tensor x, torch::Tensor fa, int64_t s, torch::Tensor flag) {
    TORCH_CHECK(x.is_contiguous() && x.dtype() == torch::kFloat32 && s > 0);
    int N = x.size(0), C = x.size(1), r = fa.size(1);
    i64 HW = x.size(2) * x.size(3);
    auto out = torch::empty({N, C * (r + 1), x.size(2), x.size(3)}, x.options());
    int th = 256;
    if (x.numel())
        expand_cat_k<<<dim3((HW + th - 1) / th, N * C), th, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), fa.data_ptr<float>(), r, 1 << (s + 1), (int)HW,
            flag.data_ptr<int>());
    return out;
}

std::vector<torch::Tensor> expand(torch::Tensor x, torch::Tensor fa, int64_t s, int64_t planes) {
    TORCH_CHECK(x.is_contiguous() && x.dtype() == torch::kFloat32 && (planes == 2 || planes == 4));
    int N = x.size(0), C = x.size(1);
    i64 HW = x.size(2) * x.size(3), per_n = C * HW, tot = N * per_n;
    auto hl = torch::empty({planes * N, C, x.size(2), x.size(3)}, x.options());
    torch::Tensor xa;
    int r = 0, m = 1 << (s + 1);
    if (s > 0) {
        r = fa.size(1);
        xa = torch::empty({N, C * r, x.size(2), x.size(3)}, x.options());
    }
    int th = 256;
    dim3 grid((HW + th - 1) / th, N * C);
    if (tot)
        expand_k<<<grid, th, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), hl.data_ptr<float>(), s > 0 ? xa.data_ptr<float>() : nullptr,
            s > 0 ? fa.data_ptr<float>() : nullptr, r, m, tot, (int)HW, (int)planes);
    return {hl, s > 0 ? xa : torch::Tensor()};
}

// ---------------------------------------------------------------- tails
struct Tail {
    // accumulator source
    const float *hl;  // MAC: [hi; lo] conv outputs (2N x C x HW); VAL: the value tensor (N x C x HW)
    const float *R;   // MAC residue conv output (nullable)
    int cst;          // MAC: n_taps * 2^(s-1)
    int s, mac, split;
    const float *wo;  // single conv: per-output-channel sum_taps G[0, k mod M] (the weight-only residue term)
    int *flag;        // |= 2 when a conv output is not within 0.25 of an integer (non-exact cuDNN algorithm)
    // conv-add
    const float *res, *f;
    float sca_inv, feed;
    // overflow
    float ovf_thr;
    float *ovf_part;
    Lossy acc;
    // bias
    const float *bias;
    int bias_per_c;
    Lossy bcfg;
    Apu A;
    // outputs
    float *y, *gs;
    int8_t *code;
    int *aux_a;       // trainable APU tables: APU input per element
    int8_t *aux_p;    // and segment | unsaturated bit
};

// grid: (ceil(HW / (BS * EPT)), N * C); each thread handles EPT elements of one (n, c) plane
template <int BS>
__global__ void tail_k(Tail T, i64 tot, int C, int HW) {
    int p0 = blockIdx.x * BS * EPT + threadIdx.x;
    int c = blockIdx.y % C;
    float ov = 0.f;
    for (int u = 0; u < EPT; ++u) {
        int p = p0 + u * BS;
        if (p >= HW) break;
        i64 i = (i64)blockIdx.y * HW + p;
        int a;
        if (T.mac && T.split) {  // integer combine; the numerator is divisible by 2^s, so >> is exact division
            // 4 planes: sum x*k can exceed int32, but only its low 16 + s bits survive the shift and the 16-bit
            // accumulator, so the combine runs mod 2^32 (unsigned wrap) and the arithmetic shift stays exact
            unsigned ua = 0u;
            for (int j = 0; j < T.split; ++j) {
                float h = T.hl[i + j * tot];
                if (fabsf(h - rintf(h)) > 0.25f) atomicOr(T.flag, 2);
                ua = ua * 16u + (unsigned)__float2int_rn(h);
            }
            if (T.s > 0) ua = ua + (unsigned)T.cst - (unsigned)__float2int_rn(T.R[i]);
            a = (int)ua >> T.s;
        } else if (T.mac) {  // single conv: sum x*k - residue
            float v = T.hl[i];
            if (fabsf(v - rintf(v)) > 0.25f) atomicOr(T.flag, 2);
            a = (__float2int_rn(v) + T.cst - (T.wo ? __float2int_rn(__ldg(T.wo + c)) : 0)) >> T.s;
        } else {
            a = __float2int_rn(T.hl[i]);
        }
        if (T.res) {
            float term = rintf(T.res[i] * __ldg(T.f + c) * T.sca_inv);
            a += __float2int_rn(term * T.feed);
        }
        if (T.ovf_part) {
            float ex = fabsf((float)a) - T.ovf_thr;
            ov += ex > 0.f ? ex : 0.f;
        }
        bool in_acc = inside(a, T.acc);
        int a1 = hw_lossy(a, T.acc, false);
        int b = a1;
        if (T.bias) b += __float2int_rn(__ldg(T.bias + (T.bias_per_c ? c : 0)));
        bool in_b = inside(b, T.bcfg);
        b = hw_lossy(b, T.bcfg, false);
        float slope;
        int8_t pk;
        int y = apu_eval(T.A, c, b, T.gs ? &slope : nullptr, T.aux_a ? &pk : nullptr);
        T.y[i] = (float)y;
        if (T.aux_a) {
            T.aux_a[i] = b;
            T.aux_p[i] = pk;
        }
        if (T.gs) {
            T.gs[i] = in_b ? slope : 0.f;
            int osgn = (T.ovf_part && fabsf((float)a) > T.ovf_thr) ? (a > 0 ? 1 : -1) : 0;
            T.code[i] = (int8_t)((in_acc ? 1 : 0) | ((osgn + 1) << 1));
        }
    }
    if (T.ovf_part) {
        float t = block_sum<BS>(ov);
        if (threadIdx.x == 0) T.ovf_part[blockIdx.y * gridDim.x + blockIdx.x] = t;
    }
}

// returns {y, gs, code, ovf_partials}
std::vector<torch::Tensor> tail(torch::Tensor src, torch::Tensor R, int64_t mac, double cst, int64_t s,
                                torch::Tensor res, torch::Tensor f, double sca_inv, double feed,
                                std::vector<int64_t> acc_cfg, bool want_ovf, torch::Tensor bias,
                                std::vector<int64_t> bias_cfg, torch::Tensor thr, torch::Tensor mant,
                                torch::Tensor shift, torch::Tensor offs, int64_t fs, bool mant_signed,
                                std::vector<int64_t> c1, std::vector<int64_t> c2, std::vector<int64_t> out,
                                bool want_grad, torch::Tensor flag, torch::Tensor wo, bool want_aux) {
    TORCH_CHECK(src.is_contiguous() && src.dtype() == torch::kFloat32);
    // mac: 0 = value tail, 1 = split [hi; lo] + residue convs, 2 = single fused conv (R carries the flag buffer)
    // mac 3 = split with 4 nibble planes (16-bit inputs)
    int planes = mac == 1 ? 2 : (mac == 3 ? 4 : 1);
    int N = src.size(0) / planes, C = src.size(1);
    i64 HW = src.size(2) * src.size(3), tot = (i64)N * C * HW;
    TORCH_CHECK(thr.size(0) == C, "APU table channels ", thr.size(0), " != tensor channels ", C);
    auto opt = src.options();
    auto y = torch::empty({N, C, src.size(2), src.size(3)}, opt);
    torch::Tensor gs, code, part;
    constexpr int BS = 256;
    dim3 grid((HW + BS * EPT - 1) / (BS * EPT), N * C);
    int nb = grid.x * grid.y;
    if (want_grad) {
        gs = torch::empty_like(y);
        code = torch::empty(y.sizes(), opt.dtype(torch::kInt8));
    }
    if (want_ovf) part = torch::empty({nb}, opt);
    torch::Tensor aux_a, aux_p;
    if (want_aux) {
        aux_a = torch::empty(y.sizes(), opt.dtype(torch::kInt32));
        aux_p = torch::empty(y.sizes(), opt.dtype(torch::kInt8));
    }
    Tail T;
    T.hl = src.data_ptr<float>();
    T.R = ((mac == 1 || mac == 3) && s > 0) ? R.data_ptr<float>() : nullptr;
    TORCH_CHECK(cst < (1 << 30));
    T.cst = (int)llround(cst);
    T.s = (int)s;
    T.mac = mac != 0;
    T.split = planes > 1 ? planes : 0;
    T.wo = wo.defined() && wo.numel() ? wo.data_ptr<float>() : nullptr;
    T.flag = flag.defined() && flag.numel() ? flag.data_ptr<int>() : nullptr;
    TORCH_CHECK(!T.mac || T.flag, "MAC tail needs the flag buffer");
    T.res = res.defined() && res.numel() ? res.data_ptr<float>() : nullptr;
    T.f = T.res ? f.data_ptr<float>() : nullptr;
    T.sca_inv = (float)sca_inv;
    T.feed = (float)feed;
    T.acc = make_lossy(acc_cfg);
    T.ovf_thr = want_ovf ? (float)(1LL << (acc_cfg[0] - 1)) : 0.f;
    T.ovf_part = want_ovf ? part.data_ptr<float>() : nullptr;
    T.bias = bias.defined() && bias.numel() ? bias.data_ptr<float>() : nullptr;
    T.bias_per_c = T.bias && bias.numel() > 1;
    T.bcfg = make_lossy(bias_cfg);
    T.A = make_apu(thr, mant, shift, offs, (int)fs, mant_signed, c1, c2, out);
    T.y = y.data_ptr<float>();
    T.gs = want_grad ? gs.data_ptr<float>() : nullptr;
    T.code = want_grad ? code.data_ptr<int8_t>() : nullptr;
    T.aux_a = want_aux ? aux_a.data_ptr<int>() : nullptr;
    T.aux_p = want_aux ? aux_p.data_ptr<int8_t>() : nullptr;
    if (tot) tail_k<BS><<<grid, BS, 0, at::cuda::getCurrentCUDAStream()>>>(T, tot, C, (int)HW);
    return {y, gs, code, part, aux_a, aux_p};
}

// ---------------------------------------------------------------- trainable APU tables
// gm[c, seg] += gy * a * 2^-(sh + fs), go[c, seg] += gy * 2^-fs over unsaturated elements (block = one channel plane;
// shared-memory atomics per block, then one global atomic per segment)
template <int BS>
__global__ void apu_tgrad_k(const float *__restrict__ gy, const int *__restrict__ aux_a,
                            const int8_t *__restrict__ aux_p, const int *__restrict__ shift, int P, int fs,
                            float *gm, float *go, int C, int HW) {
    __shared__ float sm[16], so[16];
    if (threadIdx.x < 16) sm[threadIdx.x] = so[threadIdx.x] = 0.f;
    __syncthreads();
    int p0 = blockIdx.x * BS * EPT + threadIdx.x;
    int c = blockIdx.y % C;
    float inv_fs = ldexpf(1.f, -fs);
    for (int u = 0; u < EPT; ++u) {
        int p = p0 + u * BS;
        if (p >= HW) break;
        i64 i = (i64)blockIdx.y * HW + p;
        int pk = aux_p[i];
        if (!(pk & 16)) continue;
        int seg = pk & 15;
        float g = gy[i];
        int sh = __ldg(shift + c * P + seg);
        atomicAdd(&sm[seg], g * (float)aux_a[i] * ldexpf(1.f, -(sh + fs)));
        atomicAdd(&so[seg], g * inv_fs);
    }
    __syncthreads();
    if (threadIdx.x < P) {
        atomicAdd(gm + c * P + threadIdx.x, sm[threadIdx.x]);
        atomicAdd(go + c * P + threadIdx.x, so[threadIdx.x]);
    }
}

// Faster variant: per-thread register accumulators for all 16 segments (unrolled predicated adds, no dynamic indexing,
// no contended atomics: most elements of a layer fall in 1-2 segments, so the shared atomics above serialize 32-way),
// EPT_T elements per thread, then one warp-shuffle reduction per segment and one shared atomic per warp and segment.
constexpr int EPT_T = 32;
template <int BS>
__global__ void apu_tgrad2_k(const float *__restrict__ gy, const int *__restrict__ aux_a,
                             const int8_t *__restrict__ aux_p, const int *__restrict__ shift, int P, int fs,
                             float *gm, float *go, int C, int HW) {
    __shared__ float sm[16], so[16];
    if (threadIdx.x < 16) sm[threadIdx.x] = so[threadIdx.x] = 0.f;
    __syncthreads();
    int p0 = blockIdx.x * BS * EPT_T + threadIdx.x;
    int c = blockIdx.y % C;
    float scale[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) scale[q] = q < P ? ldexpf(1.f, -(__ldg(shift + c * P + q) + fs)) : 0.f;
    float am[16], ao[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) am[q] = ao[q] = 0.f;
    for (int u = 0; u < EPT_T; ++u) {
        int p = p0 + u * BS;
        if (p >= HW) break;
        i64 i = (i64)blockIdx.y * HW + p;
        int pk = aux_p[i];
        float g = (pk & 16) ? gy[i] : 0.f;
        float ga = g * (float)aux_a[i];
        int seg = pk & 15;
#pragma unroll
        for (int q = 0; q < 16; ++q) {
            bool hit = seg == q;
            am[q] += hit ? ga * scale[q] : 0.f;
            ao[q] += hit ? g : 0.f;
        }
    }
    float inv_fs = ldexpf(1.f, -fs);
#pragma unroll
    for (int q = 0; q < 16; ++q) {
        if (q >= P) break;
        float vm = am[q], vo = ao[q];
        for (int o = 16; o > 0; o >>= 1) {
            vm += __shfl_down_sync(0xffffffffu, vm, o);
            vo += __shfl_down_sync(0xffffffffu, vo, o);
        }
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&sm[q], vm);
            atomicAdd(&so[q], vo * inv_fs);
        }
    }
    __syncthreads();
    if (threadIdx.x < P) {
        atomicAdd(gm + c * P + threadIdx.x, sm[threadIdx.x]);
        atomicAdd(go + c * P + threadIdx.x, so[threadIdx.x]);
    }
}

std::vector<torch::Tensor> apu_tgrad(torch::Tensor gy, torch::Tensor aux_a, torch::Tensor aux_p, torch::Tensor shift,
                                     int64_t fs) {
    TORCH_CHECK(gy.is_contiguous() && aux_a.is_contiguous() && aux_p.is_contiguous() && shift.is_contiguous());
    int N = gy.size(0), C = gy.size(1), P = shift.size(1);
    i64 HW = gy.size(2) * gy.size(3);
    auto gm = torch::zeros({C, P}, gy.options()), go = torch::zeros({C, P}, gy.options());
    constexpr int BS = 256;
    static const bool slow = getenv("HQ_TGRAD_SLOW") != nullptr;  // A/B switch: the original atomics kernel
    if (gy.numel() && !slow) {
        dim3 g2((HW + BS * EPT_T - 1) / (BS * EPT_T), N * C);
        apu_tgrad2_k<BS><<<g2, BS, 0, at::cuda::getCurrentCUDAStream()>>>(
            gy.data_ptr<float>(), aux_a.data_ptr<int>(), aux_p.data_ptr<int8_t>(), shift.data_ptr<int>(), P, (int)fs,
            gm.data_ptr<float>(), go.data_ptr<float>(), C, (int)HW);
        return {gm, go};
    }
    dim3 grid((HW + BS * EPT - 1) / (BS * EPT), N * C);
    if (gy.numel())
        apu_tgrad_k<BS><<<grid, BS, 0, at::cuda::getCurrentCUDAStream()>>>(
            gy.data_ptr<float>(), aux_a.data_ptr<int>(), aux_p.data_ptr<int8_t>(), shift.data_ptr<int>(), P, (int)fs,
            gm.data_ptr<float>(), go.data_ptr<float>(), C, (int)HW);
    return {gm, go};
}

// ---------------------------------------------------------------- rshift_mul
__global__ void rshift_mul_k(const float *__restrict__ x, const float *__restrict__ k, int k_per_c, int s,
                             Lossy cfg, float *__restrict__ v, float *__restrict__ gs, int C, int HW) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= HW) return;
    int c = blockIdx.y % C;
    i64 i = (i64)blockIdx.y * HW + p;
    float kk = __ldg(k + (k_per_c ? c : 0));
    int e = round_shift(__float2int_rn(x[i]) * __float2int_rn(kk), s, 3, true);
    v[i] = (float)hw_lossy(e, cfg, false);
    if (gs) gs[i] = inside(e, cfg) ? kk * ldexpf(1.f, -s) : 0.f;
}

std::vector<torch::Tensor> rshift_mul(torch::Tensor x, torch::Tensor k, int64_t s, std::vector<int64_t> cfg,
                                      bool want_grad) {
    TORCH_CHECK(x.is_contiguous() && x.dtype() == torch::kFloat32);
    int C = x.size(1);
    i64 HW = x.size(2) * x.size(3), tot = x.numel();
    auto v = torch::empty_like(x);
    torch::Tensor gs;
    if (want_grad) gs = torch::empty_like(x);
    int th = 256;
    if (tot)
        rshift_mul_k<<<dim3((HW + th - 1) / th, x.size(0) * C), th, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), k.data_ptr<float>(), k.numel() > 1, (int)s, make_lossy(cfg), v.data_ptr<float>(),
            want_grad ? gs.data_ptr<float>() : nullptr, C, (int)HW);
    return {v, gs};
}

// ---------------------------------------------------------------- MAC backward
// g_b = gy * gs; g_acc = g_b * in_acc + (govf / n) * ovf_sign; gz = g_acc * 2^-s; grad_res = (g_acc * feed) * 2^-ca_s * f[c]
// (same fp32 operation order as the unfused torch backward), per-(plane, block) partial sums of g_b for the bias grad.
template <int BS>
__global__ void mac_bwd_k(const float *__restrict__ gy, const float *__restrict__ gs, const int8_t *__restrict__ code,
                          const float *__restrict__ govf, float n_el, float inv_s, const float *__restrict__ f,
                          float feed, float inv_ca, float *__restrict__ gz, float *__restrict__ gres,
                          float *__restrict__ part, int C, int HW) {
    int p0 = blockIdx.x * BS * EPT + threadIdx.x;
    int c = blockIdx.y % C;
    float gn = govf ? __ldg(govf) / n_el : 0.f;
    float fc = gres ? __ldg(f + c) : 0.f;
    float bsum = 0.f;
    for (int u = 0; u < EPT; ++u) {
        int p = p0 + u * BS;
        if (p >= HW) break;
        i64 i = (i64)blockIdx.y * HW + p;
        float gb = gy[i] * gs[i];
        bsum += gb;
        int cd = code[i];
        float ga = gb * (float)(cd & 1);
        if (govf) ga = ga + gn * (float)((cd >> 1) - 1);
        gz[i] = ga * inv_s;
        if (gres) gres[i] = (ga * feed) * inv_ca * fc;
    }
    if (part) {
        float t = block_sum<BS>(bsum);
        if (threadIdx.x == 0) part[blockIdx.y * gridDim.x + blockIdx.x] = t;
    }
}

// returns {gz, grad_res (or undefined), bias partials (N, C, blocks)}
std::vector<torch::Tensor> mac_bwd(torch::Tensor gy, torch::Tensor gs, torch::Tensor code, torch::Tensor govf, int64_t s,
                                   bool want_res, torch::Tensor f, double feed, double inv_ca, bool want_bias) {
    TORCH_CHECK(gy.is_contiguous() && gs.is_contiguous() && code.is_contiguous());
    int N = gy.size(0), C = gy.size(1);
    i64 HW = gy.size(2) * gy.size(3);
    constexpr int BS = 256;
    dim3 grid((HW + BS * EPT - 1) / (BS * EPT), N * C);
    auto gz = torch::empty_like(gy);
    torch::Tensor gres, part;
    if (want_res) gres = torch::empty_like(gy);
    if (want_bias) part = torch::empty({N, C, (int64_t)grid.x}, gy.options());
    if (gy.numel())
        mac_bwd_k<BS><<<grid, BS, 0, at::cuda::getCurrentCUDAStream()>>>(
            gy.data_ptr<float>(), gs.data_ptr<float>(), code.data_ptr<int8_t>(),
            govf.defined() && govf.numel() ? govf.data_ptr<float>() : nullptr, (float)gy.numel(), ldexpf(1.f, -(int)s),
            want_res ? f.data_ptr<float>() : nullptr, (float)feed, (float)inv_ca, gz.data_ptr<float>(),
            want_res ? gres.data_ptr<float>() : nullptr, want_bias ? part.data_ptr<float>() : nullptr, C, (int)HW);
    return {gz, want_res ? gres : torch::Tensor(), want_bias ? part : torch::Tensor()};
}


// ---------------------------------------------------------------- wide (a16_w16) layers
// 32-bit accumulators / biases: the datapath runs in int64 (APU: mant * acc reaches 2^41). Only a handful of layers
// use it, so the slow 64-bit integer ops on sm_52 cost little; the convs stay fp32 (digit planes, exact < 2^24).
struct Lossy64 {
    int bits, sgn, wrap, on;
};

__device__ __forceinline__ i64 hw_lossy64(i64 x, Lossy64 c, bool sym) {
    if (!c.on) return x;
    i64 lo = c.sgn ? -(1LL << (c.bits - 1)) : 0, hi = c.sgn ? (1LL << (c.bits - 1)) - 1 : (1LL << c.bits) - 1;
    if (c.wrap) x = ((x - lo) & ((1LL << c.bits) - 1)) + lo;
    if (sym && c.sgn) lo += 1;
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ i64 round_shift64(i64 v, int sh, int maxb, bool sgn) {
    int br = sh < maxb ? sh : maxb;
    i64 sign = 1, tb;
    if (sgn) {
        tb = v >> (sh - br);
    } else {
        sign = (v > 0) - (v < 0);
        tb = (v < 0 ? -v : v) >> (sh - br);
    }
    i64 fl = tb >> br, odd = fl & 1, mod = tb - (fl << br), twice = 2 * mod, base = 1LL << br;
    return sign * (fl + (twice > base) + (twice == base) * odd);
}

struct Apu64 {
    const i64 *thr;  // C x (P+1), ceil of the float thresholds (integer inputs: a >= t <=> a >= ceil(t))
    const int *mant, *shift, *offs;
    int P, fs, mant_signed;
    Lossy64 c1, c2, out;
};

__device__ __forceinline__ i64 apu_eval64(const Apu64 &A, int c, i64 a, float *slope) {
    const i64 *t = A.thr + c * (A.P + 1);
    int idx = 0;
    for (int p = 1; p < A.P; ++p) idx += a >= __ldg(t + p);
    bool valid = HQ_EDGE_VALID(a >= __ldg(t) && a < __ldg(t + A.P));
    int o = c * A.P + idx;
    int m = __ldg(A.mant + o), sh = __ldg(A.shift + o), of = __ldg(A.offs + o);
    i64 v = valid ? round_shift64((i64)m * a, sh, 6, A.mant_signed) : 0;
    v = hw_lossy64(v, A.c1, false);
    v = hw_lossy64(v + (valid ? of : 0), A.c2, false);
    v = round_shift64(v, A.fs, A.fs, false);
    v = hw_lossy64(v, A.out, A.out.sgn);
    i64 lo = A.out.sgn ? -(1LL << (A.out.bits - 1)) + 1 : 0;
    i64 hi = A.out.sgn ? (1LL << (A.out.bits - 1)) - 1 : (1LL << A.out.bits) - 1;
    *slope = (v > lo && v < hi) ? ldexpf((float)m, -(sh + A.fs)) : 0.f;
    return v;
}

static Lossy64 make_lossy64(std::vector<int64_t> c) {
    if (c.empty()) return {0, 0, 0, 0};
    TORCH_CHECK(c[0] >= 1 && c[0] <= 40, "lossy element of ", c[0], " bits");
    return {(int)c[0], (int)c[1], (int)c[2], 1};
}

// x (N x C x HW, integer-valued fp32) -> P digit planes of `bits` bits, batch-stacked, most significant first (the
// top plane keeps the sign)
__global__ void expand_wide_k(const float *__restrict__ x, float *__restrict__ out, i64 tot, int P, int bits) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tot) return;
    int q = __float2int_rn(x[i]);
    int sh = bits * (P - 1);
    int top = q >> sh;
    out[i] = (float)top;
    int rest = q - (top << sh), mask = (1 << bits) - 1;
    for (int j = 1; j < P; ++j) out[i + j * tot] = (float)((rest >> (bits * (P - 1 - j))) & mask);
}

torch::Tensor expand_wide(torch::Tensor x, int64_t P, int64_t bits) {
    TORCH_CHECK(x.is_contiguous() && x.dtype() == torch::kFloat32);
    auto out = torch::empty({P * x.size(0), x.size(1), x.size(2), x.size(3)}, x.options());
    i64 tot = x.numel();
    if (tot)
        expand_wide_k<<<(unsigned)((tot + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), tot, (int)P, (int)bits);
    return out;
}

struct WideTail {
    const float *hi, *lo;  // conv: partial convs with the kernel's high / low byte, each (P*N) x C x HW; mock: hi = x
    const float *kq;       // mock: per-channel (or scalar) multiplier
    int kq_per_c, mock, P, bits, s;
    int *flag;
    Lossy64 acc, bcfg;
    const float *bias;     // per channel: mock = the bias; conv = the stored 16-bit i of bias = f32(f32(fac * i) * rep)
    float bias_fac, bias_rep;
    Apu64 A;
    float *y, *gs;
};

// conv: acc = sum_p sum_q 2^(bits (P-1-p) + 8 (1-q)) part[p][q], combined mod 2^32 (only valid for a 32-bit
// wrapping accumulator, checked on the host); mock: acc = lossy(round_shift(x * kq, s, 3)). Then bias, APU.
__global__ void wide_tail_k(WideTail T, i64 tot, int C, int HW) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tot) return;
    int c = (int)((i / HW) % C);
    i64 a;
    if (T.mock) {
        float k = T.kq_per_c ? __ldg(T.kq + c) : __ldg(T.kq);
        a = round_shift64((i64)__float2int_rn(T.hi[i]) * (i64)__float2int_rn(k), T.s, 3, true);
        a = hw_lossy64(a, T.acc, false);
    } else {
        unsigned ua = 0u;
        for (int p = 0; p < T.P; ++p) {
            float h = T.hi[i + p * tot], l = T.lo[i + p * tot];
            if (fabsf(h - rintf(h)) > 0.25f || fabsf(l - rintf(l)) > 0.25f) atomicOr(T.flag, 2);
            unsigned part = ((unsigned)__float2int_rn(h) << 8) + (unsigned)__float2int_rn(l);
            ua += part << (T.bits * (T.P - 1 - p));
        }
        a = (i64)(int)ua;
    }
    a = hw_lossy64(a + (i64)llrintf(__fmul_rn(__fmul_rn(__ldg(T.bias + c), T.bias_fac), T.bias_rep)), T.bcfg, false);
    float sl;
    T.y[i] = (float)apu_eval64(T.A, c, a, &sl);
    T.gs[i] = sl;
}

std::vector<torch::Tensor> wide_tail(torch::Tensor hi, torch::Tensor lo, torch::Tensor kq, int64_t mock, int64_t P,
                                     int64_t bits, int64_t s, std::vector<int64_t> acc_cfg, torch::Tensor bias,
                                     double bias_fac, double bias_rep, std::vector<int64_t> bias_cfg, torch::Tensor thr64, torch::Tensor mant,
                                     torch::Tensor shift, torch::Tensor offs, int64_t fs, bool mant_signed,
                                     std::vector<int64_t> c1, std::vector<int64_t> c2, std::vector<int64_t> out,
                                     torch::Tensor flag) {
    TORCH_CHECK(hi.is_contiguous() && hi.dtype() == torch::kFloat32 && thr64.dtype() == torch::kInt64);
    TORCH_CHECK(mock || (acc_cfg[0] == 32 && acc_cfg[1] && acc_cfg[2] && s == 0),
                "wide conv tail needs a 32-bit signed wrapping accumulator and pre_acc_shift 0");
    int N = (int)(hi.size(0) / (mock ? 1 : P)), C = (int)hi.size(1);
    i64 HW = hi.size(2) * hi.size(3), tot = (i64)N * C * HW;
    auto y = torch::empty({N, C, hi.size(2), hi.size(3)}, hi.options());
    auto gs = torch::empty_like(y);
    WideTail T;
    T.hi = hi.data_ptr<float>();
    T.lo = mock ? nullptr : lo.data_ptr<float>();
    T.kq = mock ? kq.data_ptr<float>() : nullptr;
    T.kq_per_c = mock && kq.numel() > 1;
    T.mock = (int)mock;
    T.P = (int)P;
    T.bits = (int)bits;
    T.s = (int)s;
    T.flag = flag.data_ptr<int>();
    T.acc = make_lossy64(acc_cfg);
    T.bcfg = make_lossy64(bias_cfg);
    T.bias = bias.data_ptr<float>();
    TORCH_CHECK(bias_rep >= 1 && bias_rep == (double)(i64)bias_rep && bias_fac >= 1 && bias_fac < 16777216.0,
                "bias factor / repeats must be positive integers (factor < 2^24)");
    T.bias_fac = (float)bias_fac;
    T.bias_rep = (float)bias_rep;
    T.A.thr = reinterpret_cast<const i64 *>(thr64.data_ptr<int64_t>());
    T.A.mant = mant.data_ptr<int>();
    T.A.shift = shift.data_ptr<int>();
    T.A.offs = offs.data_ptr<int>();
    T.A.P = (int)mant.size(1);
    T.A.fs = (int)fs;
    T.A.mant_signed = mant_signed;
    T.A.c1 = make_lossy64(c1);
    T.A.c2 = make_lossy64(c2);
    T.A.out = make_lossy64(out);
    T.y = y.data_ptr<float>();
    T.gs = gs.data_ptr<float>();
    if (tot)
        wide_tail_k<<<(unsigned)((tot + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(T, tot, C, (int)HW);
    return {y, gs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("expand_wide", &expand_wide);
    m.def("wide_tail", &wide_tail);
    m.def("expand", &expand);
    m.def("expand_cat", &expand_cat);
    m.def("mac_bwd", &mac_bwd);
    m.def("apu_tgrad", &apu_tgrad);
    m.def("tail", &tail);
    m.def("rshift_mul", &rshift_mul);
}
