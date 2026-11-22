#include "ssm_conv.hpp"
#include "common.hpp"
#include "element_wise.hpp"

#include <cstdio>

using namespace sycl;

// What the conv output goes through before its store: + a 1-D channel bias, SiLU, then
// (residual + .) * scale + shift, the residual laid out like dst. Unfused launches pass {}.
struct ssm_conv_epilogue {
    bool          silu  = false;
    const float * bias  = nullptr;
    const float * res   = nullptr;
    float         scale = 1.0f;
    float         shift = 0.0f;
};

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass an empty epilogue so it folds away.
template <int DC>
static __dpct_inline__ void ssm_conv_element(
    size_t idx,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq,
    const ssm_conv_epilogue ep
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    const int token   = static_cast<int>(idx % n_t);
    const int channel = static_cast<int>((idx / n_t) % d_inner);
    const int seq     = static_cast<int>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    const float *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    float sumf = 0.0f;
    if constexpr (DC > 0) {
#pragma unroll
        for (int i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } else {
        for (int i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    if (ep.bias != nullptr) {
        sumf += ep.bias[channel];
    }
    if (ep.silu) {
        sumf = op_silu(sumf);
    }

    const size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    if (ep.res != nullptr) {
        sumf = (ep.res[dst_idx] + sumf) * ep.scale + ep.shift;
    }
    dst_data[dst_idx] = sumf;
}

// FUSED=false keeps the epilogue out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <int DC, bool FUSED>
static void kernel_ssm_conv_impl(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs __attribute__((unused)),
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq,
    const ssm_conv_epilogue ep
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    if constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    if (idx >= total_work) {
                        return;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, ep);
                }
            );
        });
    } else {
        GGML_UNUSED(ep);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    if (idx >= total_work) {
                        return;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, ssm_conv_epilogue{});
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <int DC, int TT, int TC, int WG>
static __dpct_inline__ void ssm_conv_tile(
    nd_item<1> it, local_accessor<float, 1> tile, const float *src_data, const float *weights,
    float *dst_data, int n_t, int nt_tiles, int nc_tiles, int src_stride_inner,
    int src_stride_seq, int dst_stride_token, int dst_stride_seq, const ssm_conv_epilogue ep
) {
    const int    lid = static_cast<int>(it.get_local_id(0));
    const size_t g   = it.get_group(0);
    const int    tt  = static_cast<int>(g % nt_tiles);
    const int    ct  = static_cast<int>((g / nt_tiles) % nc_tiles);
    const int    seq = static_cast<int>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    const int    t0 = tt * TT, c0 = ct * TC;

    const int ti = lid % TT;
    const int cj = lid / TT;
#pragma unroll
    for (int r = 0; r < TC / (WG / TT); ++r) {
        const int c   = cj + r * (WG / TT);
        const int tok = t0 + ti;
        float sumf = 0.0f;
        if (tok < n_t) {
            const float *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            const float *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            for (int i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            if (ep.bias != nullptr) sumf += ep.bias[c0 + c];
            if (ep.silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    const int cc = lid % TC;
    const int tj = lid / TC;
#pragma unroll
    for (int r = 0; r < TT / (WG / TC); ++r) {
        const int t   = tj + r * (WG / TC);
        const int tok = t0 + t;
        if (tok < n_t) {
            const size_t di = static_cast<size_t>(seq) * dst_stride_seq
                            + static_cast<size_t>(tok) * dst_stride_token + c0 + cc;
            float v = tile[cc * (TT + 1) + t];
            if (ep.res != nullptr) v = (ep.res[di] + v) * ep.scale + ep.shift;
            dst_data[di] = v;
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// epilogue branches: at n_t >= 32 they are amortized over the whole tile.
template <int DC, bool FUSED>
static void kernel_ssm_conv_tiled(
    queue &q, const float *src_data, const float *weights, float *dst_data,
    int d_inner, int n_t, int n_s, int src_stride_inner, int src_stride_seq,
    int dst_stride_token, int dst_stride_seq, const ssm_conv_epilogue ep
) {
    constexpr int TT = 32, TC = 32, WG = 256;
    const int nt_tiles = (n_t + TT - 1) / TT;
    const int nc_tiles = d_inner / TC;
    const size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    if constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, ep);
            });
        });
    } else {
        GGML_UNUSED(ep);

        q.submit([&](handler &h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, ssm_conv_epilogue{});
            });
        });
    }
}

static void kernel_ssm_conv(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs,
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq,
    const ssm_conv_epilogue ep
) {
    // Only the fused instantiations carry the epilogue as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    const bool fused = ep.silu || ep.bias != nullptr || ep.res != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    if (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        if (fused) {
            kernel_ssm_conv_tiled<4, true>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, ep);
        } else {
            kernel_ssm_conv_tiled<4, false>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, ep);
        }
        return;
    }

    if (d_conv == 4) {
        if (fused) {
            kernel_ssm_conv_impl<4, true>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, ep);
        } else {
            kernel_ssm_conv_impl<4, false>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, ep);
        }
        return;
    }

    if (fused) {
        kernel_ssm_conv_impl<0, true>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, ep);
    } else {
        kernel_ssm_conv_impl<0, false>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, ep);
    }
}

// out: where the result goes, dst->data unless a fused epilogue stores into a later node
inline void ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, float * out = nullptr,
                                  const ssm_conv_epilogue ep = {}) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int d_conv   = src1->ne[0];
    const int ncs      = src0->ne[0];
    const int d_inner  = src0->ne[1];
    const int n_t      = dst->ne[1];
    const int n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const int src_stride_inner = ncs;
    const int src_stride_seq   = ncs * d_inner;
    const int dst_stride_token = d_inner;
    const int dst_stride_seq   = d_inner * n_t;

    try {
        queue *q = ctx.stream();

        const float *src_data = static_cast<const float *>(src0->data);
        const float *weights  = static_cast<const float *>(src1->data);
        float *dst_data       = out ? out : static_cast<float *>(dst->data);

        GGML_ASSERT(src_data && weights && dst_data);

        kernel_ssm_conv(
            *q,
            src_data,
            weights,
            dst_data,
            d_conv,
            d_inner,
            n_t,
            n_s,
            ncs,
            src_stride_inner,
            src_stride_seq,
            dst_stride_token,
            dst_stride_seq,
            ep
        );

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
void ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    const float * bias = nullptr;
    if (add != nullptr) {
        const ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<const float *>(bias_t->data);
    }
    ssm_conv_epilogue ep;
    ep.silu = true;
    ep.bias = bias;
    ggml_sycl_op_ssm_conv(ctx, dst, static_cast<float *>(silu_dst->data), ep);
}

// Fused ssm_conv + RESHAPE + ADD + SCALE, a short conv on a residual stream (Inkling 2's
// convolve()): write (res + conv(x)) * scale + shift into the SCALE node, res laid out like the
// conv output. The conv output and the ADD result are never written.
void ggml_sycl_ssm_conv_res_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * res,
                                  ggml_tensor * scale) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ssm_conv_epilogue ep;
    ep.res   = static_cast<const float *>(res->data);
    ep.scale = ggml_get_op_params_f32(scale, 0);
    ep.shift = ggml_get_op_params_f32(scale, 1);
    ggml_sycl_op_ssm_conv(ctx, dst, static_cast<float *>(scale->data), ep);
}
