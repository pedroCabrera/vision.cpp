#include "nn.h"
#include "util/string.h"

namespace visp {

tensor ensure_tensor_type(model_ref m, tensor t, ggml_type type) {
    if (!t || t->type == type) {
        return t;
    }
    return ggml_cast(m, t, type);
}

tensor linear(model_ref m, tensor x) {
    tensor weight = ensure_tensor_type(m, m.weights("weight"), x->type);

    auto shape = nelements(x);
    const bool needs_flatten = (shape[2] > 1 || shape[3] > 1);
    if (needs_flatten) {
        int64_t columns = shape[1] * shape[2] * shape[3];
        x = ggml_cont(m, ggml_reshape_2d(m, x, x->ne[0], columns));
    }

    x = ggml_mul_mat(m, weight, x);

    if (needs_flatten) {
        x = ggml_cont(m, ggml_reshape_4d(m, x, x->ne[0], shape[1], shape[2], shape[3]));
    }

    if (tensor bias = m.find("bias")) {
        bias = ensure_tensor_type(m, bias, x->type);

        // Avoid CUDA grid-y overflow for very wide 2D tensors by swapping the fast axes
        // so the large extent sits on dimension 0 where the kernel uses block.x instead of grid.y.
        if (x->ne[1] > 65535 && x->ne[2] == 1 && x->ne[3] == 1) {
            auto find_divisor = [](int64_t n) {
                int64_t min_d = (n + 65535 - 1) / 65535;
                for (int64_t d = std::max<int64_t>(min_d, 1); d <= 65535; ++d) {
                    if (n % d == 0) {
                        return d;
                    }
                }
                return int64_t(0);
            };

            int64_t divisor = find_divisor(x->ne[1]);
            ASSERT(divisor != 0, "Unable to factor tensor for bias addition");
            int64_t dim1 = divisor;
            int64_t dim2 = x->ne[1] / divisor;
            ASSERT(dim1 <= 65535 && dim2 <= 65535, "Dimension factoring did not respect CUDA limits");

            tensor bias_view = bias;
            if (!(bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1)) {
                bias_view = ggml_cont(m, ggml_reshape_4d(m, bias, bias->ne[0], 1, 1, 1));
            }

            tensor x_factored = ggml_cont(m, ggml_reshape_4d(m, x, x->ne[0], dim1, dim2, 1));
            x_factored = ggml_add_inplace(m, x_factored, bias_view);
            x = ggml_cont(m, ggml_reshape_4d(m, x_factored, x->ne[0], x->ne[1], 1, 1));
        } else {
            x = ggml_add_inplace(m, x, bias);
        }
    }
    return x;
}

tensor layer_norm(model_ref m, tensor x, float eps) {
    x = ggml_norm(m, x, eps);
    tensor weight = ensure_tensor_type(m, m.weights("weight"), x->type);
    tensor bias = ensure_tensor_type(m, m.weights("bias"), x->type);
    // Avoid CUDA grid-y overflow for very wide 2D tensors by factoring dim1 into two dims
    if (m.backend == backend_type::cuda && x->ne[1] > 65535 && x->ne[2] == 1 && x->ne[3] == 1) {
        auto find_divisor = [](int64_t n) {
            int64_t min_d = (n + 65535 - 1) / 65535;
            for (int64_t d = std::max<int64_t>(min_d, 1); d <= 65535; ++d) {
                if (n % d == 0) return d;
            }
            return int64_t(0);
        };
        int64_t divisor = find_divisor(x->ne[1]);
        ASSERT(divisor != 0, "Unable to factor tensor for layer_norm scale/add");
        int64_t dim1 = divisor;
        int64_t dim2 = x->ne[1] / divisor;
        ASSERT(dim1 <= 65535 && dim2 <= 65535, "Dimension factoring did not respect CUDA limits");

        tensor weight_view = weight;
        tensor bias_view = bias;
        if (!(weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1)) {
            weight_view = ggml_cont(m, ggml_reshape_4d(m, weight, weight->ne[0], 1, 1, 1));
        }
        if (!(bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1)) {
            bias_view = ggml_cont(m, ggml_reshape_4d(m, bias, bias->ne[0], 1, 1, 1));
        }

        tensor x_factored = ggml_cont(m, ggml_reshape_4d(m, x, x->ne[0], dim1, dim2, 1));
        x_factored = ggml_mul_inplace(m, x_factored, weight_view);
        x_factored = ggml_add_inplace(m, x_factored, bias_view);
        x = ggml_cont(m, ggml_reshape_4d(m, x_factored, x->ne[0], x->ne[1], 1, 1));
    } else {
        x = ggml_mul_inplace(m, x, weight);
        x = ggml_add_inplace(m, x, bias);
    }
    return named(m, x);
}

tensor permute_cwhn_to_whcn(model_ref m, tensor x) {
    return ggml_permute(m, x, 2, 0, 1, 3);
}

tensor permute_whcn_to_cwhn(model_ref m, tensor x) {
    return ggml_permute(m, x, 1, 2, 0, 3);
}

std::array<int64_t, 4> nelements_whcn(model_ref const& m, tensor t) {
    auto ne = nelements(t);
    return (m.flags & model_build_flag::cwhn) ? std::array{ne[1], ne[2], ne[0], ne[3]} : ne;
}

tensor cwhn_to_contiguous_2d(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return x; // preferred 2D layout is CWHN too
    }
    return ggml_cont(m, permute_cwhn_to_whcn(m, x));
}

tensor whcn_to_contiguous_2d(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return ggml_cont(m, permute_whcn_to_cwhn(m, x));
    }
    return x;
}

tensor contiguous_2d_to_cwhn(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return x; // x is already CWHN
    }
    return ggml_cont(m, permute_whcn_to_cwhn(m, x));
}

tensor contiguous_2d_to_whcn(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    return x;
}

tensor add_bias_2d(model_ref m, tensor x) {
    if (tensor bias = m.find("bias")) {
        bias = ensure_tensor_type(m, bias, x->type);
        if (!(m.flags & model_build_flag::cwhn)) {
            bias = ggml_cont(m, ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1));
        }
        x = ggml_add_inplace(m, x, bias);
    }
    return x;
}

tensor conv_2d(model_ref m, tensor x, int stride, int pad) {
    tensor weight = ensure_tensor_type(m, m.weights("weight"), x->type);

    if (m.flags & model_build_flag::cwhn) {
        if ((m.flags & model_build_flag::conv_2d_direct_cwhn) &&
            weight->ne[1] == 1 && weight->ne[2] == 1 && stride == 1) {
            auto [c, w, h, b] = nelements(x);
            weight = ggml_cont(m, ggml_reshape_2d(m, weight, weight->ne[0], weight->ne[3]));
            x = ggml_cont(m, ggml_reshape_2d(m, x, x->ne[0], w * h * b));
            x = ggml_mul_mat(m, weight, x);
            x = ggml_cont(m, ggml_reshape_4d(m, x, weight->ne[1], w, h, b));

        } else if (m.flags & model_build_flag::conv_2d_direct_cwhn) { 
            weight = permute_cwhn_to_whcn(m, weight);
            x = permute_cwhn_to_whcn(m, x);
            x = ggml_conv_2d_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
            x = permute_whcn_to_cwhn(m, x);

        } else {
            weight = ggml_cont(m, permute_cwhn_to_whcn(m, weight));
            x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
            x = ggml_conv_2d(m, weight, x, stride, stride, pad, pad, 1, 1);
            x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
        }
    } else { // WHCN layout
        x = ggml_conv_2d_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_2d_depthwise(model_ref m, tensor x, int stride, int pad) {
    tensor weight = ensure_tensor_type(m, m.weights("weight"), x->type);

    if (m.flags & model_build_flag::cwhn) {
        weight = ggml_cont(m, ggml_permute(m, weight, 3, 2, 0, 1));
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
        x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
        x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    } else {
        x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_transpose_2d(model_ref m, tensor x, int stride) {
    tensor weight = m.weights("weight");
    if (m.backend == backend_type::cuda) {
        if (x->type != GGML_TYPE_F32) {
            x = ggml_cast(m, x, GGML_TYPE_F32);
        }
        if (weight->type != GGML_TYPE_F16) {
            weight = ggml_cast(m, weight, GGML_TYPE_F16);
        }
    } else if (m.flags & model_build_flag::f16_conv_transpose) {
        // TODO: ggml_conv_transpose_2d_p0 expects fp16 weights (cpu backend)
        weight = ggml_cast(m, weight, GGML_TYPE_F16);
    }
    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    x = ggml_conv_transpose_2d_p0(m, weight, x, stride);

    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_2d_deform(
    model_ref m, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad) {

    ggml_type input_type = x->type;
    const bool force_fp32 = (m.backend == backend_type::cuda && input_type == GGML_TYPE_F16);
    const ggml_type working_type = force_fp32 ? GGML_TYPE_F32 : input_type;

    if (force_fp32) {
        // CUDA backend benefits from higher precision for deformable conv; cast activations up.
        x = ggml_cast(m, x, GGML_TYPE_F32);
    }

    weight = ensure_tensor_type(m, weight, working_type);
    offset = ensure_tensor_type(m, offset, working_type);
    if (mask) {
        mask = ensure_tensor_type(m, mask, working_type);
    }

    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
        weight = ggml_cont(m, permute_cwhn_to_whcn(m, weight));
        offset = ggml_cont(m, permute_cwhn_to_whcn(m, offset));
        if (mask) {
            mask = ggml_cont(m, permute_cwhn_to_whcn(m, mask));
        }
    }

    x = ggml_conv_2d_deform(m, weight, x, offset, mask, stride, stride, pad, pad);

    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    }

    if (force_fp32) {
        x = ggml_cast(m, x, input_type);
    }
    return x;
}

tensor batch_norm_2d(model_ref m, tensor x) {
    // Batch norm is expected to be have been fused into mul+add. See convert.py
    ASSERT(m.find("running_mean") == nullptr, "Batch norm was not fused");
    ASSERT(m.find("running_var") == nullptr, "Batch norm was not fused");

    ggml_type in_type = x->type;
    auto is_fp = [](ggml_type t) { return t == GGML_TYPE_F16 || t == GGML_TYPE_F32; };
    bool promote = (m.backend == backend_type::cuda) && is_fp(in_type) && in_type != GGML_TYPE_F32;
    if (promote) {
        x = ggml_cast(m, x, GGML_TYPE_F32);
    }

    tensor weight = ensure_tensor_type(m, m.weights("weight"), x->type);
    tensor bias = ensure_tensor_type(m, m.weights("bias"), x->type);
    if (!(m.flags & model_build_flag::cwhn)) { // WHCN layout
        weight = ggml_cont(m, ggml_reshape_4d(m, weight, 1, 1, weight->ne[0], 1));
        bias = ggml_cont(m, ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1));
    }
    x = ggml_mul_inplace(m, x, weight);
    x = ggml_add_inplace(m, x, bias);
    if (promote) {
        x = ggml_cast(m, x, in_type);
    }
    return named(m, x);
}

tensor patch_embed(model_ref m, tensor x, int patch_size) {
    ASSERT(x->ne[1] % patch_size == 0 && x->ne[2] % patch_size == 0);
    char const* proj = m.find("proj.weight") ? "proj" : "projection";

    m.flags |= model_build_flag::cwhn;
    x = conv_2d(m[proj], x, patch_size);

    if (m.find("norm.weight")) {
        auto [c, w, h, b] = nelements(x);
        x = ggml_cont(m, ggml_reshape_3d(m, x, c, w * h, b));
        x = layer_norm(m["norm"], x);
        x = ggml_cont(m, ggml_reshape_4d(m, x, c, w, h, b));
    }
    return named(m, x);
}

} // namespace visp