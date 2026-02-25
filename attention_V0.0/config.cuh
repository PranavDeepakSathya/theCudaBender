#pragma once

#include <cuda_bf16.h>

struct AttnConfig
{
    // Warp tile
    static constexpr int w_d  = 16;
    static constexpr int w_lq = 16;
    static constexpr int w_lk = 8;

    static constexpr int D  = w_d;
    static constexpr int LQ = w_lq;
    static constexpr int LK = w_lk;

    static constexpr int w_Q_bytes = w_d*w_lq*sizeof(nv_bfloat16);
    static constexpr int w_K_bytes = w_d*w_lq*sizeof(nv_bfloat16);
    static constexpr int w_S_bytes = w_lq*w_lk*sizeof(nv_bfloat16);
    static constexpr int w_P_bytes = w_lq*w_lk*sizeof(nv_bfloat16);
    static constexpr int w_O_bytes = w_d*w_lq*sizeof(nv_bfloat16);

    static constexpr int block_size = 32;
    static constexpr int grid_size  = 1;

    // single warp → minimal shared
    static constexpr int shared_bytes =
        w_K_bytes + w_Q_bytes;  // adjust when you finalize layout
};