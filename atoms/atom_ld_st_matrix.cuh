#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace ld_st_mat_atom {

    // -----------------------------------------------------------------
    // 1. Layout Policies (Address Generation Logic)
    // -----------------------------------------------------------------
    // "Stride" is always the dimension of the SLOW axis.
    // For RowMajor, Stride = Width. For ColMajor, Stride = Height.

    struct RowMajor {
        // x = row, y = col. 
        // Serializes to: row * Stride + col
        __device__ __forceinline__ static int get_offset(int x, int y, int stride) {
            return x * stride + y;
        }
    };

    struct ColMajor {
        // x = row, y = col.
        // Serializes to: col * Stride + x (User's "x + Ay" logic)
        __device__ __forceinline__ static int get_offset(int x, int y, int stride) {
            return y * stride + x;
        }
    };

    // -----------------------------------------------------------------
    // 2. The LdMatrix Atom
    // -----------------------------------------------------------------
    /**
     * @tparam NumRegs: 1 (.x1), 2 (.x2), or 4 (.x4)
     * @tparam Layout: atom::RowMajor or atom::ColMajor
     * @param dst: Array of registers (uint32_t or __nv_bfloat162)
     * Length must be >= NumRegs.
     * @param smem_base: Pointer to start of SMEM buffer (typed T*)
     * @param x: The ROW index of the tile start (e.g., warp_m * 16)
     * @param y: The COL index of the tile start (e.g., warp_n * 8)
     * @param stride: The dimension of the major axis (in Elements, not bytes)
     */
    template <int NumRegs, typename Layout, typename T>
    __device__ __forceinline__ void ldmatrix(
        void* dst_regs_void, 
        const T* smem_base, 
        int x, 
        int y, 
        int stride
    ) {
        // 1. Calculate Address (Layout Logic)
        int element_offset = Layout::get_offset(x, y, stride);
        
        // ldmatrix expects a generic shared memory pointer
        // We calculate byte offset relative to base
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_base)) 
                             + (element_offset * sizeof(T));

        // 2. Cast registers to uint32_t for ASM
        // (nv_bfloat162 is 32-bit, so this cast is safe and standard)
        uint32_t* dst = reinterpret_cast<uint32_t*>(dst_regs_void);

        // 3. The Instruction (Atom)
        // Note: No ".trans" used. Pure load.
        
        if constexpr (NumRegs == 1) {
            // .x1: Loads 1 reg per thread (8x8 tile total)
            asm volatile (
                "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                : "=r"(dst[0]) 
                : "r"(smem_addr)
            );
        }
        else if constexpr (NumRegs == 2) {
            // .x2: Loads 2 regs per thread (16x8 or 8x16)
            asm volatile (
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                : "=r"(dst[0]), "=r"(dst[1]) 
                : "r"(smem_addr)
            );
        }
        else if constexpr (NumRegs == 4) {
            // .x4: Loads 4 regs per thread (16x16)
            asm volatile (
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) 
                : "r"(smem_addr)
            );
        }
    }
    // -----------------------------------------------------------------
    // 3. The StMatrix Atom (sm_90+ Only)
    // -----------------------------------------------------------------
    /**
     * @brief Inverse of ldmatrix. Stores registers TO Shared Memory.
     * Note: stmatrix does NOT support transposition. It is a direct dump.
     *
     * @tparam NumRegs: 1 (.x1), 2 (.x2), or 4 (.x4)
     * @tparam Layout: atom::RowMajor or atom::ColMajor
     * @param src_regs_void: Source registers (must be 32-bit packed, e.g., bf16x2)
     * @param smem_base: Destination SMEM pointer
     * @param x: ROW index
     * @param y: COL index
     * @param stride: Major axis dimension
     */
    template <int NumRegs, typename Layout, typename T>
    __device__ __forceinline__ void stmatrix(
        const void* src_regs_void, // Source is const
        T* smem_base,             // Dest is mutable SMEM
        int x, 
        int y, 
        int stride
    ) {
        // 1. Calculate Address (Layout Logic)
        int element_offset = Layout::get_offset(x, y, stride);
        
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_base)) 
                             + (element_offset * sizeof(T));

        // 2. Cast registers to uint32_t for ASM (Reads 32-bit chunks)
        const uint32_t* src = reinterpret_cast<const uint32_t*>(src_regs_void);

        // 3. The Instruction
        // stmatrix writes to memory, so we add "memory" to the clobber list.
        
        if constexpr (NumRegs == 1) {
            // .x1: Stores 1 reg/thread (8x8 tile)
            asm volatile (
                "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                : // No output registers
                : "r"(smem_addr), "r"(src[0])
                : "memory"
            );
        }
        else if constexpr (NumRegs == 2) {
            // .x2: Stores 2 regs/thread (16x8 tile)
            asm volatile (
                "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
                : 
                : "r"(smem_addr), "r"(src[0]), "r"(src[1])
                : "memory"
            );
        }
        else if constexpr (NumRegs == 4) {
            // .x4: Stores 4 regs/thread (16x16 tile)
            asm volatile (
                "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                : 
                : "r"(smem_addr), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3])
                : "memory"
            );
        }
    }
}   