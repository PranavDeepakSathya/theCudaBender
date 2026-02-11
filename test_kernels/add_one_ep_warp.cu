#include "../atoms/all.cuh"
constexpr int tm = 4;
constexpr int tn = 4; 
constexpr int tpw_n = 8;
constexpr int tpw_m = 32/tpw_n; 
constexpr int WM = tm*tpw_m; 
constexpr int WN = tn*tpw_n; 
constexpr int wpb_m = 4;
constexpr int wpb_n = 4; 
constexpr int load_warp_id = wpb_m*wpb_n; 
constexpr int store_warp_id = load_warp_id + 1; 
constexpr int threads_per_block = wpb_m*wpb_n; 
constexpr int BM = WM*wpb_m; 
constexpr int BN = WN*wpb_n; 
constexpr int num_load_stages = 3;
constexpr int num_store_stages = 2; 
constexpr uint32_t tile_size = BM*BN*sizeof(nv_bfloat16); 
constexpr uint32_t shared_bytes = (num_load_stages + num_store_stages)*tile_size + (4*1024); 
constexpr int BM_iters = 2; 
constexpr int BN_iters = 2; 
constexpr int SM = BM*BM_iters; 
constexpr int SN = BN*BN_iters; 
constexpr int M = 8192; 
constexpr int N = 8192; 
constexpr int GM = M/SM; 
constexpr int GN = N/SN; 


__global__ void add_one (__grid_constant__ const CUtensorMap a_map, __grid_constant__ const CUtensorMap b_map)
{
  
}