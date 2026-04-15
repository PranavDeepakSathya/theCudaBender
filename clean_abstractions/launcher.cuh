#pragma once
#include "utils.cuh"
#include <cuda_runtime.h>

class NaiveLauncher {
public:
  int grid_size_clusters;   
  int cluster_size_blocks;  
  int block_size_threads;   
  int smem_bytes;           

  NaiveLauncher(int g_clusters, int c_blocks, int b_threads, int smem = 0) 
    : grid_size_clusters(g_clusters), 
      cluster_size_blocks(c_blocks), 
      block_size_threads(b_threads), 
      smem_bytes(smem) {}

  template <typename KernelFunc, typename... KernelArgs>
  void launch(KernelFunc kernel, KernelArgs... args) {
      
    int total_blocks = grid_size_clusters * cluster_size_blocks;

    cudaLaunchConfig_t config = {0};
    config.gridDim = total_blocks;
    config.blockDim = block_size_threads;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = 0; 

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size_blocks;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.attrs = attribute;
    config.numAttrs = 1;

    CHECK_CUDA(cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared
    ));

    CHECK_CUDA(cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes
    ));

    CHECK_CUDA(cudaLaunchKernelEx(&config, kernel, args...));
  }
};