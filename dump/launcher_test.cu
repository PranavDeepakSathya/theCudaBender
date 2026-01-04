#include "../atoms/utils.cuh"
#include "../atoms/naive_launcher.cuh"

namespace cg = cooperative_groups;


__global__ void inspect_hierarchy_kernel()
{
    // 1. Get Handles
    auto grid    = cg::this_grid();
    auto cluster = cg::this_cluster();
    
    // 2. Query Hardware Intrinsics (The Truth)
    unsigned int cluster_size    = cluster.num_blocks();   // Size of THIS cluster
    unsigned int local_block_id  = cluster.block_rank();   // My rank inside THIS cluster (0..Size-1)
    unsigned int global_block_id = grid.block_rank();      // My rank in the whole grid (0..Total-1)
    
    // 3. Derived Metrics (The Math Verification)
    // We expect these to match the hardware intrinsics exactly
    unsigned int calc_cluster_id    = global_block_id / cluster_size; 
    unsigned int calc_local_block_id = global_block_id % cluster_size;

    // 4. Pretty Print
    // Only let the first thread of each block print to avoid spam
    if (threadIdx.x == 0) {
        printf("| Cluster: %02u | Block(Global): %03u | Block(Local): %02u | Size: %u |\n", 
               calc_cluster_id, 
               global_block_id, 
               local_block_id, 
               cluster_size);
        
        // Sanity Check: If math is wrong, yell.
        if (local_block_id != calc_local_block_id) {
            printf("  [ERROR] Mismatch! Hardware says LocalID=%u, Math says %u\n", 
                   local_block_id, calc_local_block_id);
        }
    }
}

int main() {
    // Configuration: 
    // 4 Clusters total.
    // Each Cluster has 2 Blocks.
    // Each Block has 1 Thread (just for printing).
    int grid_clusters = 4;
    int cluster_blocks = 1;
    int threads = 1;

    std::cout << "Launching Hierarchy Inspector..." << std::endl;
    std::cout << "Expectation: " << grid_clusters << " Clusters, " 
              << cluster_blocks << " Blocks/Cluster." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    NaiveLauncher launcher(grid_clusters, cluster_blocks, threads, 0);
    launcher.launch(inspect_hierarchy_kernel);

    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Done." << std::endl;
    
    return 0;
}