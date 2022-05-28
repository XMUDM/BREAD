#ifndef BIPPR_CUH_
#define BIPPR_CUH_

#include "PprDataD.cuh"

extern float fwdCheckTime;
extern float fwdRunTime;
extern float initTime;
extern float bwdFlagTime;
extern float bwdGetTime;
extern float bwdUpdateTime;
extern float bwdCpyTime;
extern float arrowTime;
extern float khopGapTime;

__global__ void UpdateNextFrontierFlags(float *d_residuals, unsigned char *d_new_frontier_flags, double alpha, double r_max, int numVertices) {
    int block_begin_pos = blockDim.x * blockIdx.x;
    while(block_begin_pos < numVertices) {
        int idx = block_begin_pos + threadIdx.x;
        if(idx < numVertices) {
            // float residual = d_residuals[idx];
            if(d_residuals[idx] >= r_max) {
                d_new_frontier_flags[idx] = 1;
            }
        }
        block_begin_pos += blockDim.x * gridDim.x;
    }
}

void GetNextFrontiers(void *d_temp_storage_for_flags, size_t temp_storage_bytes_for_flags, int *d_vertices, unsigned char *d_new_frontier_flags, int *d_frontiers, int *d_frontiers_cnt, int numVertices) {
    cub::DeviceSelect::Flagged(d_temp_storage_for_flags, temp_storage_bytes_for_flags, d_vertices, d_new_frontier_flags, d_frontiers, d_frontiers_cnt, numVertices);
}

__global__ void UpdateResidualsBlock(int* d_csr_begins, int* d_csr_adjs, int* d_csr_times, float *d_residuals, int *d_frontiers, int *d_frontiers_cnt, unsigned char *d_visited, int *d_distance,
int endTime, double alpha, bool isDynamic, bool isHop) {
    int frontier_idx = blockIdx.x;
    int frontiers_offset_grid = gridDim.x;
    int total_frontier_cnt = *d_frontiers_cnt;
    int block_size = blockDim.x;
    int thread_id = threadIdx.x;

    int frontier;
    int frontier_degree = 0;
    float neighbor_residual_inc = 0.0;
    int neighbor_begin_itr = 0, neighbor_end_pos = 0;
    
    __shared__ float shared_frontier_residual[1];

    while(frontier_idx < total_frontier_cnt) {
        frontier = d_frontiers[frontier_idx];
        if(thread_id == 0) {
            shared_frontier_residual[0] = atomicExch(d_residuals + frontier, 0.0);
        }
        __syncthreads();
        
        neighbor_begin_itr = d_csr_begins[frontier];
		neighbor_end_pos = d_csr_begins[frontier + 1]; //exclusive
		frontier_degree = neighbor_end_pos - neighbor_begin_itr;

        if (frontier_degree == 0) {
			if (thread_id == 0) atomicAdd(d_residuals + frontier, (1.0 - alpha) * shared_frontier_residual[0]);
		}
        if (frontier_degree != 0) {
            neighbor_residual_inc = (1.0 - alpha) * shared_frontier_residual[0] / frontier_degree;
            int neighbor_offset = 0;
            while (neighbor_offset < frontier_degree) {
                int neighbor_pos = neighbor_begin_itr + neighbor_offset + thread_id;
                if (neighbor_pos < neighbor_end_pos) {
                    if(isDynamic) {
                        int startTime = d_csr_times[neighbor_pos];
                        if(startTime <= endTime){
                            atomicAdd(d_residuals + d_csr_adjs[neighbor_pos], neighbor_residual_inc);
                            if(isHop) {
                                if(d_distance[d_csr_adjs[neighbor_pos]] > d_distance[frontier]+1) {
                                    d_distance[d_csr_adjs[neighbor_pos]] = d_distance[frontier]+1;
                                }
                            }
                        }
                    }else{
                        atomicAdd(d_residuals + d_csr_adjs[neighbor_pos], neighbor_residual_inc);
                    }
                }
                neighbor_offset += block_size;
            }
        }
        frontier_idx += frontiers_offset_grid;
    }
}

__global__ void reset_frontiers_flag(unsigned char* d_new_frontier_flags, int numVertices) {
    int block_begin_pos = blockDim.x * blockIdx.x;
    while(block_begin_pos < numVertices) {
        int idx = block_begin_pos + threadIdx.x;
        if(idx < numVertices) {
            d_new_frontier_flags[idx] = 0;
        }
        block_begin_pos += blockDim.x * gridDim.x;
    }
}


void bwd_search_parallel(GraphD *graph, GraphD *graphR, PprDataD *ppr, Config* config, Query *q, int numVertices) {
    int* frontiers_cnt = (int*)malloc(sizeof(int));
    gpuErrchk(cudaMemcpy(frontiers_cnt, ppr->d_frontiers_cnt, sizeof(int), cudaMemcpyDeviceToHost));
    int block_size = 512;
    cudaEvent_t Start, Stop;
    gpuErrchk(cudaEventCreate(&Start));
    gpuErrchk(cudaEventCreate(&Stop));
    float cudaEventRecordTime = 0.0;
    // cout << "Begin backward search" << endl;
    while(1) {
        // cout << frontiers_cnt[0] << ",";

        int num_blocks = calc_num_blocks(graph->numVertices, block_size);
        // cout << "UpdateNextFrontierFlags" << endl;
        gpuErrchk(cudaEventRecord(Start, 0));
        reset_frontiers_flag<<<num_blocks, block_size>>>(ppr->d_new_frontier_flags, ppr->numVertices);
        UpdateNextFrontierFlags<<<num_blocks, block_size>>>(ppr->d_residuals, ppr->d_new_frontier_flags, config->alpha, config->bwd_delta, ppr->numVertices);
        gpuErrchk(cudaEventRecord(Stop, 0));
        gpuErrchk(cudaEventSynchronize(Stop));
        gpuErrchk(cudaEventElapsedTime(&cudaEventRecordTime, Start, Stop));
        bwdFlagTime += cudaEventRecordTime / 1000;

        // cout << "GetNextFrontiers" << endl;
        gpuErrchk(cudaEventRecord(Start, 0));
        GetNextFrontiers(ppr->d_temp_storage, ppr->temp_stroage_bytes, ppr->d_vertices, ppr->d_new_frontier_flags, ppr->d_frontiers, ppr->d_frontiers_cnt, ppr->numVertices);
        gpuErrchk(cudaEventRecord(Stop, 0));
        gpuErrchk(cudaEventSynchronize(Stop));
        gpuErrchk(cudaEventElapsedTime(&cudaEventRecordTime, Start, Stop));
        bwdGetTime += cudaEventRecordTime / 1000;

        gpuErrchk(cudaEventRecord(Start, 0));
        gpuErrchk(cudaMemcpy(frontiers_cnt, ppr->d_frontiers_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaEventRecord(Stop, 0));
        gpuErrchk(cudaEventSynchronize(Stop));
        gpuErrchk(cudaEventElapsedTime(&cudaEventRecordTime, Start, Stop));
        bwdCpyTime += cudaEventRecordTime / 1000;

        if(frontiers_cnt[0] == 0) break;

        num_blocks = calc_num_blocks(frontiers_cnt[0] * block_size, block_size);

        // cout << "UpdateResidualsBlock" << endl;
        gpuErrchk(cudaEventRecord(Start, 0));
        UpdateResidualsBlock<<<num_blocks, block_size>>>(graphR->d_csr_begins, graphR->d_csr_adjs, graphR->d_csr_times, ppr->d_residuals, 
        ppr->d_frontiers, ppr->d_frontiers_cnt, ppr->d_visited, ppr->d_distance, q->endTime, config->alpha, config->dynamic, config->hop);
        gpuErrchk(cudaEventRecord(Stop, 0));
        gpuErrchk(cudaEventSynchronize(Stop));
        gpuErrchk(cudaEventElapsedTime(&cudaEventRecordTime, Start, Stop));
        bwdUpdateTime += cudaEventRecordTime / 1000;

        gpuErrchk(cudaMemset(ppr->d_new_frontier_flags, 0, sizeof(unsigned char) * ppr->numVertices));

    }
    gpuErrchk(cudaEventDestroy(Start));
    gpuErrchk(cudaEventDestroy(Stop));
    if(frontiers_cnt) free(frontiers_cnt);
}

__device__ int random_walk_last_reachability(int source, int endTime, int hop, int* d_csr_begins, int* d_csr_times, int* d_csr_adjs, int* d_lengths, float* d_residuals,
 int *d_distance, unsigned char *d_result, int idx, double alpha, bool isDynamic, bool isHop) {
    int cur = source;
    int neighbor_begin_itr = 0, neighbor_end_pos = 0;
    int degree = 0;
    int length = 0;

    curandState state;
    curand_init((unsigned long long)clock(), 0, 0, &state);
    while (true) {
        double rand01 = curand_uniform_double(&state);
        if (rand01 < alpha) {
            break;
        }
        neighbor_begin_itr = d_csr_begins[cur];
		neighbor_end_pos = d_csr_begins[cur + 1];
		degree = neighbor_end_pos - neighbor_begin_itr;

        int numPresentEdges = 0;
        int next = cur;
        for(int i = 0; i < degree; i++) {
            int neighbor_pos = neighbor_begin_itr + i;
            int startTime = d_csr_times[neighbor_pos];
            int nodeId = d_csr_adjs[neighbor_pos];
            if(isDynamic) {
                if(startTime <= endTime) {
                    numPresentEdges += 1;
                    int z = curand(&state) % numPresentEdges;
                    if(z == 0) {
                        next = nodeId;
                    }
                }
            }else{
                numPresentEdges += 1;
                int z = curand(&state) % numPresentEdges;
                if(z == 0) {
                    next = nodeId;
                }
            }
        }
        if(isDynamic) {
            if(isHop) {
                if(d_residuals[cur] > 0.0 && length + d_distance[cur] <= hop) {
                    d_result[0] = 1;
                    break;
                }
            }
            else{
                if(d_residuals[cur] > 0.0) {
                    d_result[0] = 1;
                    break;
                }
            }
        }
        cur = next;
        length++;
    }
    d_lengths[idx] = length;
    return cur;
}

__global__ void sample_fwd_reachability(int source, int endTime, int hop, int *d_csr_begins, int *d_csr_times, int *d_csr_adjs, 
int *d_record, int *d_lengths, float* d_residuals, int *d_distance, unsigned char *d_result, int num_tries, double alpha, bool isDynamic, bool isHop) {
    int block_begin_pos = blockDim.x * blockIdx.x;
    while(block_begin_pos < num_tries) {
        int idx = block_begin_pos + threadIdx.x;
        if(idx < num_tries) {
            int l = random_walk_last_reachability(source, endTime, hop, d_csr_begins, d_csr_times, d_csr_adjs, d_lengths, 
            d_residuals, d_distance, d_result, idx, alpha, isDynamic, isHop);
            d_record[idx] = l;
        }
        block_begin_pos += blockDim.x * gridDim.x;
    }
}

__global__ void check(float* d_residuals, int *d_record, int *d_lengths, int *d_distance, int *d_ans,
unsigned char* d_result, int num_tries, int index, int hop, bool isHop) {
    int block_begin_pos = blockDim.x * blockIdx.x;
    while(block_begin_pos < num_tries) {
        int idx = block_begin_pos + threadIdx.x;
        if(idx == 0) {
            if(d_result[idx] == 1) d_ans[index] = 1;
        }
        block_begin_pos += blockDim.x * gridDim.x;
    }
}

__global__ void init(int* d_frontiers_cnt, float *d_residuals, int *d_vertices, unsigned char *d_new_frontier_flags, unsigned char *d_visited, 
unsigned char* d_result, int *d_frontiers, int *d_distance, int target, int numVertices) {
    int block_begin_pos = blockDim.x * blockIdx.x;
    while(block_begin_pos < numVertices) {
        int idx = block_begin_pos + threadIdx.x;
        if(idx < numVertices) {
            if(idx == 0) {
                d_frontiers_cnt[idx] = 1;
                d_frontiers[idx] = target;
                d_result[idx] = 0;
            }
            if(idx == target) {
                d_residuals[target] = 1.0;
                d_distance[target] = 0;
                d_visited[target] = 1;
            }
            if(idx != target) {
                d_residuals[idx] = 0.0;
                d_distance[idx] = 10000;
            }
            d_vertices[idx] = idx;
            d_new_frontier_flags[idx] = 0;
            d_visited[idx] = 0;
            if(idx != 0) d_frontiers[idx] = 0;
        }
        block_begin_pos += blockDim.x * gridDim.x;
    }
}

__global__ void init_rw(int* d_record, int num_tries) {
    int block_begin_pos = blockDim.x * blockIdx.x;
    while(block_begin_pos < num_tries) {
        int idx = block_begin_pos + threadIdx.x;
        if(idx < num_tries) {
            d_record[idx] = 0;
        }
        block_begin_pos += blockDim.x * gridDim.x;
    }
}

void fwd_walk_parallel(Config* config, Query *q, PprDataD* ppr, GraphD* graph) {
    int block_size = 512;
    int num_blocks = calc_num_blocks(ppr->num_tries, block_size);
    cudaEvent_t Start, Stop;
    gpuErrchk(cudaEventCreate(&Start));
    gpuErrchk(cudaEventCreate(&Stop));
    float cudaEventRecordTime = 0.0;

    gpuErrchk(cudaEventRecord(Start, 0));
    init_rw<<<num_blocks, block_size>>>(ppr->d_record, ppr->num_tries);
    sample_fwd_reachability<<<num_blocks, block_size>>>(q->source, q->endTime, q->hop, graph->d_csr_begins, graph->d_csr_times, graph->d_csr_adjs, 
    ppr->d_record, ppr->d_lengths, ppr->d_residuals, ppr->d_distance, ppr->d_result, ppr->num_tries, config->fwd_alpha, config->dynamic, config->hop);
    gpuErrchk(cudaEventRecord(Stop, 0));
    gpuErrchk(cudaEventSynchronize(Stop));
    gpuErrchk(cudaEventElapsedTime(&cudaEventRecordTime, Start, Stop));
    fwdRunTime += cudaEventRecordTime / 1000;

    gpuErrchk(cudaEventRecord(Start, 0));
    check<<<num_blocks, block_size>>>(ppr->d_residuals, ppr->d_record, ppr->d_lengths, ppr->d_distance, ppr->d_ans, ppr->d_result, 
    ppr->num_tries, q->index, q->hop, config->hop);
    gpuErrchk(cudaEventRecord(Stop, 0));
    gpuErrchk(cudaEventSynchronize(Stop));
    gpuErrchk(cudaEventElapsedTime(&cudaEventRecordTime, Start, Stop));
    fwdCheckTime += cudaEventRecordTime / 1000;

    gpuErrchk(cudaEventDestroy(Start));
    gpuErrchk(cudaEventDestroy(Stop));
}

void bread(GraphD* graph, GraphD* graphR, PprDataD* ppr, Config *config, Query *q, int numVertices) {
    cudaEvent_t Start, Stop;
    gpuErrchk(cudaEventCreate(&Start));
    gpuErrchk(cudaEventCreate(&Stop));
    float cudaEventRecordTime = 0.0;
    gpuErrchk(cudaEventRecord(Start, 0));

    int block_size = 512;
    int num_blocks = calc_num_blocks(numVertices, block_size);
    init<<<num_blocks, block_size>>>(ppr->d_frontiers_cnt, ppr->d_residuals, ppr->d_vertices, ppr->d_new_frontier_flags, 
    ppr->d_visited, ppr->d_result, ppr->d_frontiers, ppr->d_distance, q->target, numVertices);

    bwd_search_parallel(graph, graphR, ppr, config, q, numVertices);

    fwd_walk_parallel(config, q, ppr, graph);

    gpuErrchk(cudaEventRecord(Stop, 0));
    gpuErrchk(cudaEventSynchronize(Stop));
    gpuErrchk(cudaEventElapsedTime(&cudaEventRecordTime, Start, Stop));
    khopGapTime += cudaEventRecordTime / 1000;
}

__global__ void stops_reset(int* stops, int numStops) {
    int block_begin_pos = blockDim.x * blockIdx.x;
    while(block_begin_pos < numStops) {
        int idx = block_begin_pos + threadIdx.x;
        if(idx < numStops) {
            stops[idx] = -1;
        }
        block_begin_pos += blockDim.x * gridDim.x;
    }
}

#endif