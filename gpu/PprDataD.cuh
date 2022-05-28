#ifndef PPRDATAD_H_
#define PPRDATAD_H_

#include "PprData.cuh"
#include "GraphD.cuh"
#include "cub/cub/cub.cuh"

class PprDataD {
public:
	int source = -1, target = -1;
	int startTime = 0, endTime = 0;
	int numVertices = 0;
	int* d_frontiers_cnt = NULL;
	int* d_record = NULL;
	int* d_lengths = NULL;
	int* d_distance = NULL;
	int* d_ans = NULL;
    int* d_frontiers = NULL;
    float* d_residuals = NULL;
    unsigned char* d_new_frontier_flags = NULL;
	unsigned char* d_visited = NULL;
	unsigned char* d_result = NULL;
    int* d_vertices = NULL;
	void *d_temp_storage;
    size_t temp_stroage_bytes;
	int num_tries = 0;
	int query_size = 0;

	PprDataD(PprData* p){
		CudaInitPpr(p);
	}

	void CudaInitPpr(PprData* p) {
		this->source = p->source;
		this->target = p->target;
		this->startTime = p->startTime;
		this->endTime = p->endTime;
		this->numVertices = p->numVertices;
		this->num_tries = p->num_tries;
		this->query_size = p->query_size;
		
		d_temp_storage = NULL;
		temp_stroage_bytes = 0;
		cub::DeviceSelect::Flagged(d_temp_storage, temp_stroage_bytes, d_vertices, d_new_frontier_flags, 
    d_frontiers, d_frontiers_cnt, numVertices);
		cout << "allocated temp_stroage for cub::DeviceSelect::Flagged: " << temp_stroage_bytes << endl;
    	gpuErrchk(cudaMalloc(&d_temp_storage, temp_stroage_bytes));

		gpuErrchk(cudaMalloc(&d_frontiers_cnt, sizeof(int)));
		gpuErrchk(cudaMalloc(&d_record, num_tries * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_lengths, num_tries * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_ans, query_size * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_frontiers, numVertices * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_distance, numVertices * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_residuals, numVertices * sizeof(float)));
		gpuErrchk(cudaMalloc(&d_new_frontier_flags, numVertices * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc(&d_visited, numVertices * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc(&d_result, sizeof(unsigned char)));
		gpuErrchk(cudaMalloc(&d_vertices, numVertices * sizeof(int)));

		gpuErrchk(cudaMemcpy(d_frontiers_cnt, p->frontiers_cnt, sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_frontiers, p->frontiers, numVertices * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_distance, p->distance, numVertices * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_record, p->record, num_tries * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_lengths, p->lengths, num_tries * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_ans, p->ans, query_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_residuals, p->residuals, numVertices * sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_new_frontier_flags, p->new_frontier_flags, numVertices * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_visited, p->visited, numVertices * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_result, p->result, sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_vertices, p->vertices, numVertices * sizeof(int), cudaMemcpyHostToDevice));
	}

	~PprDataD() {
		FreeGPUMemory();
	}

	void FreeGPUMemory() {
		if(d_temp_storage) gpuErrchk(cudaFree(d_temp_storage));
		if(d_frontiers_cnt) gpuErrchk(cudaFree(d_frontiers_cnt));
		if(d_frontiers) gpuErrchk(cudaFree(d_frontiers));
		if(d_record) gpuErrchk(cudaFree(d_record));
		if(d_ans) gpuErrchk(cudaFree(d_ans));
		if(d_residuals) gpuErrchk(cudaFree(d_residuals));
		if(d_vertices) gpuErrchk(cudaFree(d_vertices));
		if(d_new_frontier_flags) gpuErrchk(cudaFree(d_new_frontier_flags));
		if(d_visited) gpuErrchk(cudaFree(d_visited));
		if(d_result) gpuErrchk(cudaFree(d_result));
	}
};

#endif