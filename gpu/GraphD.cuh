#ifndef GRAPHD_H_
#define GRAPHD_H_

#include "Graph.h"
#include "CudaUtil.cuh"

class GraphD {
public:
	int *d_csr_adjs = NULL;
    int *d_csr_times = NULL;
	int *d_csr_begins = NULL;
	int *d_degrees = NULL;
	int numVertices = 0, numEdges = 0;

	GraphD(Graph* g) {
		CudaInitGraph(g);
	}

	~GraphD() {
		FreeGPUMemory();
	}

	void CudaInitGraph(Graph* g) {
		this->numVertices = g->numVertices;
		this->numEdges = g->numEdges;
        gpuErrchk(cudaMalloc(&d_csr_adjs, numEdges * sizeof(int)));
        gpuErrchk(cudaMalloc(&d_csr_times, numEdges * sizeof(int)));
        gpuErrchk(cudaMalloc(&d_csr_begins, (numVertices + 1) * sizeof(int)));
        gpuErrchk(cudaMalloc(&d_degrees, numVertices * sizeof(int)));

        gpuErrchk(cudaMemcpy(d_csr_adjs, g->csr_adjs, numEdges * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csr_times, g->csr_times, numEdges * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_csr_begins, g->csr_begins, (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_degrees, g->degrees, numVertices * sizeof(int), cudaMemcpyHostToDevice));
    }

	void FreeGPUMemory() {
		if (d_csr_adjs) gpuErrchk(cudaFree(d_csr_adjs));
        if (d_csr_times) gpuErrchk(cudaFree(d_csr_times));
		if (d_csr_begins) gpuErrchk(cudaFree(d_csr_begins));
		if (d_degrees) gpuErrchk(cudaFree(d_degrees));
	}
};

#endif

// class d_Node {
// public:
// 	int nodeId;
// 	Edge* presentEdges;
// 	int size;

// 	d_Node();

// 	d_Node(int isNodeId, int size);

// 	int getIndexOfEdge(int destId);

// 	void addEdge(int destId, int startTime);

// 	int removeEdge(int destId);

// };

// d_Node* graph_cpu_to_gpu(Node* graph) {
//     d_Node *h_graph, *dev_graph;
//     h_graph = (d_Node*)malloc(numVertices * sizeof(d_Node));
//     for (int i = 0; i < numVertices; i++) {
//         //复制类成员presentEdges
//         int size = graph[i].presentEdges.size();
//         Edge *d_graph_edge;
//         if (size != 0) {
//             gpuErrchk(cudaMalloc((void**)&d_graph_edge, size * sizeof(Edge)));
//             Edge *edge = (Edge*)malloc(size * sizeof(Edge));
//             for (int j = 0; j < size; j++) {
//                 memcpy(&edge[j], &graph[i].presentEdges[j], sizeof(Edge));
//             }
//             gpuErrchk(cudaMemcpy((void*)d_graph_edge, (void*)edge, size * sizeof(Edge), cudaMemcpyHostToDevice));
//             free(edge);
//         }
//         else {
//             gpuErrchk(cudaMalloc((void**)&d_graph_edge, sizeof(Edge)));
//         }
//         h_graph[i].size = size;
//         h_graph[i].nodeId = graph[i].nodeId;
//         h_graph[i].presentEdges = d_graph_edge;
//     }
//     gpuErrchk(cudaMalloc((void**)&dev_graph, numVertices * sizeof(d_Node)));
//     gpuErrchk(cudaMemcpy((void*)dev_graph, (void*)h_graph, numVertices * sizeof(d_Node), cudaMemcpyHostToDevice));
//     free(h_graph);
//     return dev_graph;
// }