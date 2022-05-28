#ifndef QUERY_H_
#define QUERY_H_

#include "bread.cuh"

const int deviceId = 1;
float bwdTime, fwdTime, fwdCheckTime, fwdRunTime;
float bwdFlagTime, bwdGetTime, bwdUpdateTime, bwdCpyTime;
float arrowTime, khopGapTime;

GraphD *graph_cpu2gpu(Graph* graph) {
    GraphD* tmp_graph = NULL, *d_graph = NULL;
    tmp_graph = (GraphD*)malloc(sizeof(GraphD));
    tmp_graph->CudaInitGraph(graph);
    gpuErrchk(cudaMalloc(&d_graph, sizeof(GraphD)));
    gpuErrchk(cudaMemcpy(d_graph, tmp_graph, sizeof(GraphD), cudaMemcpyHostToDevice));
    tmp_graph->FreeGPUMemory();
    free(tmp_graph);
    return tmp_graph;
}

PprDataD *ppr_cpu2gpu(PprData* p) {
    PprDataD* tmp_ppr = NULL, *d_ppr = NULL;
    tmp_ppr = (PprDataD*)malloc(sizeof(PprDataD));
    tmp_ppr->CudaInitPpr(p);
    gpuErrchk(cudaMalloc(&d_ppr, sizeof(PprDataD)));
    gpuErrchk(cudaMemcpy(d_ppr, tmp_ppr, sizeof(PprDataD), cudaMemcpyHostToDevice));
    // tmp_ppr->FreeGPUMemory();
    free(tmp_ppr);
    return d_ppr;
}

void query(Graph* graph, Graph* graphR, string queryFile, Config* config) {
    gpuErrchk(cudaSetDevice(deviceId));
    int numVertices = graph->numVertices;

    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    GraphD* d_graph = new GraphD(graph);
    GraphD* d_graphR = new GraphD(graphR);

    clock_gettime(CLOCK_MONOTONIC, &finish);
    float copyTime = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / pow(10, 9);
    cout << "Data Copy Time: " << copyTime << endl;

    vector<pair<int,int> > p2p_query, p2p_query_timestamp;
    vector<int> hops;
    load_p2p_query_reachability(queryFile, p2p_query, p2p_query_timestamp, hops, config->dynamic, config->hop);
    int size = p2p_query.size();
    int *answers = (int*)malloc(sizeof(int) * size);

    float queryTime = 0.0;
    bwdTime = fwdTime = fwdCheckTime = fwdRunTime = khopGapTime = 0.0;
    bwdFlagTime = bwdGetTime = bwdUpdateTime = bwdCpyTime = 0.0;

    Query* q = new Query();
    PprData *p = new PprData(q, config, size, numVertices);
    PprDataD *ppr = new PprDataD(p);
    int numQueries = 0;
    for(int i = 0; i < size; i++) {
        auto query_pair = p2p_query[i];
        int source = query_pair.first, target = query_pair.second;
        if(config->dynamic) {//动态图
            auto query_pair_timestamp = p2p_query_timestamp[i];
            int startTime = query_pair_timestamp.first, endTime = query_pair_timestamp.second;
            if(config->hop) {
                int hop = hops[i];
                q->assign(source, target, i, startTime, endTime, hop);
            }else{
                q->assign(source, target, i, startTime, endTime);
            }
        }else{
            q->assign(source, target, i);
        }
        if(i % 100 == 0) cout << "Query finished " << i << endl;
        
        bread(d_graph, d_graphR, ppr, config, q, numVertices);
        numQueries++;
    }

    gpuErrchk(cudaMemcpy(answers, ppr->d_ans, ppr->query_size * sizeof(int), cudaMemcpyDeviceToHost));
    for(int i = 0; i < size; i++) {
        auto query_pair = p2p_query[i];
        int source = query_pair.first, target = query_pair.second;
        if(config->dynamic) {
            auto query_pair_timestamp = p2p_query_timestamp[i];
            int startTime = query_pair_timestamp.first, endTime = query_pair_timestamp.second;
            printf("Q: %d %d %d %d %d\n", source, target, startTime, endTime, answers[i]);
        }else printf("Q: %d %d %d\n", source, target, answers[i]);
    }
    
    bwdTime = bwdFlagTime + bwdGetTime + bwdUpdateTime + bwdCpyTime;
    fwdTime = fwdRunTime + fwdCheckTime;
    queryTime = bwdTime + fwdTime;

    delete q;
    delete p;
    delete ppr;
    
    cout << "Backward Search Time: " << bwdTime  << " FlagTime: " << bwdFlagTime 
    << " GetTime: " << bwdGetTime << " UpdateTime: " << bwdUpdateTime << " MemcpyTime: " << bwdCpyTime << endl;
    cout << "Random Walk Time: " << fwdTime << " RunTime: " << fwdRunTime << " CheckTime: " << fwdCheckTime << endl;
    cout << "Query Time: " << queryTime << endl;

    delete d_graph;
    delete d_graphR;
    
    if(config->write2file) {
        string file;
        if(config->hop) {
            file = "result/bippr_cuda_result_" + config->dataset + "_dynamic_hop_1k.txt";
        }
        else{
            file = "result/bippr_cuda_result_" + config->dataset + "_dynamic_1k.txt";
        }
        FILE *fptr = fopen(file.c_str(), "w");
        for(int i = 0; i < size; i++)
            fprintf(fptr, "%d,", answers[i]);
        fclose(fptr);
    }

    free(answers);
}

#endif /* QUERY_H_ */
