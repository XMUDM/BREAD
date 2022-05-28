#ifndef PPRDATA_H_
#define PPRDATA_H_

#include <stdlib.h>
#include "config.h"

class PprData {
public:
    int* frontiers_cnt = NULL;
    int* record = NULL;
    int* lengths = NULL;
    int* ans = NULL;
    int* frontiers = NULL;
    float* residuals = NULL;
    int* distance = NULL;
    int* vertices = NULL;
    unsigned char* new_frontier_flags = NULL;
    unsigned char* visited = NULL;
    unsigned char* result = NULL;
    int numVertices = 0;
    int source = 0, target = 0;
    int startTime = 0, endTime = 0;
    int num_tries = 0;
    int query_size = 0;

    PprData(Query *q, Config *config, int query_size, int numVertices) : numVertices(numVertices){
        this->source = q->source;
        this->target = q->target;
        this->startTime = q->startTime;
        this->endTime = q->endTime;
        this->query_size = query_size;
        frontiers_cnt = (int*)malloc(sizeof(int));
        frontiers = (int*)malloc(numVertices * sizeof(int));
        residuals = (float*)malloc(numVertices * sizeof(float));
        distance = (int*)malloc(numVertices * sizeof(int));
        vertices = (int*)malloc(numVertices * sizeof(int));
        new_frontier_flags = (unsigned char*)malloc(numVertices * sizeof(unsigned char));
        visited = (unsigned char*)malloc(numVertices * sizeof(unsigned char));
        result = (unsigned char*)malloc(sizeof(unsigned char));
        num_tries = config->fwd_rw_count;
        record = (int*)malloc(num_tries * sizeof(int));
        lengths = (int*)malloc(num_tries * sizeof(int));
        ans = (int*)malloc(query_size * sizeof(int));

        frontiers_cnt[0] = 1; //初始只有target一个需要push的顶点
        for(int i = 0; i < numVertices; i++) {
            distance[i] = 100000;//初始均不可达
            residuals[i] = 0.0;
            vertices[i] = i;
            new_frontier_flags[i] = 0;
            visited[i] = 0;
            frontiers[i] = 0;
        }
        distance[target] = 0;
        residuals[target] = 1.0;
        frontiers[0] = target;
        result[0] = 0;
        for(int i = 0; i < num_tries; i++) {
            record[i] = 0;
            lengths[i] = 0;
        }
        for(int i = 0; i < query_size; i++) ans[i] = 0;
    }

    ~PprData() {
        if(frontiers_cnt) free(frontiers_cnt);
        if(frontiers) free(frontiers);
        if(residuals) free(residuals);
        if(vertices) free(vertices);
        if(new_frontier_flags) free(new_frontier_flags);
        if(visited) free(visited);
        if(record) free(record);
        if(ans) free(ans);
    }

};

#endif