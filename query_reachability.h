#ifndef HUBPPR_QUERY_H
#define HUBPPR_QUERY_H

#include "algo_reachability.h"
#include "graph.h"
#include "heap.h"
#include "config.h"

extern iMap<double> bwd_residuals;
extern iMap<double> bwd_reserves;

void bread_setting(const Graph& graph) {
    INFO("bread setting");
    
    config.bwd_delta = calculate_bwd_delta_bread(config.delta, config.epsilon, config.dbar, config.pfail, config.opt);

    double fwd_rw_count = calculate_fwd_count_bread(config.bwd_delta, config.delta, config.epsilon, config.pfail, config.opt);

    config.fwd_delta = 1.0 / fwd_rw_count;
}

int bfsDia(int source, const Graph& graph){
	queue<int> q;
	q.push(source);
	vector<int> color(graph.n, 0);
	color[source] = 1;

	while(!q.empty()) {
		int u = q.front();
		q.pop();
		for(int i = 0; i < graph.g[u].size(); i++) {
			int v = graph.g[u][i];
			if(color[v] == 0) {
				color[v] = color[u] + 1;
				q.push(v);
			}
		}
	}
	int dia = 1;
	for(int i = 0; i < graph.n; i++) {
		if(color[i] - 1 > dia) dia = color[i] - 1;
	}
	return dia;
}

vector<int> maxDegreeNodes(const Graph& graph) {
	int numNodesToMaintain = 10;
	vector<int> maxSet(numNodesToMaintain, -1);

	for(int i = 0; i < graph.n; i++) {
		int j = 0;
		while(j < numNodesToMaintain) {
			if (maxSet[j] == -1){
				maxSet[j] = i;
				break;
			}
			if(graph.g[maxSet[j]].size() < graph.g[i].size()) {
				int temp = i;
				while(j < numNodesToMaintain) {
					swap(maxSet[j], temp);
					j++;
				}
				break;
			}
			j++;
		}
	}
	return maxSet;
}

int query(int source, int target, int endTime, int hop, const Graph &graph) {
    if(config.action != RWBFS) {
        backward_search(target, endTime, graph);
        if(monte_carlo(source, endTime, hop, graph)) return 1;
        else return 0;
    }
    else{
        bfs_search(target, endTime, hop, graph);
        if(rwbfs_monte_carlo(source, endTime, hop, graph)) return 1;
        else return 0;
    }
}

vector<pair<int, int>> p2p_query;
vector<int> hops;
vector<pair<int, int>> p2p_query_timestamp;

static void load_p2p_query(const Graph& graph){
    string queryFile = config.query_file;
    assert_file_exist("query file", queryFile);
    string line;
    ifstream myfile(queryFile);

    while (getline(myfile, line)){
        char *cstr = &line[0u];
        char *t = strtok(cstr," ");
        t = strtok(NULL, " ");
        int u = atoi(t);
        t = strtok(NULL," ");
        int v = atoi(t);
        p2p_query.push_back(make_pair(u, v));
    }
    myfile.close();
}

static void load_p2p_query_reachability(const Graph& graph){
    string queryFile = config.query_file;
    assert_file_exist("query file", queryFile);
    string line;
    ifstream myfile(queryFile);

    while (getline(myfile, line)){
        char *cstr = &line[0u];
        char *t = strtok(cstr," ");
        t = strtok(NULL, " ");
        int u = atoi(t);
        t = strtok(NULL," ");
        int v = atoi(t);
        t = strtok(NULL, " ");
        int startTime = atoi(t);
        t = strtok(NULL, " ");
        int endTime = atoi(t);
        p2p_query.push_back(make_pair(u, v));
        p2p_query_timestamp.push_back(make_pair(startTime, endTime));
    }
    myfile.close();
}

void load_p2p_query_reachability_hop(const Graph& graph){
    string queryFile = config.query_file;
    assert_file_exist("query file", queryFile);
    string line;
    ifstream myfile(queryFile);

    while (getline(myfile, line)){
        char *cstr = &line[0u];
        char *t = strtok(cstr," ");
        t = strtok(NULL, " ");
        int u = atoi(t);
        t = strtok(NULL," ");
        int v = atoi(t);
        t = strtok(NULL, " ");
        int startTime = atoi(t);
        t = strtok(NULL, " ");
        int endTime = atoi(t);
        t = strtok(NULL, " ");
        int hop = atoi(t);
        p2p_query.push_back(make_pair(u, v));
        p2p_query_timestamp.push_back(make_pair(startTime, endTime));
        hops.push_back(hop);
    }
    myfile.close();
}

void plot_in_out_degree(const Graph& graph) {
    int samplesNum = 1000;
    vector<pair<int, int> > record;
    record.resize(samplesNum);
    int numVertices = graph.n;
    for(int i = 0; i < samplesNum; i++) {
        int v = rand() % numVertices;
        int in_degree = graph.g[v].size();
        int out_degree = graph.gr[v].size();
        record[i] = make_pair(in_degree, out_degree);
    }
    FILE *fin = fopen("result/flickr_in_degree_curve.txt", "w");
    FILE *fout = fopen("result/flickr_out_degree_curve.txt", "w");
    for(int i = 0; i < samplesNum; i++) {
        auto degree = record[i];
        fprintf(fin, "%d ", degree.first);
        fprintf(fout, "%d ", degree.second);
    }
    fclose(fin);
    fclose(fout);
}

void query(const Graph& graph, string& action) {
    bwd_residuals.initialize(graph.n);
    bwd_reserves.initialize(graph.n);
    bread_setting(graph);
    if(config.isHop) {
        bwd_distances.resize(graph.n);
        load_p2p_query_reachability_hop(graph);
    }
    else if(config.isDynamic) load_p2p_query_reachability(graph);
    else load_p2p_query(graph);
    
    int size = min(config.query_size, (int)p2p_query.size());
    int *answers = (int*)malloc(sizeof(int) * size);
    srand(2021);
    double queryTime = 0.0;
    int numQueries = 0;
    struct timespec start, finish;
    for(int i = 0; i < size; i++){
        Timer timer1(1);
        auto query_pair = p2p_query[i];
        int source = query_pair.first, target = query_pair.second;
        int startTime = -1, endTime = -1, hop = -1;
        if(config.isDynamic) {
            auto query_pair_timestamp = p2p_query_timestamp[i];
            startTime = query_pair_timestamp.first;
            endTime = query_pair_timestamp.second;
        }
        if(!hops.empty()) hop = hops[i];
        int answer;
        clock_gettime(CLOCK_MONOTONIC, &start);
        answer = query(source, target, endTime, hop, graph);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        queryTime += (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / pow(10, 9);
        answers[i] = answer;
        numQueries += 1;

        if(config.isDynamic) {
            if(config.isHop)
                printf("%d %d %d %d %d %d\n", source, target, startTime, endTime, hop, answer);
            else
                printf("%d %d %d %d %d\n", source, target, startTime, endTime, answer);
        }else{
            if(config.isHop)
                printf("%d %d %d %d\n", source, target, hop, answer);
            else
                printf("%d %d %d\n", source, target, answer);
        }
        if(i % 100 == 0)
            cout << "Query processed: " << i << endl;
        // if(numQueries % 100 == 0) {
        //     printf("Query time for hop %d is %.3f\n", hop, queryTime);
        //     queryTime = 0.0;
        // }
    }
    INFO(Timer::used(1));
    if(config.write2file) {
        string file;
        if(config.isHop) {
            file = "result/" + config.action + "_result_" + config.dataset + "_dynamic_hop_1k.txt";
        }
        else{
            file = "result/" + config.action + "_result_" + config.dataset + "_dynamic_1k.txt";
        }
        FILE *fptr = fopen(file.c_str(), "w");
        for(int i = 0; i < size; i++)
            fprintf(fptr, "%d,", answers[i]);
        fclose(fptr);
    }
}

void bfs_query(const Graph& graph, string& action) {
    if(config.isHop)
        load_p2p_query_reachability_hop(graph);
    else if(config.isDynamic) load_p2p_query_reachability(graph);

    int size = min(config.query_size, (int)p2p_query.size());
    int *answers = (int*)malloc(sizeof(int) * size);
    int answer;
    for(int i = 0; i < size; i++){
        Timer timer1(1);
        auto query_pair = p2p_query[i];
        int source = query_pair.first, target = query_pair.second;
        int startTime = -1, endTime = -1, hop = -1;
        if(config.isDynamic) {
            auto query_pair_timestamp = p2p_query_timestamp[i];
            startTime = query_pair_timestamp.first;
            endTime = query_pair_timestamp.second;
        }
        if(!hops.empty()) hop = hops[i];
        answer = bfs(source, target, startTime, endTime, hop, graph);
        if(config.isDynamic) {
            if(config.isHop)
                printf("%d %d %d %d %d %d\n", source, target, startTime, endTime, hop, answer);
            else
                printf("%d %d %d %d %d\n", source, target, startTime, endTime, answer);
        }
        answers[i] = answer;
    }
    INFO(Timer::used(1));
    if(config.write2file) {
        string file;
        if(config.isHop) {
            file = "result/" + config.action + "_result_" + config.dataset + "_dynamic_hop_1k.txt";
        }
        else{
            file = "result/" + config.action + "_result_" + config.dataset + "_dynamic_1k.txt";
        }
        FILE *fptr = fopen(file.c_str(), "w");
        for(int i = 0; i < size; i++)
            fprintf(fptr, "%d,", answers[i]);
        fclose(fptr);
    }
}

#endif 
