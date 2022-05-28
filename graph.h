#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "mylib.h"
#include "config.h"
#include <random>

class Graph {
public:

    vector<vector<int>> g;
    vector<vector<int>> g_time;
    vector<vector<int>> gr;
    vector<vector<int>> gr_time;
    string data_folder;
    string dataset;

    // the teleport ratio for random walk
    double alpha;

    static bool cmp(const pair<int, double> &t1, const pair<int, double> &t2) {
        return t1.second > t2.second;
    }

    int n;
    long long m;

    Graph(string data_folder, string dataset) {
        this->data_folder = data_folder;
        this->dataset = dataset;
        this->alpha = ALPHA_DEFAULT;
        INFO("reading graph");
        init_graph_reachability();
        INFO("reading stream");
        init_stream_reachability();
    }


    void init_nm() {
        string attribute_file = data_folder + FILESEP + "attribute.txt";
        assert_file_exist("attribute file", attribute_file);
        ifstream attr(attribute_file);
        string line1, line2;
        char c;
        while (true) {
            attr >> c;
            if (c == '=') break;
        }
        attr >> n;
        while (true) {
            attr >> c;
            if (c == '=') break;
        }
        attr >> m;
    }


    void init_graph_reachability() {
        init_nm();
        g = vector<vector<int>>(n, vector<int>());
        g_time = vector<vector<int>>(n, vector<int>());
        gr = vector<vector<int>>(n, vector<int>());
        gr_time = vector<vector<int>>(n, vector<int>());
        string graphFile = data_folder + FILESEP + dataset + "Graph";
        assert_file_exist("graph file", graphFile);
        ifstream myfile(graphFile);
        string line;

        while (getline(myfile,line)){
            if (line[0] == '#') continue;
            char *cstr = &line[0u];
            char *t = strtok(cstr,"\t");
            int u = atoi(t);
            t = strtok(NULL,"\t");
            int v = atoi(t);
            if ((u < n) && (v < n)){
                g[u].push_back(v);
                g_time[u].push_back(0);
                gr[v].push_back(u);
                gr_time[v].push_back(0);
            }
        }
        myfile.close();
    }

    void init_stream_reachability() {
        string streamFile = data_folder + FILESEP + dataset + "Stream";
        assert_file_exist("stream file", streamFile);
        ifstream myfile(streamFile);
        string line;
        int startTime = 0;
        while (getline(myfile,line)){
            char *cstr = &line[0u];
            char *f = strtok(cstr," ");

            if ((f[0]== 'I')){
                char *t = strtok(NULL," ");
                int u = atoi(t);
                t = strtok(NULL," ");
                int v = atoi(t);
                t = strtok(NULL," ");
                startTime = atoi(t);
                if (f[0] == 'I'){
                    g[u].push_back(v);
                    g_time[u].push_back(startTime);
                    gr[v].push_back(u);
                    gr_time[v].push_back(startTime);
                }
            }
        }
        myfile.close();
    }

    double get_avg_degree() const {
        return double(m) / double(n);
    }


};

static void init_parameter(Config &config, const Graph &graph) {

    INFO("init parameters", graph.n);
    config.delta = 1.0 / graph.n;
    config.pfail = 1.0 / graph.n; 
    INFO(config.delta);
    INFO(config.pfail);

    config.dbar = double(graph.m) / double(graph.n); // average degree
}



#endif
