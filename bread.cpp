#define _CRT_SECURE_NO_DEPRECATE
#define HEAD_INFO

#include "mylib.h"
#include <iostream>
#include <fstream>
#include <map>
#include <stdlib.h>
#include <set>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "graph.h"
#include "config.h"
#include "algo_reachability.h"
#include "query_reachability.h"


using namespace std;

iMap<double> bwd_residuals;
iMap<double> bwd_reserves;
vector<int> bwd_distances;

int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    program_start(argc, argv);

    // this is main entry
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--alpha") {
            config.alpha = atof(argv[i + 1]);
        }
        else if (arg == "--fwd_alpha") {
            config.fwd_alpha = atof(argv[i + 1]);
        }
        else if (arg == "--query_size") {
            config.query_size = atoi(argv[i+1]);
        }
        else if (arg == "--num_thread") {
            config.num_thread = atoi(argv[i+1]);
        }
        else if (arg == "--query_file") {
            config.query_file = argv[i+1];
        }
        else if (arg == "--opt") {
            config.opt = atoi(argv[i+1]);
        }
        else if (arg == "--c") {
            config.c_numWalks = atof(argv[i+1]);
        }
        else if (arg == "--hop") {
            config.isHop = true;
        }
        else if (arg == "--output") {
            config.write2file = true;
        }
        else if (arg == "--dynamic") {
            config.isDynamic = true;
        }
        else if (arg == "--prefix") {
            config.prefix = argv[i + 1];
        }
        else if (arg == "--d") {
            config.davg = atoi(argv[i+1]);
        }
        else if (arg == "--dataset") {
            config.dataset = argv[i + 1];
        }
        else if (arg.substr(0, 2) == "--") {
            cerr << "command not recognize " << arg << endl;
            exit(1);
        }
    }

    config.action = argv[1];
    cout << "action: " << config.action << endl;

    config.graph_location = config.get_graph_folder();
    if(config.isHop) {
        config.query_file = "./queryFile/" + config.dataset + FILESEP + config.dataset + "Query_dynamic_hop_1k.txt";
    }
    else{
        config.query_file = "./queryFile/" + config.dataset + FILESEP + config.dataset + "Query_dynamic_1k.txt";
    }

    Graph graph(config.graph_location, config.dataset);
    INFO("load graph finish");

    INFO(graph.n, graph.m);
    
    INFO(config.action);

    if(config.action != BFS) {
        init_parameter(config, graph);
    }
    if(config.action == RWBFS) {
        Timer timer(4, "diameter estimation");
        vector<int> maxSet = maxDegreeNodes(graph);
		int d = 6;
		for(int i = 0; i < 10; i++) {
			int d1 = bfsDia(maxSet[i], graph);
			if(d1 > d) d = d1;
		}
        config.walkLength = d;
        double c_numWalks = config.c_numWalks;
	    config.numWalks = (int)(c_numWalks*(floor(cbrt(pow(graph.n, 2) * log(graph.n)))));
    }
   
    INFO("finished initing parameters");
    if(config.action == BFS)
        bfs_query(graph, config.action);
    else
        query(graph, config.action);

    Timer::show();
    program_stop();
    return 0;
}
