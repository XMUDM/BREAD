#include "query.h"

void readGraph(Graph *graph, Graph *graphR) {
    string graphFile = graph->data_folder + FILESEP + graph->dataset + "Graph";
    assert_file_exist("graphFile", graphFile);
    
    ifstream myfile(graphFile);
    string line;

    while (getline(myfile, line)) {
        if (line[0] == '#') continue;
        char *cstr = &line[0u];
        char *t = strtok(cstr, "\t");
        int u = atoi(t);
        t = strtok(NULL, "\t");
        int v = atoi(t);
        if ((u < graph->numVertices) && (v < graphR->numVertices)) {
            graph->graph[u].addEdge(v, 0);
            graphR->graph[v].addEdge(u, 0);
        }
    }
    myfile.close();
    cout << "Finished Reading Graph!" << endl;
}

void readStream(Graph *graph, Graph *graphR) {
    string streamFile = graph->data_folder + FILESEP + graph->dataset + "Stream";
    assert_file_exist("updateFile", streamFile);
    ifstream myfile(streamFile);
    string line;
    int startTime;

    while (getline(myfile, line)) {
        char *cstr = &line[0u];
        char *f = strtok(cstr, " ");

        if ((f[0] == 'I') || (f[0] == 'D')) {	//insert or delete Edge
            char *t = strtok(NULL, " ");
            int u = atoi(t);
            t = strtok(NULL, " ");
            int v = atoi(t);
            t = strtok(NULL, " ");
            startTime = atoi(t);
            if (f[0] == 'I') {
                if ((u < graph->numVertices) && (v < graphR->numVertices)) {
                    graph->graph[u].addEdge(v, startTime);
                    graphR->graph[v].addEdge(u, startTime);
                }
            }
        }
    }
    myfile.close();
    cout << "Finished Reading Stream!" << endl;
}

int main(int argc, char *argv[]) {

    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Config* config = new Config();

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--alpha") {
            config->alpha = atof(argv[i+1]);
        }else if (arg == "--fwd_alpha") {
            config->fwd_alpha = atof(argv[i+1]);
        }else if (arg == "--opt") {
            config->opt = atoi(argv[i+1]);
        }else if (arg == "--rwopt") {
            config->rwopt = atoi(argv[i+1]);
        }else if (arg == "--output") {
            config->write2file = true;
        }else if (arg == "--dynamic") {
            config->dynamic = true;
        }else if (arg == "--method") {
            config->method = argv[i+1];
        }else if (arg == "--hop") {
            config->hop = true;
        }else if (arg == "--prefix") {
            config->prefix = argv[i+1];
        }else if (arg == "--dataset") {
            config->dataset = argv[i+1];
        }else if (arg == "--query_file") {
            config->query_file = argv[i+1];
        }else if (arg.substr(0, 2) == "--") {
            cerr << "command not recognize " << arg << endl;
            exit(1);
        }
    }
    config->graph_location = config->get_graph_folder();
    if(config->hop) {
        config->query_file = "../queryFile/" + config->dataset + FILESEP + config->dataset + "Query_dynamic_hop_1k.txt";
    }
    else{
        config->query_file = "../queryFile/" + config->dataset + FILESEP + config->dataset + "Query_dynamic_1k.txt";
    }

    Graph *graph = NULL, *graphR = NULL;
    graph = new Graph(config->graph_location, config->dataset);
    graphR = new Graph(config->graph_location, config->dataset);
    printf("numVertices: %d numEdges: %d\n", graph->numVertices, graph->numEdges);
    readGraph(graph, graphR);
    readStream(graph, graphR);
    graph->init_csr_graph();
    graphR->init_csr_graph();

    clock_gettime(CLOCK_MONOTONIC, &finish);
    float readTime = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / pow(10, 9);
    cout << "Data Read Time: " << readTime << endl;
    
    config->init_parameter(graph);
    
    query(graph, graphR, config->query_file, config);

    delete graph;
    delete graphR;
    delete config;
}
