#include "Graph.h"

#ifdef WIN32
#define FILESEP "\\"
#else
#define FILESEP "/"
#endif

class Config {
public:
    double bwd_delta = 0.0;
    int fwd_rw_count = 0;
	double fwd_delta = 0.0;
    double pfail = 0;
    double dbar = 0;
    double epsilon = 0.5;
    double delta = 0;
    double alpha = 0.2;
    double fwd_alpha = 0.2;
    int opt = 64;
    int rwopt = 1;
    bool write2file = false;
    bool dynamic = false;
    bool hop = false;
    int batch_size = 1000;
    int numWalks = 0;
    int walkLength = 0;
    int numStops = 0;
    int dia = 6;

    string dataset = "";
    string prefix = "";
    string graph_location = "";
    string query_file = "";
    string method = "";

    string get_graph_folder() {
        return prefix + dataset + FILESEP;
    }

    void init_parameter(Graph *graph) {
        this->delta = 1.0 / graph->numVertices;
        this->pfail = 1.0 / graph->numVertices; 
        this->dbar = double(graph->numEdges) / double(graph->numVertices); // average degree
        calculate_bwd_delta_bippr(this->delta, this->epsilon, this->dbar, this->pfail);
        this->fwd_rw_count = calculate_fwd_count_bippr(this->bwd_delta, this->delta, this->epsilon, this->pfail);
        this->fwd_delta = 1.0 / fwd_rw_count;
    }

    void calculate_bwd_delta_bippr(double delta, double epsilon, double dbar, double pfail){
        // calculate r_max
        this->bwd_delta = epsilon * sqrt(delta * dbar / log(2 / pfail)) / opt;
    }

    double calculate_fwd_count_bippr(double bwd_delta, double delta, double epsilon, double pfail){
        return 3 * bwd_delta / delta / epsilon / epsilon * log(2 / pfail) * opt * rwopt;
    }
};