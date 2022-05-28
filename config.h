#ifndef __CONFIG_H__
#define __CONFIG_H__


#ifdef WIN32
#define FILESEP "\\"
#else
#define FILESEP "/"
#endif

const double ALPHA_DEFAULT = 0.2;

const int NUM_OF_QUERY = 20;


const string REACHABILITY = "bread";
const string HOP = "hop";
const string BFS = "bfs";
const string RWBFS = "rwbfs";

static double calculate_bwd_delta_bread(double delta, double epsilon, double dbar, double pfail, int opt){
    // calculate r_max
    return epsilon * sqrt(delta * dbar / log(2 / pfail)) / opt;
}

static double calculate_fwd_count_bread(double bwd_delta, double delta, double epsilon, double pfail, int opt){
    return 3 * bwd_delta / delta / epsilon / epsilon * log(2 / pfail);
}


#ifdef WIN32
const string parent_folder = "../../";
#else
const string parent_folder = string("./") + FILESEP;
#endif

class Config {
public:
    string dataset;
    string graph_location;
    string query_file;
    string action = "";
    string prefix = "";

    string get_graph_folder() {
        return prefix + dataset + FILESEP;
    }

    double fwd_delta; // 1/omega  omega = # of random walk
    double bwd_delta; // identical to r_max

    int query_size = 1000;
    int num_thread = 8;
    int opt = 16;
    bool write2file = false;
    bool isDynamic = false;
    bool isHop = false;

    double pfail = 0;
    double dbar = 0;
    double epsilon = 0.5;
    double delta = 0;
    double rsum = 1;
    double omega;
    double rmax = 0;

    double alpha = ALPHA_DEFAULT;
    double fwd_alpha = ALPHA_DEFAULT;
    int walkLength = 0;
    int numWalks = 0;
    double c_numWalks = 0.5;
    int davg = 100;
};


extern Config config;

bool exists_test(const std::string &name);

void assert_file_exist(string desc, string name);

#endif
