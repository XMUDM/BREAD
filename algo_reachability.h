#ifndef __ALGO_H__
#define __ALGO_H__

#include "graph.h"
#include "heap.h"
#include "config.h"
#include <tuple>

extern iMap<double> bwd_residuals;
extern iMap<double> bwd_reserves;
extern vector<int> bwd_distances;

class Avg {
public:
    int cnt = 0;
    double avg = 0;

    double update(double t) {
        cnt++;
        avg = (avg * (cnt - 1) + t) / cnt;
        return avg;
    };
};

static inline unsigned long lrand() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, INT_MAX);
    return dis(gen);
}

static inline double rand_double() {
    // return (double)(lrand()) / (double)RAND_MAX;
    static size_t seed = static_cast<unsigned int>(2021);
    static boost::random::mt19937 engine(seed);
    static boost::function<double()> randu =  boost::bind(boost::random::uniform_real_distribution<>(0, 1), engine);
    return randu();
}

int bfs(int source, int target, int startTime, int endTime, int hop, const Graph& graph) {
    Timer timer(1, "start bfs");
    if(graph.g[source].size() == 0) return 0;
    queue<int> q;
    unordered_set<int> record;

    q.push(source);
    record.insert(source);
    int layer = 0;

    while(!q.empty()) {
        int size = q.size();
        for(int i = 0; i < size; i++) {
            int t = q.front();
            q.pop();
            if(config.isHop) {
                if(t == target && layer <= hop) return 1;
            }else if(config.isDynamic) {
                if(t == target) return 1;
            }
            int deg = graph.g[t].size();
            for(int j = 0; j < deg; j++) {
                int next = graph.g[t][j];
                if(graph.g_time[t][j] <= endTime && record.count(next) == 0) {
                    q.push(next);
                    record.insert(next);
                }
            }
        }
        layer++;
    }
    return 0;
}

bool rwbfs_monte_carlo(int source, int endTime, int hop, const Graph &graph) {
    Timer timer(3, "generate random walk");
    if(graph.g[source].size() == 0) return false;
    int walkLength = config.walkLength;
    int numWalks = config.numWalks;
    if(!config.isHop) hop = walkLength;
    for(int i = 0; i < numWalks; i++) {
        int cur = source;
        int length = 0;
        for(int j = 0; j < hop; j++) {
            int size = graph.g[cur].size();
            int numPresentEdges = 0;
            if(endTime != -1) {
                vector<int> presentEdges;
                for(int k = 0; k < size; k++) {
                    int startTime = graph.g_time[cur][k];
                    int next = graph.g[cur][k];
                    if(startTime <= endTime) {
                        numPresentEdges++;
                        presentEdges.push_back(next);
                    }
                }

                if(numPresentEdges) {
                    cur = presentEdges[rand() % numPresentEdges];
                    length++;
                }
                if(config.isHop) {
                    if(bwd_residuals.exist(cur) && length + bwd_distances[cur] <= hop) return true;
                }else{
                    if(bwd_residuals.exist(cur)) return true;
                }
            }
            
        }
    }
    return false;
}

bool monte_carlo(int source, int endTime, int hop, const Graph &graph) {
    if(config.isHop) {
        if(bwd_reserves.exist(source) && bwd_distances[source] <= hop) return true;
    }
    else{
        if(bwd_reserves.exist(source)) return true;
    }
    Timer timer(3, "generate forward random walk");
    if(graph.g[source].size() == 0) return false;
    int num_tries;
    num_tries = 1 / config.fwd_delta;

    for(int i = 0; i < num_tries; i++) {
        int cur = source;
        unordered_set<int> record;
        record.insert(source);
        int length = 0;

        while(true) {
            double r = rand_double();
            if(r < config.fwd_alpha) break;

            int size = graph.g[cur].size();
            int numPresentEdges = 0;

            if(endTime != -1) {
                vector<int> presentEdges;
                for(int j = 0; j < size; j++) {
                    int startTime = graph.g_time[cur][j];
                    int next = graph.g[cur][j];
                    if(startTime <= endTime) {
                        numPresentEdges++;
                        presentEdges.push_back(next);
                    }
                }

                if(numPresentEdges) {
                    int next = presentEdges[rand() % numPresentEdges];
                    if(record.count(next) == 0) {
                        record.insert(next);
                        cur = next;
                        if(config.isHop) {
                            length++;
                            if(length > hop) break;
                        }
                    }
                }
            }else{
                if(size) {
                    if(config.isHop) {
                        int next = graph.g[cur][lrand() % size];
                        if(record.count(next) == 0) {
                            record.insert(next);
                            cur = next;
                            if(config.isHop) length++;
                        }
                    }else{
                        cur = graph.g[cur][lrand() % size];
                    }
                }
            }

            if(config.isHop) {
                if(bwd_residuals.exist(cur) && length + bwd_distances[cur] <= hop) return true;
            }
            else{
                if(bwd_residuals.exist(cur)) return true;
            }
        }

        if(config.isHop) {
            if(bwd_residuals.exist(cur) && length + bwd_distances[cur] <= hop) return true;
        }
        else{
            if(bwd_residuals.exist(cur)) return true;
        }
    }
    return false;
}

void bfs_search(int target, int endTime, int hop, const Graph &graph) {
    Timer timer(2, "start bfs search");
    queue<int> q;
    unordered_set<int> record;
    bwd_residuals.clean();

    q.push(target);
    record.insert(target);
    bwd_residuals.insert(target, 1);
    if(config.isHop) {
        for(int i = 0; i < bwd_distances.size(); i++) bwd_distances[i] = 1e6;
        bwd_distances[target] = 0;
    }
    
    while(!q.empty()) {
        int v = q.front();
        q.pop();
        int size = graph.gr[v].size();
        if(size > config.davg * config.dbar) continue;
        for(int i = 0; i < size; i++) {
            int next = graph.gr[v][i];
            if(endTime != -1) {
                int startTime = graph.gr_time[v][i];
                if(startTime <= endTime) {
                    if(record.count(next) == 0) {
                        record.insert(next);
                        q.push(next);
                        bwd_residuals.insert(next, 1);
                        if(config.isHop) 
                            bwd_distances[next] = min(bwd_distances[next], bwd_distances[v]+1);
                    }
                }
            }else{
                if(record.count(next) == 0) {
                    record.insert(next);
                    q.push(next);
                    bwd_residuals.insert(next, 1);
                }
            }
        }
    }
}

void backward_search(int target, int endTime, const Graph &graph) {
    Timer timer(2, "start backward search");
    BinaryHeap<double, greater<double> > heap(graph.n, greater<double>());
    double r_max = config.bwd_delta;
    int init_residual = 1;

    if(config.isHop) {
        for(int i = 0; i < bwd_distances.size(); i++) bwd_distances[i] = 1e6;
        bwd_distances[target] = 0;
    }

    heap.clear();
    bwd_residuals.clean();
    bwd_reserves.clean();
    heap.insert(target, init_residual);
    unordered_set<int> record;
    record.insert(target);

    while(heap.size()) {
        auto top = heap.extract_top();
        double residual = top.first;
        int v = top.second;
        if (residual < r_max)
            break;
        if(bwd_reserves.notexist(v)) {
            bwd_reserves.insert(v, residual * config.alpha);
        }
        else{
            bwd_reserves[v] += residual * config.alpha;
        }
        heap.delete_top();
        int size = graph.gr[v].size();
        for (int i = 0; i < size; i++) {
            int next = graph.gr[v][i];
            int outdegree = graph.g[next].size();
            int indegree = graph.gr[v].size();
            int cnt = indegree;
            double delta = ((1 - config.alpha) * residual) / cnt;
            if(endTime != -1) {//动态图
                int startTime = graph.gr_time[v][i];
                if(startTime <= endTime) {
                    if(heap.has_idx(next)) {
                        heap.modify(next, heap.get_value(next) + delta);
                    }else {
                        heap.insert(next, delta);
                    }
                    if(config.isHop)
                        bwd_distances[next] = min(bwd_distances[next], bwd_distances[v]+1);
                }
            }else{
                if(record.count(next) == 0) {
                    record.insert(next);
                    if (heap.has_idx(next)) {
                        heap.modify(next, heap.get_value(next) + delta);
                    }
                    else {
                        heap.insert(next, delta);
                    }
                    if(config.isHop)
                        bwd_distances[next] = min(bwd_distances[next], bwd_distances[v]+1);
                }
            }
        }
    }
    for(auto item: heap.get_elements()){
        bwd_residuals.insert(item.second, item.first);
    }
}

#endif
