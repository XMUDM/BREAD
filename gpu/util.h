#ifndef UTIL_H_
#define UTIL_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
#include <string.h>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#ifdef WIN32
#define FILESEP "\\"
#else
#define FILESEP "/"
#endif

using namespace std;

class Query {
public:
    int source = 0;
    int target = 0;
    int startTime = 10000;
    int endTime = 10000;//静态图时一定能满足edge->startTime <= endTime
    int index = 0;
    int hop = 0;

    Query() {

    }

    void assign(int source, int target, int index, int startTime = 10000, int endTime = 10000, int hop = 0) {
        this->source = source;
        this->target = target;
        this->startTime = startTime;
        this->endTime = endTime;
        this->index = index;
        this->hop = hop;
    }

};

bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    if (f.good()) {
        f.close();
        return true;
    }
    else {
        f.close();
        return false;
    }
}

void assert_file_exist(string desc, string name) {

    if (!exists_test(name)) {
        cerr << desc << " " << name << " not find " << endl;
        exit(1);
    }

}

void load_p2p_query_reachability(string queryFile, vector<pair<int,int> >& p2p_query, 
vector<pair<int,int> >& p2p_query_timestamp, vector<int>& hops, bool isDynamic, bool isHop){
    string line;
    ifstream myfile(queryFile);
    assert_file_exist("query file", queryFile);

    while (getline(myfile, line)){
        char *cstr = &line[0u];
        char *t = strtok(cstr," ");
        t = strtok(NULL, " ");
        int u = atoi(t);
        t = strtok(NULL," ");
        int v = atoi(t);
        if(isDynamic) {
            t = strtok(NULL, " ");
            int startTime = atoi(t);
            t = strtok(NULL, " ");
            int endTime = atoi(t);
            p2p_query_timestamp.push_back(make_pair(startTime, endTime));
            if(isHop) {
                t = strtok(NULL, " ");
                int hop = atoi(t);
                hops.push_back(hop);
            }
        }
        p2p_query.push_back(make_pair(u, v));
    }
    myfile.close();
}

int length(char *File) {
    ifstream myfile(File);
    string line;
    int lineNum = 0;
    while (getline(myfile, line))
        lineNum++;
    myfile.clear();
    myfile.seekg(0, ios::beg); //return to the first line of the file
    return lineNum;
}

#endif