#ifndef GRAPH_H_
#define GRAPH_H_

#include "util.h"

class Edge {
public:
	int destId;
	int startTime;

	Edge() {

    }

	Edge(int isDestId, int isStartTime) {
        destId = isDestId;
        startTime = isStartTime;
    }
};

class Node {
public:
	int nodeId;
	vector<Edge> presentEdges;

	Node() {

    }

	Node(int isNodeId) {
        nodeId = isNodeId;
    }

	int getIndexOfEdge(int destId) {
        for (int i = 0; i< presentEdges.size(); i++)
            if (presentEdges[i].destId == destId) return i;

        return -1;
    }

	void addEdge(int destId, int startTime) {
        Edge e(destId, startTime);
        presentEdges.push_back(e);
    }

	int removeEdge(int destId) {
        int edgeIndex = getIndexOfEdge(destId);
        if (edgeIndex == -1) return -1;	//no such edge found
        presentEdges.erase(presentEdges.begin() + edgeIndex);
        return 0;
    }

};

class Graph {
public:
    Node *graph = NULL;
    int numVertices = 0, numEdges = 0;

    int *csr_adjs = NULL;
    int *csr_times = NULL;
	int *csr_begins = NULL;
	int *degrees = NULL;

    string data_folder = "";
    string dataset = "";

    Graph(string data_folder, string dataset) : data_folder(data_folder), dataset(dataset) {
        init_nm();
        graph = (Node*)malloc(numVertices * sizeof(Node));
        for(int i = 0; i < numVertices; i++) {
            graph[i] = Node(i);
        }
    }

    void init_nm() {
        string attribute_file = data_folder + "attribute.txt";
        assert_file_exist("attribute file", attribute_file);
        ifstream attr(attribute_file);
        string line1, line2;
        char c;
        while (true) {
            attr >> c;
            if (c == '=') break;
        }
        attr >> numVertices;
        while (true) {
            attr >> c;
            if (c == '=') break;
        }
        attr >> numEdges;
    }

    void init_csr_graph() {
        degrees = (int*)malloc(numVertices * sizeof(int));
        for(int i = 0; i < numVertices; i++) {
            degrees[i] = graph[i].presentEdges.size();
        }

        csr_begins = (int*)malloc((numVertices + 1) * sizeof(int));

        for(int i = 0; i < numVertices; i++) {
            int beginPos = 0;
            if(i == 0)  
                beginPos = 0;
            else 
                beginPos = csr_begins[i-1] + degrees[i-1];
            csr_begins[i] = beginPos;
        }

        csr_begins[numVertices] = csr_begins[numVertices-1] + degrees[numVertices-1];

        csr_adjs = (int*)malloc(numEdges * sizeof(int));
        csr_times = (int*)malloc(numEdges * sizeof(int));

        for(int i = 0; i < numVertices; i++) {
            int beginPos = csr_begins[i];
            for(int j = 0; j < degrees[i]; j++) {
                csr_adjs[beginPos + j] = graph[i].presentEdges[j].destId;
                csr_times[beginPos + j] = graph[i].presentEdges[j].startTime;
            }            
        }
    }

    ~Graph() {
        if(graph) free(graph);
        if(csr_adjs) free(csr_adjs);
        if(csr_times) free(csr_times);
        if(csr_begins) free(csr_begins);
        if(degrees) free(degrees);
    }
};

#endif