#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "TrieTable.cpp"

using namespace std;

double qTime1 = 0, qTime2 = 0;

struct Edge {
	int destId;
	int startTime;

	Edge(){
		destId = -1;
		startTime = -1;
	}

	Edge(int isDestId, int isStartTime) {
		destId = isDestId;
		startTime = isStartTime;
	}
};

struct Node{
	int nodeId;
	TrieTable *t;
	vector<Edge*> presentEdges;

	Node() {
		nodeId = 0;
		t = new TrieTable();
	}

	Node(int isNodeId) {
		nodeId = isNodeId;
		t = new TrieTable();
	}
	
	int getIndexOfEdge(int destId) {
		for (int i = 0; i< presentEdges.size(); i++)
			if (presentEdges[i]->destId == destId) return i;

		return -1;
	}

	void addEdge(int destId, int timestep) {
		presentEdges.push_back(new Edge(destId, timestep));
	}

	int removeEdge(int destId, int timestep) {
		int edgeIndex = getIndexOfEdge(destId);
		if (edgeIndex == -1) return -1;	//no such edge found
		int startTime = presentEdges[edgeIndex]->startTime;
		int endTime = timestep;
		int numBits = getCommonLabel(startTime, endTime);
		t->addEdge(destId, startTime, numBits);
		presentEdges.erase(presentEdges.begin() + edgeIndex);
		return startTime;
	}

	int randomNeighbor(int label, int numBits, int startTime, int endTime){
		struct timespec start, finish;
		int presentSample = -1;
		int numPresentEdges = 0;
		for (int i = 0; i < presentEdges.size(); i++){
			if (presentEdges[i]->startTime <= endTime){
				numPresentEdges += 1;
				int z = rand() % numPresentEdges;
				if (z == 0) presentSample = presentEdges[i]->destId;
			}
		}
		int rN = t->randomNeighbor(startTime, endTime, presentSample, numPresentEdges);
		return rN;
	}

	int randomNeighborStatic(){
		struct timespec start, finish;
		int presentSample = -1;
		int numPresentEdges = 0;
		for (int i = 0; i < presentEdges.size(); i++){
			numPresentEdges += 1;
			int z = rand() % numPresentEdges;
			if (z == 0) presentSample = presentEdges[i]->destId;
		}
		return presentSample;
	}

	int randomNeighbor_without_tri(int startTime, int endTime){
		vector<int> record;
		for(int i = 0; i < presentEdges.size(); i++){
			if(presentEdges[i]->startTime <= endTime){
				record.push_back(presentEdges[i]->destId);
			}	
		}
		int size = record.size();
		if(size == 0) return -1;
		return record[rand() % size];
	}

};
