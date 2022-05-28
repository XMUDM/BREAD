
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "Node.cpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>

#ifdef WIN32
#define FILESEP "\\"
#else
#define FILESEP "/"
#endif

using namespace std;

int numVertices = 1000;
int numQueries = 0;
int querySize = 1000;

double queryTime = 0;
double c_numWalks = 0.1;

Node **graph, **graphR;
int *uStops, *vStops, *answers;
int numWalks, walkLength, numStops;

void readGraph(string& graphFile){
	ifstream myfile(graphFile);
	string line;

	while (getline(myfile,line)){
		if (line[0] == '#') continue;
		char *cstr = &line[0u];
		char *t = strtok(cstr,"\t");
		int u = atoi(t);
		t = strtok(NULL,"\t");
		int v = atoi(t);
		if ((u <= numVertices) && (v <= numVertices)){
			graph[u]->addEdge(v, 0);
			graphR[v]->addEdge(u, 0);
		}
	}
	myfile.close();
}

void readStream(string& streamFile){
	ifstream myfile(streamFile);
	string line;
	int timestep;
	while (getline(myfile, line)){
		char *cstr = &line[0u];
		char *f = strtok(cstr," ");

		if ((f[0]== 'I') || (f[0] == 'D')){	//insert or delete Edge
			char *t = strtok(NULL," ");
			int u = atoi(t);
			t = strtok(NULL," ");
			int v = atoi(t);
			t = strtok(NULL," ");
			timestep = atoi(t);
			if (f[0] == 'I'){
				if ((u <= numVertices) && (v <= numVertices)){
					graph[u]->addEdge(v, timestep);
					graphR[v]->addEdge(u, timestep);
				}
			}
			else{
				// graph[u]->removeEdge(v, timestep);
				// graphR[v]->removeEdge(u, timestep);
			}
		}
	}
	myfile.close();
}

int cmpfunc (const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}

void getStops(int sourceNode, int *stops, int dir, int startTime, int endTime, vector<int>& distance, int hop){
    int labelBits = getCommonLabel(startTime, endTime);
    int label = setLabel(startTime, labelBits);
    for (int i = 0; i <= numStops; i++) stops[i] = -1;
    int n, ctr = 0;
    
    distance[sourceNode] = 0;
    for (int i = 0; i < numWalks; i++){
        n = sourceNode;
        for (int j = 0; j < hop; j++){
            int d = -1;
            if (dir == 0) d = graph[n]->randomNeighbor(label, labelBits, startTime, endTime);
            if (dir == 1) d = graphR[n]->randomNeighbor(label, labelBits, startTime, endTime);
            if (d != -1){
                distance[d] = min(distance[d], distance[n]+1);
                n = d;
				if(ctr < numStops) {
					stops[ctr] = n;
					ctr += 1;
				}
            }
        }
    }
    int i = 0;
    while (i <= numStops && stops[i] > -1){
        i++;
    }
    qsort(stops, i, sizeof(int), cmpfunc);
}

int doesIntersect(int *uStops, int *vStops, int hop, vector<int>& distanceU, vector<int>& distanceV){
    int i = 0, j = 0;
    int answer = 0;
    while ((i <= numStops) && (j <= numStops)) {
        if ((uStops[i] == -1) || (vStops[j] == -1))
            break;
        if (uStops[i] == vStops[j] && distanceU[uStops[i]] + distanceV[vStops[j]] <= hop){
            answer = 1;
            break;
        }
        if (uStops[i] < vStops[j]) i++;
        else j++;
    }
    return answer;
}

int runQuery(int u, int v, int startTime, int endTime, int hop){
    int answer = 0;
    vector<int> distanceU, distanceV;
    distanceU.resize(numVertices);
    distanceV.resize(numVertices);
    for(int i = 0; i < numVertices; i++) {
        distanceU[i] = 1e6;
        distanceV[i] = 1e6;
    }
    getStops(u, uStops, 0, startTime, endTime, distanceU, hop);
    getStops(v, vStops, 1, startTime, endTime, distanceV, hop);
    answer |= doesIntersect(uStops, vStops, hop, distanceU, distanceV);
    return answer;
}

void readAndRunQuery(string& queryFile){
    struct timespec start, finish;
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
        clock_gettime(CLOCK_MONOTONIC, &start);
        int answer = runQuery(u, v, startTime, endTime, hop);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        queryTime += (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / pow(10, 9);
        printf("Q: %d %d %d %d %d %d\n", u, v, startTime, endTime, hop, answer);
        answers[numQueries] = answer;
        numQueries += 1;
        if (numQueries % 100 == 0)
            cout << "Query processed: " << numQueries << endl;
        if (numQueries >= querySize) break;
    }
    myfile.close();
}

void setParams(int Dia){
	walkLength = Dia;
	int c_walkLength = 1;
	numWalks = (int)(c_numWalks*(floor(cbrt(pow(numVertices, 2) * log(numVertices)))));
	numStops = numWalks*walkLength;
	uStops = (int*)malloc(sizeof(int) * (numStops+1));
	vStops = (int*)malloc(sizeof(int) * (numStops+1));
	answers = (int*)malloc(sizeof(int) * querySize);
}

int bfsDia(int source){
	queue<int> q;
	q.push(source);
	vector<int> color(numVertices, 0);
	color[source] = 1;

	while(!q.empty()) {
		int u = q.front();
		q.pop();
		for(int i = 0; i < graph[u]->presentEdges.size(); i++) {
			int v = graph[u]->presentEdges[i]->destId;
			if(color[v] == 0) {
				color[v] = color[u] + 1;
				q.push(v);
			}
		}
	}
	int dia = 1;
	for(int i = 0; i < numVertices; i++) {
		if(color[i] - 1 > dia) dia = color[i] - 1;
	}
	return dia;
}

vector<int> maxDegreeNodes() {
	int numNodesToMaintain = 10;
	vector<int> maxSet(numNodesToMaintain, -1);

	for(int i = 0; i < numVertices; i++) {
		int j = 0;
		while(j < numNodesToMaintain) {
			if (maxSet[j] == -1){
				maxSet[j] = i;
				break;
			}
			if(graph[maxSet[j]]->presentEdges.size() < graph[i]->presentEdges.size()) {
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

int init_nm(string& data_folder) {
	string attribute_file = data_folder + FILESEP + "attribute.txt";
	assert_file_exist("attribute file", attribute_file);
	ifstream attr(attribute_file);
	string line1, line2;
	char c;
	int n = 0, m = 0;
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
	return n;
}

int main(int argc, char *argv[]){
	bool streamFlag = true;
	bool write2file = false;
    string dataset, prefix, queryFile;
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--stream") {
            streamFlag = true;
        }
        else if (arg == "--dataset") {
            dataset = argv[i+1];
        }
        else if (arg == "--prefix") {
            prefix = argv[i + 1];
        }
        else if (arg == "--querySize") {
            querySize = atoi(argv[i+1]);
        }
		else if (arg == "--query_file") {
            queryFile = argv[i+1];
        }
		else if (arg == "--output") {
			write2file = true;
		}
		else if (arg == "--c") {
			c_numWalks = atof(argv[i+1]);
		}
        else if (arg.substr(0, 2) == "--") {
            cerr << "command not recognize " << arg << endl;
            exit(1);
        }
    }

	cout << "start running dynamic query for " << dataset << endl;
	string graph_location = prefix + dataset + FILESEP;

	struct timespec start, finish;
	queryFile = "../queryFile/" + dataset + FILESEP + dataset + "Query_dynamic_hop_1k.txt";

	numVertices = init_nm(graph_location);

	graph = (Node**)malloc((numVertices+1)*sizeof(Node*));
	graphR = (Node**)malloc((numVertices+1)*sizeof(Node*));

	for (int i = 0; i <= numVertices; i++){
		graph[i] = new Node(i);
		graphR[i] = new Node(i);
	}

	clock_gettime(CLOCK_MONOTONIC, &start);

	string graphFile = graph_location + FILESEP + dataset + "Graph";
	assert_file_exist("graph file", graphFile);
	readGraph(graphFile);
	cout << "Finished Reading Graph!" << endl << flush;
	if(streamFlag) {
		string streamFile = graph_location + FILESEP + dataset + "Stream";
		assert_file_exist("stream file", streamFile);
		readStream(streamFile);
		cout << "Finished Reading Stream!" << endl;
	}
	clock_gettime(CLOCK_MONOTONIC, &finish);
	double readTime = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / pow(10,9);
	cout << "Data read time = " << readTime << endl;

	setParams(30);
	
	queryTime = 0.0;
	numQueries = 0;
	srand(2021);
	readAndRunQuery(queryFile);
	if(write2file) {
		string file = "./result/arrow_result_" + dataset + "_dynamic_hop_1k.txt";
		FILE *fptr = fopen(file.c_str(), "w");
		for(int i = 0; i < numQueries; i++)
			fprintf(fptr, "%d,", answers[i]);
		fclose(fptr);
	}
	cout << "Total Query Time = " << queryTime << endl;

	for (int i = 0; i <= numVertices; i++){
		delete graph[i];
		delete graphR[i];
	}
	if(graph) free(graph);
	if(graphR) free(graphR);
}
