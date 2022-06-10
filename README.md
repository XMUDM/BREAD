# BREAD

This repository holds the code for our MDM paper: ``Efficiently Answering k-hop Reachability Queries in Large Dynamic Graphs for Fraud Feature Extraction`` [[Paper](https://vgate.cs.ucy.ac.cy/public/mdm2022/pdfs/517600a238.pdf)]. If you find it is useful for your work, please consider citing our paper.

## Baseline:

1. [**ARROW**](https://github.com/senguptaneha/temporalReachabilityC): ARROW: Approximating Reachability using Random walks Over Web scale graphs. ICDE'19

## Environment:

- CUDA 10.1
- Boost 1.65.1
- GCC 4.8

## ARROW

### Build

cd bread/ARROW

1. Reachability query: `g++ -o main main.cpp`
2. k-hop reachability query: `g++ -o hop hop.cpp`

### Reachability query

```bash
./main --prefix ../dataset/ --querySize 1000 --stream --output --dataset bibsonomy
```

### k-hop reachability query

```bash
./hop --prefix ../dataset/ --querySize 1000 --stream --output --dataset bibsonomy
```

## RWBFS

### Build

The same as BREAD below.

### Reachability query

```bash
./bread rwbfs --prefix ./dataset/ --query_size 1000 --output --dynamic --dataset bibsonomy
```

### k-hop reachability query

```bash
./bread rwbfs --prefix ./dataset/ --query_size 1000 --output --dynamic --hop --dataset bibsonomy
```

## BREAD

### Build

BREAD requires boost library, please set the location of boost in CMakeLists.txt (If not installed, you can install the library with /bread/boost_1_65_1.tar.bz2)

1. `cd bread/`
2. `cmake .`
3. `make`

### Reachability query

```bash
./bread bread --prefix ./dataset/ --query_size 1000 --output --dynamic --dataset bibsonomy
```

### k-hop reachability query

```bash
./bread hop --prefix ./dataset/ --query_size 1000 --output --dynamic --hop --dataset bibsonomy
```

## BREAD+GPU

### Build

1. `cd bread/gpu`
2. `nvcc -o main main.cu -L/usr/local/cuda-10.1/lib64 -lcudart -lcuda`

### Reachability query

```bash
./main --prefix ../dataset/ --dynamic --output --dataset bibsonomy
```

### k-hop reachability query

```bash
./main --prefix ../dataset/ --dynamic --hop --output --dataset bibsonomy
```
