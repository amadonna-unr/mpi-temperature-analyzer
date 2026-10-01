# MPI Temperature Analyzer

A C++ program that uses MPI to distribute daily temperature readings across processes and OpenMP to calculate the average in parallel. It prints the average TMAX and groups dates as above, below, or at the average.

## Requirements

- Linux or WSL with Ubuntu
- C++17 compiler such as GCC
- OpenMPI OpenMP support (included with GCC)
- `daily_temps.csv` (example file)

## Build and run

In Ubuntu or WSL, confirm you have MPI tools:

```
sudo apt install g++ openmpi-bin libopenmpi-dev
```

From this directory (the one containing `mpi-temperature-analyzer.cpp` and `daily_temps.csv`), compile and run with four MPI processes:

```
mpic++ -O2 -fopenmp mpi-temperature-analyzer.cpp -o mpi-temperature-analyzer
mpirun -np 4 ./mpi-temperature-analyzer
```

To set the number of OpenMP threads per MPI process, for example two threads per process:

```
OMP_NUM_THREADS=2 mpirun -np 4 ./mpi-temperature-analyzer
```

The program prints the overall average, the three groups of readings, and elapsed time in seconds. Change `-np 4` to choose a different number of MPI processes.
