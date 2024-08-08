# The CUDA Explorer

The CUDA Explorer is a tool designed to explore memory operations and timings between CPU and GPU in CUDA-accelerated applications. This tool provides insights into memory access patterns and performance characteristics of different memory types.

## Usage

To use the CUDA Explorer, follow these steps:

1. **Compile the Code**: Compile the code files using a CUDA-enabled compiler.

```bash
make
```

This command will use the provided Makefile to compile the source files and generate the executable.

2. **Cleanup the Executable Filenames**: Run the included Python file `cleanup_executables.py` to rename the clunky executables. 

```bash
python3 cleanup_executables.py
```

3. **Run the Experiments**: Run the included Python file `run_experiments.py` to run a specified experiment for all executables.

```bash
python3 run_experiments.py -o <COMMA_SEPARATED_LIST_OF_OPERATION_SEQUENCES> -d <OUTPUT_DIRECTORY> -m <COMMA_SEPARATED_LIST_OF_MEMORY_ALLOCATORS>
```

Example:
```bash
python3 run_experiments.py -o Cgggg,PcCgggg -d output_dir -m UM,GDDR,SYS
```

This creates a `Cgggg` and `PcCgggg` subdirectory in `output_dir`, and then runs both experiments for all three memory allocators and stores the results in the respective subdirectory.


4. **Interpret Results**: After execution, the program provides results including the sequence of memory operations and their corresponding timings for CPU and GPU events.

```bash
python3 read_data.py -o <OPERATION_SEQUENCE> -d <DIRECTORY_CONTAINING_TXT_FILES>
```

This command reads all the data files in the operation subdirectory and compiles it into a single CSV file. 

## Command-Line Options

The following options are available when running the program:

- `-n <int>`: Specifies the number of objects to be used in the exploration.
- `-o <string>`: Defines the order of memory operations to be performed.
- `-m <string>`: Specifies the type of memory to use, either `SYS` (malloc()), `DRAM` (cudaMallocHost()), `UM` (cudaMallocManaged()), or `GDDR` (cudaMalloc())
- `-l <1, 1K>`: Specifies the number of outer-loop iterations around the load/store operation
- `-c <none, acq, rel, acq-acq, acq-rel>`: Specifies the memory ordering semantics to use for CPU operations
- `-g <none, acq, rel, acq-acq, acq-rel>`: Specifies the memory ordering semantics to use for GPU operations
- `-t <array, linkedlist>`: Specifies the data structures to use for the experiment. 
- `-h`: Displays the help message with usage instructions.

## Sample Usage

```bash
$ ./theCudaExplorer -n 1024 -m DRAM -o PcCgg
Size of Object: 6.78 MB
Number of Objects: 1024
CPU Events Timed: 1      GPU Events Timed: 2

Using cudaMallocHost for Objects

Sequence
======================
CPU st
                GPU ld
                GPU ld

Results
======================
CPU st
    (59 ns)
                GPU ld
                    (7953 ns)
                GPU ld
                    (1778 ns)
```

```bash
$ ./theCudaExplorer -n 1024 -m UM -o PcgCgcg
Size of Object: 6.78 MB
Number of Objects: 1024
CPU Events Timed: 2      GPU Events Timed: 3

Using cudaMallocManaged for Objects

Sequence
======================
CPU st
                GPU st
                GPU ld
CPU ld
                GPU ld

Results
======================
CPU st
    (64 ns)
                GPU st
                    (35192 ns)
                GPU ld
                    (1032 ns)
CPU ld
    (14260 ns)
                GPU ld
                    (19426 ns)
```

### Operation Sequence String

```
Actions
-------
P - Producer
C - Consumer

Device
------
g - GPU
c - CPU
```

The sequence is defined first by the action (Producer or Consumer), followed by the device or devices performing it. Only when the action changes does the corresponding action specifier need to be used again. 

```
Pccgc       PgCcgc

CPU st      GPU st
CPU st      CPU ld
GPU st      GPU ld
CPU st      CPU ld
```
