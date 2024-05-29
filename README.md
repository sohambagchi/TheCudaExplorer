# The CUDA Explorer

The CUDA Explorer is a tool designed to explore memory operations and timings between CPU and GPU in CUDA-accelerated applications. This tool provides insights into memory access patterns and performance characteristics of different memory types.

## Usage

To use the CUDA Explorer, follow these steps:

1. **Compile the Code**: Compile the code files using a CUDA-enabled compiler.

```bash
make
```

This command will use the provided Makefile to compile the source files and generate the executable.

2. **Run the Program**: Run the included Python file `cleanup_executables.py` to rename the clunky executables, and also run them all. 

```bash
python3 cleanup_executables.py
```

This command runs the exploration with 512 objects, using only GPU Memory

3. **Interpret Results**: After execution, the program provides results including the sequence of memory operations and their corresponding timings for CPU and GPU events.

```bash
python3 read_data.py
```

This command reads all the data and compiles it into a CSV file. 

## Command-Line Options

The following options are available when running the program:

- `-n <int>`: Specifies the number of objects to be used in the exploration.
- `-o <string>`: Defines the order of memory operations to be performed.
- `-m <string>`: Specifies the type of memory to use, either "DRAM" or "UM" (Unified Memory).
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
