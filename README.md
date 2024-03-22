Here's the README.md file updated with the Makefile section:

# The CUDA Explorer

The CUDA Explorer is a tool designed to explore memory operations and timings between CPU and GPU in CUDA-accelerated applications. This tool provides insights into memory access patterns and performance characteristics of different memory types.

## Usage

To use the CUDA Explorer, follow these steps:

1. **Compile the Code**: Compile the code files using a CUDA-enabled compiler.

```bash
make
```

This command will use the provided Makefile to compile the source files and generate the executable.

2. **Run the Program**: Execute the compiled program with appropriate command-line arguments to specify the number of objects, the order of operations, and the type of memory to use.

```bash
./theCudaExplorer -n <num_objects> -o <operation_order> -m <memory_type>
```

Replace `<num_objects>` with the desired number of objects, `<operation_order>` with the sequence of memory operations, and `<memory_type>` with either "DRAM" or "UM" (Unified Memory).

For example:

```bash
./theCudaExplorer -n 1024 -o "PcCgg" -m DRAM
```

This command runs the exploration with 1024 objects, following the specified operation order using CUDA mallocHost for DRAM memory.

3. **Interpret Results**: After execution, the program provides results including the sequence of memory operations and their corresponding timings for CPU and GPU events.


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


## Contributors

- [Your Name or Organization]

## License

[License Information]

## Acknowledgments

[Optional: Acknowledge any resources, libraries, or individuals that contributed to the project.]

---

Feel free to modify and expand this README according to your project's specifics and requirements.