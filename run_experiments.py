import os
import subprocess

import argparse

parser = argparse.ArgumentParser(description='Run experiments for all executables in the current directory')

parser.add_argument('--operations', '-o', dest='operation_list', type=str, help='The operation to run', required=True)
parser.add_argument('--output_dir', '-d', dest='output_dir', type=str, help='The output directory to store the results', required=True)
parser.add_argument('--mem_type', '-m', dest='mem_type', type=str, help='The memory type to run the experiments on', required=True)

args = parser.parse_args()

operations = args.operation_list.split(',')
output_dir = args.output_dir
mem_type = args.mem_type.split(',') # SYS, NUMA_DEV, NUMA_HOST, DRAM, GDDR, UM

if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

for operation in operations:
    if not os.path.isdir(os.path.join(output_dir, operation)):
        os.makedirs(os.path.join(output_dir, operation), exist_ok=True)

    for filename in os.listdir('.'):
        if os.path.isfile(filename) and os.access(filename, os.X_OK) and '.out' in filename:
            for mem in mem_type:
                # print("Running", filename, "with Empty Kernel for", mem)
                # empty_kernel_time = subprocess.run([f'./{filename}', '-n', '128', '-o', 'Pgg', '-m', mem, '-c', 'none', '-g', 'none', '-l', '1'], capture_output=True)
                
                # with open(f'{filename.replace(".out", f"_EmptyKernel_{mem}.txt")}', 'w') as f:
                #     f.write(empty_kernel_time.stdout.decode('utf-8'))
                    
                # for loop_count in ['1', '1K', '10K', '100K', '1M', '10M', '100M']:
                    # for order in ['acq', 'rel', 'none', 'acq-acq', 'acq-rel']:
                    
                for loop_count in ['1', '1K']:
                    for order in ['none', 'rel', 'acq', 'acq-acq', 'acq-rel']:
                        
                        output_filename = f'{filename.replace(".out", f"_{order}_{mem}_{loop_count}_LNNW.txt")}'
                        
                        if os.path.isfile(os.path.join(output_dir, operation, output_filename)) and os.path.getsize(os.path.join(output_dir, operation, output_filename)) > 0:
                            print(f'{output_filename} already exists. Skipping')
                        else:
                            print("Running", filename, "with", order, "and", mem, "with", loop_count, "iterations (Linked List No Warmup)")
                            cuda_explorer_output = subprocess.run([f'./{filename}', '-n', '128', '-o', operation, '-m', mem, '-c', order, '-g', order, '-l', loop_count, '-t', 'linkedlist'], capture_output=True)
                        
                            with open(os.path.join(output_dir, operation, output_filename), 'w') as f:
                                f.write(cuda_explorer_output.stdout.decode('utf-8'))   
                            
                        # output_filename = f'{filename.replace(".out", f"_{order}_{mem}_{loop_count}_LN.txt")}'
                        
                        # if os.path.isfile(os.path.join(output_dir, operation, output_filename)) and os.path.getsize(os.path.join(output_dir, operation, output_filename)) > 0:
                        #     print(f'{output_filename} already exists. Skipping')
                        # else:                                
                        #     print ("Running", filename, "with", order, "and", mem, "with", loop_count, "iterations (Linked List)")
                        #     cuda_explorer_output = subprocess.run([f'./{filename}', '-n', '128', '-o', operation, '-m', mem, '-c', order, '-g', order, '-l', loop_count, '-t', 'linkedlist', '-w'], capture_output=True)
                            
                        #     with open(os.path.join(output_dir, operation, output_filename), 'w') as f:
                        #         f.write(cuda_explorer_output.stdout.decode('utf-8'))
                
                    for order in ['none', 'rel', 'acq', 'acq-acq', 'acq-rel']:
                        
                        output_filename = f'{filename.replace(".out", f"_{order}_{mem}_{loop_count}_ANW.txt")}'
                        
                        if os.path.isfile(os.path.join(output_dir, operation, output_filename)) and os.path.getsize(os.path.join(output_dir, operation, output_filename)) > 0:
                            print(f'{output_filename} already exists. Skipping')
                        else:    
                            print ("Running", filename, "with", order, "and", mem, "with", loop_count, "iterations (Array No Warmup)")
                            cuda_explorer_output = subprocess.run([f'./{filename}', '-n', '128', '-o', operation, '-m', mem, '-c', order, '-g', order, '-l', loop_count, '-t', 'array'], capture_output=True)
                            
                            with open(os.path.join(output_dir, operation, output_filename), 'w') as f:
                                f.write(cuda_explorer_output.stdout.decode('utf-8'))    
                        
                        # output_filename = f'{filename.replace(".out", f"_{order}_{mem}_{loop_count}_A.txt")}'
                        
                        # if os.path.isfile(os.path.join(output_dir, operation, output_filename)) and os.path.getsize(os.path.join(output_dir, operation, output_filename)) > 0:
                        #     print(f'{output_filename} already exists. Skipping')
                        # else:    
                        #     print ("Running", filename, "with", order, "and", mem, "with", loop_count, "iterations (Array)")
                        #     cuda_explorer_output = subprocess.run([f'./{filename}', '-n', '128', '-o', operation, '-m', mem, '-c', order, '-g', order, '-l', loop_count, '-t', 'array', '-w'], capture_output=True)
                            
                        #     with open(os.path.join(output_dir, operation, output_filename), 'w') as f:
                        #         f.write(cuda_explorer_output.stdout.decode('utf-8'))    
                    
                    # for order in ['none', 'rel']:#, 'acq', 'acq-acq', 'acq-rel']:
                        
                    #     output_filename = f'{filename.replace(".out", f"_{order}_{mem}_{loop_count}_LLNW.txt")}'
                        
                    #     if os.path.isfile(os.path.join(output_dir, operation, output_filename)) and os.path.getsize(os.path.join(output_dir, operation, output_filename)) > 0:
                    #         print(f'{output_filename} already exists. Skipping')
                    #     else:    
                    #         print ("Running", filename, "with", order, "and", mem, "with", loop_count, "iterations (Loaded-List No Warmup)")
                    #         cuda_explorer_output = subprocess.run([f'./{filename}', '-n', '128', '-o', operation, '-m', mem, '-c', order, '-g', order, '-l', loop_count, '-t', 'loaded'], capture_output=True)
                            
                    #         with open(os.path.join(output_dir, operation, output_filename), 'w') as f:
                    #             f.write(cuda_explorer_output.stdout.decode('utf-8'))    
                        
                        # output_filename = f'{filename.replace(".out", f"_{order}_{mem}_{loop_count}_LL.txt")}'
                        
                        # if os.path.isfile(os.path.join(output_dir, operation, output_filename)) and os.path.getsize(os.path.join(output_dir, operation, output_filename)) > 0:
                        #     print(f'{output_filename} already exists. Skipping')
                        # else:    
                        #     print ("Running", filename, "with", order, "and", mem, "with", loop_count, "iterations (Loaded-List)")
                        #     cuda_explorer_output = subprocess.run([f'./{filename}', '-n', '128', '-o', operation, '-m', mem, '-c', order, '-g', order, '-l', loop_count, '-t', 'loaded', '-w'], capture_output=True)
                            
                        #     with open(os.path.join(output_dir, operation, output_filename), 'w') as f:
                        #         f.write(cuda_explorer_output.stdout.decode('utf-8'))
