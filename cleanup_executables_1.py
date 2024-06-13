import os
import subprocess
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'clean':
    only_clean = True
else: only_clean = False
    


for file in os.listdir('.'):
    if os.path.isfile(file) and '.ptx' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('.ptx', '').replace('241664', '60MB').replace('479228', '117MB').replace('16777216', '4GB') + '.ptx'
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)
        
    if os.path.isfile(file) and '.sass' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('.sass', '').replace('241664', '60MB').replace('479228', '117MB').replace('16777216', '4GB') + '.sass'
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)
        
    if os.path.isfile(file) and os.access(file, os.X_OK) and '.out' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('241664', '60MB').replace('479228', '117MB').replace('16777216', '4GB')
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)
        
        
        if not only_clean:
            for mem_type in ['GDDR', 'UM', 'DRAM']:
                print("Running", new_filename, "with Empty Kernel for", mem_type)
                empty_kernel_time = subprocess.run([f'./{new_filename}', '-n', '128', '-o', 'Pgg', '-m', mem_type, '-c', 'none', '-g', 'none', '-l', '1'], capture_output=True)
                
                with open(f'{new_filename.replace(".out", f"_EmptyKernel_{mem_type}.txt")}', 'w') as f:
                    f.write(empty_kernel_time.stdout.decode('utf-8'))
                    
                for loop_count in ['1', '1K', '10K', '100K', '1M', '10M', '100M']:
                    for order in ['acq', 'rel', 'none', 'acq-acq', 'acq-rel']:
                        # print("Running", new_filename, "with", order, "and", mem_type)
                        # cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', 'Cgg', '-m', mem_type, '-c', order, '-g', order], capture_output=True)
                        
                        # with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}.txt")}', 'w') as f:
                        #     f.write(cuda_explorer_output.stdout.decode('utf-8'))
                            
                        print ("Running", new_filename, "with", order, "and", mem_type, "with", loop_count, "iterations")
                        cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', 'Cgg', '-m', mem_type, '-c', order, '-g', order, '-l', loop_count, '-w'], capture_output=True)
                        
                        with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}.txt")}', 'w') as f:
                            f.write(cuda_explorer_output.stdout.decode('utf-8'))    
                            
                        

            # for order in ['acq', 'rel', 'none']: # 'acq-acq', 'acq-rel', 'none']:
            #         for mem_type in ['UM', 'DRAM']:
            #             print("Running", new_filename, "with", order, "and", mem_type)
            #             cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', 'PcCggPgCccggcc', '-m', mem_type, '-c', order, '-g', order], capture_output=True)
                        
            #             with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_PcCggPgCccggcc.txt")}', 'w') as f:
            #                 f.write(cuda_explorer_output.stdout.decode('utf-8'))
