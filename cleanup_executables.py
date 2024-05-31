import os
import subprocess

for file in os.listdir('.'):
    if os.path.isfile(file) and '.ptx' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('.ptx', '') + '.ptx'
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)
        
    if os.path.isfile(file) and '.sass' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('.sass', '') + '.sass'
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)
        
    if os.path.isfile(file) and os.access(file, os.X_OK) and '.out' in file:
        new_filename = file.replace('cuda::thread_scope_', '')
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)
        
        for order in ['acq', 'rel', 'none']: #'acq-acq', 'acq-rel', 'none']:
                for mem_type in ['GDDR', 'UM', 'DRAM']:
                    print("Running", new_filename, "with", order, "and", mem_type)
                    cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', 'Cgg', '-m', mem_type, '-c', order, '-g', order], capture_output=True)
                    
                    with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}.txt")}', 'w') as f:
                        f.write(cuda_explorer_output.stdout.decode('utf-8'))

        for order in ['acq', 'rel', 'none']: # 'acq-acq', 'acq-rel', 'none']:
                for mem_type in ['UM', 'DRAM']:
                    print("Running", new_filename, "with", order, "and", mem_type)
                    cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', 'PcCggPgCccggcc', '-m', mem_type, '-c', order, '-g', order], capture_output=True)
                    
                    with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_PcCggPgCccggcc.txt")}', 'w') as f:
                        f.write(cuda_explorer_output.stdout.decode('utf-8'))
