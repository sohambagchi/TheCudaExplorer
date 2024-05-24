import os
import subprocess

for file in os.listdir('.'):
    if os.path.isfile(file) and '.ptx' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('cuda::memory_order_', '').replace('acquire', '_acq').replace('relaxed', '_rel').replace('.ptx', '') + '.ptx'
        os.rename(file, new_filename)
        
    if os.path.isfile(file) and os.access(file, os.X_OK) and '.out' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('cuda::memory_order_', '').replace('acquire', '_acq').replace('relaxed', '_rel')
        os.rename(file, new_filename)
        
        cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '512', '-o', 'Cgg', '-m', 'GDDR'], capture_output=True)
        
        with open(f'{new_filename.replace(".out", ".txt")}', 'w') as f:
            f.write(cuda_explorer_output.stdout.decode('utf-8'))
