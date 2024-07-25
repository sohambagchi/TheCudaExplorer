import os
import subprocess
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'clean':
    only_clean = True
else: 
    only_clean = False
    operation = sys.argv[1]

extra = list()

if len(sys.argv) > 2 and sys.argv[2] == 'GDDR':
    extra.append(sys.argv[2])
    
for file in os.listdir('.'):
    if os.path.isfile(file) and '.ptx' in file:
        # new_filename = file.replace('cuda::thread_scope_', '').replace('.ptx', '').replace('131066', '32MB').replace('241664', '60MB').replace('1048570', '256MB').replace('2097146', '512MB').replace('479228', '117MB').replace('16777216', '4GB').replace('4194298', '1GB').replace('4092', '1MB').replace('8188', '2MB').replace('16380', '4MB').replace('32764', '8MB').replace('65532', '16MB').replace('260', '64KB').replace('1284', '320KB').replace('2564', '640KB').replace('8388604', '2GB').replace('12', '32B').replace('28', '64B').replace('60', '128B').replace('92', '192B').replace('124', '256B').replace('188', '384B').replace('252', '512B').replace('508', '1KB').replace('1020', '2KB').replace('2044', '4KB').replace('4092', '8KB').replace('8188', '16KB').replace('16380', '32KB').replace('32764', '64KB').replace('65532', '128KB').replace('131068', '256KB').replace('262140', '512KB').replace('524284', '1MB').replace('1048572', '2MB').replace('2097148', '4MB').replace('4194300', '8MB').replace('8388604', '16MB').replace('16777212', '32MB').replace('33554428', '64MB').replace('67108860', '128MB').replace('134217724', '256MB') + '.ptx'
        new_filename = file.replace('cuda::thread_scope_', '').replace('134217724.', '256MB.').replace('67108860.', '128MB.').replace('33554428.', '64MB.').replace('16777212.', '32MB.').replace('8388604.', '16MB.').replace('4194300.', '8MB.').replace('2097148.', '4MB.').replace('1048572.', '2MB.').replace('524284.', '1MB.').replace('262140.', '512KB.').replace('131068.', '256KB.').replace('65532.', '128KB.').replace('32764.', '64KB.').replace('16380.', '32KB.').replace('8188.', '16KB.').replace('4092.', '8KB.').replace('2044.', '4KB.').replace('1020.', '2KB.').replace('508.', '1KB.').replace('252.', '512B.').replace('188.', '384B.').replace('124.', '256B.').replace('92.', '192B.').replace('60.', '128B.').replace('28.', '64B.').replace('12.', '32B.').replace('.ptx', '') + '.ptx'
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)

    if os.path.isfile(file) and '.sass' in file:
        # new_filename = file.replace('cuda::thread_scope_', '').replace('.sass', '').replace('131066', '32MB').replace('241664', '60MB').replace('1048570', '256MB').replace('2097146', '512MB').replace('479228', '117MB').replace('16777216', '4GB').replace('4194298', '1GB').replace('4092', '1MB').replace('8188', '2MB').replace('16380', '4MB').replace('32764', '8MB').replace('65532', '16MB').replace('260', '64KB').replace('1284', '320KB').replace('2564', '640KB').replace('8388604', '2GB').replace('12', '32B').replace('28', '64B').replace('60', '128B').replace('92', '192B').replace('124', '256B').replace('188', '384B').replace('252', '512B').replace('508', '1KB').replace('1020', '2KB').replace('2044', '4KB').replace('4092', '8KB').replace('8188', '16KB').replace('16380', '32KB').replace('32764', '64KB').replace('65532', '128KB').replace('131068', '256KB').replace('262140', '512KB').replace('524284', '1MB').replace('1048572', '2MB').replace('2097148', '4MB').replace('4194300', '8MB').replace('8388604', '16MB').replace('16777212', '32MB').replace('33554428', '64MB').replace('67108860', '128MB').replace('134217724', '256MB') + '.sass'
        new_filename = file.replace('cuda::thread_scope_', '').replace('134217724.', '256MB.').replace('67108860.', '128MB.').replace('33554428.', '64MB.').replace('16777212.', '32MB.').replace('8388604.', '16MB.').replace('4194300.', '8MB.').replace('2097148.', '4MB.').replace('1048572.', '2MB.').replace('524284.', '1MB.').replace('262140.', '512KB.').replace('131068.', '256KB.').replace('65532.', '128KB.').replace('32764.', '64KB.').replace('16380.', '32KB.').replace('8188.', '16KB.').replace('4092.', '8KB.').replace('2044.', '4KB.').replace('1020.', '2KB.').replace('508.', '1KB.').replace('252.', '512B.').replace('188.', '384B.').replace('124.', '256B.').replace('92.', '192B.').replace('60.', '128B.').replace('28.', '64B.').replace('12.', '32B.').replace('.sass', '') + '.sass'
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)

    if os.path.isfile(file) and os.access(file, os.X_OK) and '.out' in file:
        # new_filename = file.replace('cuda::thread_scope_', '').replace('131066', '32MB').replace('241664', '60MB').replace('1048570', '256MB').replace('479228', '117MB').replace('2097146', '512MB').replace('16777216', '4GB').replace('4194298', '1GB').replace('4092', '1MB').replace('8188', '2MB').replace('16380', '4MB').replace('32764', '8MB').replace('65532', '16MB').replace('260', '64KB').replace('1284', '320KB').replace('2564', '640KB').replace('8388604', '2GB').replace('12', '32B').replace('28', '64B').replace('60', '128B').replace('92', '192B').replace('124', '256B').replace('188', '384B').replace('252', '512B').replace('508', '1KB').replace('1020', '2KB').replace('2044', '4KB').replace('4092', '8KB').replace('8188', '16KB').replace('16380', '32KB').replace('32764', '64KB').replace('65532', '128KB').replace('131068', '256KB').replace('262140', '512KB').replace('524284', '1MB').replace('1048572', '2MB').replace('2097148', '4MB').replace('4194300', '8MB').replace('8388604', '16MB').replace('16777212', '32MB').replace('33554428', '64MB').replace('67108860', '128MB').replace('134217724', '256MB')
        new_filename = file.replace('cuda::thread_scope_', '').replace('134217724.', '256MB.').replace('67108860.', '128MB.').replace('33554428.', '64MB.').replace('16777212.', '32MB.').replace('8388604.', '16MB.').replace('4194300.', '8MB.').replace('2097148.', '4MB.').replace('1048572.', '2MB.').replace('524284.', '1MB.').replace('262140.', '512KB.').replace('131068.', '256KB.').replace('65532.', '128KB.').replace('32764.', '64KB.').replace('16380.', '32KB.').replace('8188.', '16KB.').replace('4092.', '8KB.').replace('2044.', '4KB.').replace('1020.', '2KB.').replace('508.', '1KB.').replace('252.', '512B.').replace('188.', '384B.').replace('124.', '256B.').replace('92.', '192B.').replace('60.', '128B.').replace('28.', '64B.').replace('12.', '32B.')
        
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)
        
        if not only_clean:
            for mem_type in ['UM', 'DRAM', 'GDDR'] + extra:
                # print("Running", new_filename, "with Empty Kernel for", mem_type)
                # empty_kernel_time = subprocess.run([f'./{new_filename}', '-n', '128', '-o', 'Pgg', '-m', mem_type, '-c', 'none', '-g', 'none', '-l', '1'], capture_output=True)
                
                # with open(f'{new_filename.replace(".out", f"_EmptyKernel_{mem_type}.txt")}', 'w') as f:
                #     f.write(empty_kernel_time.stdout.decode('utf-8'))
                    
                # for loop_count in ['1', '1K', '10K', '100K', '1M', '10M', '100M']:
                    # for order in ['acq', 'rel', 'none', 'acq-acq', 'acq-rel']:
                    
                for loop_count in ['1', '1K', '10K']:
                    for order in ['none', 'acq', 'rel', 'acq-acq', 'acq-rel']:
                        if os.path.isfile(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LNNW.txt")}') and os.path.getsize(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LNNW.txt")}') > 0:
                            print(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LNNW.txt")} already exists. Skipping')
                        else:
                            print("Running", new_filename, "with", order, "and", mem_type, "with", loop_count, "iterations (Linked List No Warmup)")
                            cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', operation, '-m', mem_type, '-c', order, '-g', order, '-l', loop_count, '-t', 'linkedlist'], capture_output=True)
                        
                            with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LNNW.txt")}', 'w') as f:
                                f.write(cuda_explorer_output.stdout.decode('utf-8'))   
                             
                        if os.path.isfile(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LN.txt")}') and os.path.getsize(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LN.txt")}') > 0:
                            print(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LN.txt")} already exists. Skipping')
                        else:                                
                            print ("Running", new_filename, "with", order, "and", mem_type, "with", loop_count, "iterations (Linked List)")
                            cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', operation, '-m', mem_type, '-c', order, '-g', order, '-l', loop_count, '-t', 'linkedlist', '-w'], capture_output=True)
                            
                            with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LN.txt")}', 'w') as f:
                                f.write(cuda_explorer_output.stdout.decode('utf-8'))
                
                    for order in ['none', 'acq', 'rel', 'acq-acq', 'acq-rel']:
                        if os.path.isfile(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_ANW.txt")}') and os.path.getsize(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_ANW.txt")}') > 0:
                            print(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_ANW.txt")} already exists. Skipping')
                        else:    
                            print ("Running", new_filename, "with", order, "and", mem_type, "with", loop_count, "iterations (Array No Warmup)")
                            cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', operation, '-m', mem_type, '-c', order, '-g', order, '-l', loop_count, '-t', 'array'], capture_output=True)
                            
                            with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_ANW.txt")}', 'w') as f:
                                f.write(cuda_explorer_output.stdout.decode('utf-8'))    
                        
                        if os.path.isfile(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_A.txt")}') and os.path.getsize(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_A.txt")}') > 0:
                            print(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_A.txt")} already exists. Skipping')
                        else:    
                            print ("Running", new_filename, "with", order, "and", mem_type, "with", loop_count, "iterations (Array)")
                            cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', operation, '-m', mem_type, '-c', order, '-g', order, '-l', loop_count, '-t', 'array', '-w'], capture_output=True)
                            
                            with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_A.txt")}', 'w') as f:
                                f.write(cuda_explorer_output.stdout.decode('utf-8'))    
                    
                    for order in ['none', 'acq', 'rel', 'acq-acq', 'acq-rel']:
                        if os.path.isfile(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LLNW.txt")}') and os.path.getsize(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LLNW.txt")}') > 0:
                            print(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LLNW.txt")} already exists. Skipping')
                        else:    
                            print ("Running", new_filename, "with", order, "and", mem_type, "with", loop_count, "iterations (Array No Warmup)")
                            cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', operation, '-m', mem_type, '-c', order, '-g', order, '-l', loop_count, '-t', 'loaded'], capture_output=True)
                            
                            with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LLNW.txt")}', 'w') as f:
                                f.write(cuda_explorer_output.stdout.decode('utf-8'))    
                        
                        if os.path.isfile(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LL.txt")}') and os.path.getsize(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LL.txt")}') > 0:
                            print(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LL.txt")} already exists. Skipping')
                        else:    
                            print ("Running", new_filename, "with", order, "and", mem_type, "with", loop_count, "iterations (Array)")
                            cuda_explorer_output = subprocess.run([f'./{new_filename}', '-n', '128', '-o', operation, '-m', mem_type, '-c', order, '-g', order, '-l', loop_count, '-t', 'loaded', '-w'], capture_output=True)
                            
                            with open(f'{new_filename.replace(".out", f"_{order}_{mem_type}_{loop_count}_LL.txt")}', 'w') as f:
                                f.write(cuda_explorer_output.stdout.decode('utf-8'))    