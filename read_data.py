import os
import pandas as pd

from pprint import pprint

template = {
    "scope": "thread",
    "memory_order": "acq",
    "mem_type": "GDDR",
    "first_load": {
        "kernel": 0,
        "clock64": 0,
    },
    "second_load": {
        "kernel": 0,
        "clock64": 0,
    }
}

data = list()
pccgg_data = list()

for file in os.listdir('.'):
    if os.path.isfile(file) and '.txt' in file:
        print(file)
        with open(file, 'r') as f:
            lines = f.readlines()
            
            file_split = file.split('_')
            
            if 'PcCggPgCccggcc' not in file:    
                scope = file_split[1]
                memory_order = file_split[2]
                mem_type = file_split[3].replace('.txt', '')
                
                data_obj = {
                    "scope": scope,
                    "memory_order": memory_order,
                    "mem_type": mem_type,
                    "kernel_0": lines[14].replace('(', '').replace('ns)', '').strip(),
                    "clock64_0": lines[15].replace('(', '').replace('cycles)', '').strip(),
                    "kernel_1": lines[17].replace('(', '').replace('ns)', '').strip(),
                    "clock64_1": lines[18].replace('(', '').replace('cycles)', '').strip(),
                }
                
                data.append(data_obj)
            else:
                scope = file_split[1]
                memory_order = file_split[2]
                mem_type = file_split[3].replace('_PcCggPgCccggcc.txt', '')
                
                data_obj = {
                    "scope": scope,
                    "memory_order": memory_order,
                    "mem_type": mem_type,
                    "cpu_0": lines[23].replace('(', '').replace('ns)', '').strip(),
                    "gpu_0": lines[26].replace('(', '').replace('ns)', '').strip(),
                    "gpu_1": lines[29].replace('(', '').replace('ns)', '').strip(),
                    "gpu_2": lines[32].replace('(', '').replace('ns)', '').strip(),
                    "cpu_1": lines[35].replace('(', '').replace('ns)', '').strip(),
                    "cpu_2": lines[38].replace('(', '').replace('ns)', '').strip(),
                    "gpu_3": lines[41].replace('(', '').replace('ns)', '').strip(),
                    "gpu_4": lines[44].replace('(', '').replace('ns)', '').strip(), 
                    "cpu_3": lines[47].replace('(', '').replace('ns)', '').strip(),
                    "cpu_4": lines[50].replace('(', '').replace('ns)', '').strip(),
                }
                
                pccgg_data.append(data_obj)
            
            print("================================")
            
            
data = sorted(data, key=lambda x: x['memory_order'])
data = sorted(data, key=lambda x: x['scope'])
data = sorted(data, key=lambda x: x['mem_type'])

data_df = pd.DataFrame.from_dict(data)

# print(data_df)

data_df.to_csv('data.csv', index=False)


pccgg_data = sorted(pccgg_data, key=lambda x: x['memory_order'])
pccgg_data = sorted(pccgg_data, key=lambda x: x['scope'])
pccgg_data = sorted(pccgg_data, key=lambda x: x['mem_type'])

pccgg_data_df = pd.DataFrame.from_dict(pccgg_data)

# print(pccgg_data_df)

pccgg_data_df.to_csv('pccgg_data.csv', index=False)