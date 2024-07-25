import os
import pandas as pd

import sys

from pprint import pprint

def size_to_bytes(size_str):
    size_str = size_str.upper()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if size_str[-2:] in units:
        return int(size_str[:-2]) * units[size_str[-2:]]
    elif size_str[-1:] in units:
        return int(size_str[:-1]) * units[size_str[-1:]]
    else:
        raise ValueError(f"Unknown size unit in {size_str}")

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

def parse_op_seq(operation_sequence):
    op_seq = list(operation_sequence)

    operations = list()
    
    mode = 'X'
    
    cpu_st_c = 0
    cpu_ld_c = 0
    gpu_ld_c = 0
    gpu_st_c = 0
    
    for op in op_seq:
        if op == 'P':
            mode = 'producer'
        elif op == 'C':
            mode = 'consumer'
        elif op == 'g':
            if mode == 'producer':
                operations.append('gpu_' + mode + '_' + str(gpu_st_c))
                gpu_st_c += 1
            elif mode == 'consumer':
                operations.append('gpu_' + mode + '_' + str(gpu_ld_c))
                gpu_ld_c += 1
        elif op == 'c':
            if mode == 'producer':
                operations.append('cpu_' + mode + '_' + str(cpu_st_c))
                cpu_st_c += 1
            elif mode == 'consumer':
                operations.append('cpu_' + mode + '_' + str(cpu_ld_c))
                cpu_ld_c += 1
                
    op_seq_dict = dict()
    
    for i, op in enumerate(operations):
        op_seq_dict[op] = 3 * i + 3
        
    return op_seq_dict

def get_latencies(lines, results_line, operation_sequence):
    
    latencies = dict()
    
    for op, offset in operation_sequence.items():
        latencies[op] = dict()
        # print(op, results_line, offset)
        latencies[op]["latency"] = lines[results_line + offset].strip().split('\t')[0].replace('(', '').replace('ns)', '').strip()
        latencies[op]["latency_total"] = lines[results_line + offset].strip().split('\t')[1].replace('[', '').replace('ns]', '').strip()
    
    return latencies


operation_sequence = parse_op_seq(sys.argv[1])   

output_directory = sys.argv[2]

data = list()
pccgg_data = list()
million_data = list()

# print(os.listdir(output_directory))

for file in os.listdir(output_directory):
    if os.path.isfile(os.path.join(output_directory, file)) and '.txt' in file:
        print(file)
        with open(os.path.join(output_directory, file), 'r') as f:
            lines = f.readlines()
            
            file_split = file.replace('.txt', '').split('_')
            print(file_split) 
            memory_scope = file_split[1].split('-')[0]
            memory_order = file_split[2]
            mem_type = file_split[3]
            if memory_order == 'EmptyKernel':
                outer_loop_size = '1'
            else:
                outer_loop_size = file_split[4]
                object_type = file_split[5]
                if object_type == 'LL':
                    object_type = 'LoadedList'
                elif object_type == 'LLNW':
                    object_type = 'LoadedListNoWarmup'
                elif object_type == 'LN':
                    object_type = 'LinkedList'
                elif object_type == 'LNNW':
                    object_type = 'LinkedListNoWarmup'
                elif object_type == 'A':
                    object_type = 'Array'
                elif object_type == 'ANW':
                    object_type = 'ArrayNoWarmup'
            total_object_capacity = file_split[1].split('-')[1]
            
            if outer_loop_size == '1K':
                outer_loop_size = 1000
            elif outer_loop_size == '10K':
                outer_loop_size = 10000
            elif outer_loop_size == '100K':
                outer_loop_size = 100000
            elif outer_loop_size == '1M':
                outer_loop_size = 1000000
            elif outer_loop_size == '10M':
                outer_loop_size = 10000000
            elif outer_loop_size == '100M':
                outer_loop_size = 100000000
            elif outer_loop_size == '1B':
                outer_loop_size = 1000000000
            elif outer_loop_size == '1':
                outer_loop_size = 1
            elif outer_loop_size == '0':
                outer_loop_size = 0
            
            line_index = []
            
            for i, line in enumerate(lines):
                # if 'GPU' in line and ('ld' in line or 'st' in line):
                #     line_index.append(i+1)
                if 'Results' in line:
                    latencies = get_latencies(lines, i, operation_sequence)
            
            data_obj = {
                "scope": memory_scope,
                "memory_order": memory_order,
                "mem_type": mem_type,
                "outer_loop_size": outer_loop_size,
                "object_region_size": total_object_capacity,
                "object_type": object_type,
                # "kernel": lines[line_index[-2]].strip().split('\t')[0].replace('(', '').replace('ns)', '').strip(),
                # "kernel_total": lines[line_index[-2]].strip().split('\t')[1].replace('[', '').replace('ns]', '').strip(),
                # "kernel_sanity": lines[line_index[-1]].strip().split('\t')[0].replace('(', '').replace('ns)', '').strip(),
                # "kernel_total_sanity": lines[line_index[-1]].strip().split('\t')[1].replace('[', '').replace('ns]', '').strip(),
            }
            
            for op, op_data in latencies.items():
                data_obj[op] = op_data['latency']
                data_obj[op + '_total'] = op_data['latency_total']
            
            # data_obj['per_object'] = float(data_obj['kernel']) / int(data_obj['outer_loop_size'])
            # data_obj['per_object_sanity'] = float(data_obj['kernel_sanity']) / int(data_obj['outer_loop_size'])
            
            data.append(data_obj)
            
            # if '1M' in file:
            #     scope = file_split[1]
            #     memory_order = file_split[2]
            #     mem_type = file_split[3].replace('_1M.txt', '')
                
            #     data_obj = {
            #         "scope": scope,
            #         "memory_order": memory_order,
            #         "mem_type": mem_type,
            #         "kernel_0": lines[15].replace('(', '').replace('ns)', '').strip(),
            #         "clock64_0": lines[16].replace('(', '').replace('cycles)', '').strip(),
            #         "kernel_1": lines[18].replace('(', '').replace('ns)', '').strip(),
            #         "clock64_1": lines[19].replace('(', '').replace('cycles)', '').strip(),
            #     }
                
            #     million_data.append(data_obj)
            
            # elif 'PcCggPgCccggcc' in file:
            #     scope = file_split[1]
            #     memory_order = file_split[2]
            #     mem_type = file_split[3].replace('_PcCggPgCccggcc.txt', '')
                
            #     data_obj = {
            #         "scope": scope,
            #         "memory_order": memory_order,
            #         "mem_type": mem_type,
            #         "cpu_0": lines[23].replace('(', '').replace('ns)', '').strip(),
            #         "gpu_0": lines[26].replace('(', '').replace('ns)', '').strip(),
            #         "gpu_1": lines[29].replace('(', '').replace('ns)', '').strip(),
            #         "gpu_2": lines[32].replace('(', '').replace('ns)', '').strip(),
            #         "cpu_1": lines[35].replace('(', '').replace('ns)', '').strip(),
            #         "cpu_2": lines[38].replace('(', '').replace('ns)', '').strip(),
            #         "gpu_3": lines[41].replace('(', '').replace('ns)', '').strip(),
            #         "gpu_4": lines[44].replace('(', '').replace('ns)', '').strip(), 
            #         "cpu_3": lines[47].replace('(', '').replace('ns)', '').strip(),
            #         "cpu_4": lines[50].replace('(', '').replace('ns)', '').strip(),
            #     }
                
            #     pccgg_data.append(data_obj)
            # else:
            #     scope = file_split[1]
            #     memory_order = file_split[2]
            #     mem_type = file_split[3].replace('.txt', '')
                
            #     data_obj = {
            #         "scope": scope,
            #         "memory_order": memory_order,
            #         "mem_type": mem_type,
            #         "kernel_0": lines[15].replace('(', '').replace('ns)', '').strip(),
            #         "clock64_0": lines[16].replace('(', '').replace('cycles)', '').strip(),
            #         "kernel_1": lines[18].replace('(', '').replace('ns)', '').strip(),
            #         "clock64_1": lines[19].replace('(', '').replace('cycles)', '').strip(),
            #     }
                
            #     data.append(data_obj)
            
            print("================================")
            
            
data = sorted(data, key=lambda x: x['outer_loop_size'])
data = sorted(data, key=lambda x: x['memory_order'])
data = sorted(data, key=lambda x: x['scope'])
data = sorted(data, key=lambda x: x['mem_type'])
data = sorted(data, key=lambda x: size_to_bytes(x['object_region_size']))

data_df = pd.DataFrame.from_dict(data)

# print(data_df)

data_df.to_csv(f'{output_directory}/data.csv', index=False)


# pccgg_data = sorted(pccgg_data, key=lambda x: x['memory_order'])
# pccgg_data = sorted(pccgg_data, key=lambda x: x['scope'])
# pccgg_data = sorted(pccgg_data, key=lambda x: x['mem_type'])

# pccgg_data_df = pd.DataFrame.from_dict(pccgg_data)

# # print(pccgg_data_df)

# pccgg_data_df.to_csv('pccgg_data.csv', index=False)


# million_data = sorted(million_data, key=lambda x: x['memory_order'])
# million_data = sorted(million_data, key=lambda x: x['scope'])
# million_data = sorted(million_data, key=lambda x: x['mem_type'])

# million_data_df = pd.DataFrame.from_dict(million_data)

# # print(million_data_df)

# million_data_df.to_csv('million_data.csv', index=False)
