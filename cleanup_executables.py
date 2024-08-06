import os

for file in os.listdir('.'):
    if os.path.isfile(file) and '.ptx' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('134217724.', '256MB.').replace('67108860.', '128MB.').replace('33554428.', '64MB.').replace('16777212.', '32MB.').replace('8388604.', '16MB.').replace('4194300.', '8MB.').replace('2097148.', '4MB.').replace('1048572.', '2MB.').replace('524284.', '1MB.').replace('262140.', '512KB.').replace('131068.', '256KB.').replace('65532.', '128KB.').replace('32764.', '64KB.').replace('16380.', '32KB.').replace('8188.', '16KB.').replace('4092.', '8KB.').replace('2044.', '4KB.').replace('1020.', '2KB.').replace('508.', '1KB.').replace('252.', '512B.').replace('188.', '384B.').replace('124.', '256B.').replace('92.', '192B.').replace('60.', '128B.').replace('28.', '64B.').replace('12.', '32B.').replace('.ptx', '') + '.ptx'
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)

    if os.path.isfile(file) and '.sass' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('134217724.', '256MB.').replace('67108860.', '128MB.').replace('33554428.', '64MB.').replace('16777212.', '32MB.').replace('8388604.', '16MB.').replace('4194300.', '8MB.').replace('2097148.', '4MB.').replace('1048572.', '2MB.').replace('524284.', '1MB.').replace('262140.', '512KB.').replace('131068.', '256KB.').replace('65532.', '128KB.').replace('32764.', '64KB.').replace('16380.', '32KB.').replace('8188.', '16KB.').replace('4092.', '8KB.').replace('2044.', '4KB.').replace('1020.', '2KB.').replace('508.', '1KB.').replace('252.', '512B.').replace('188.', '384B.').replace('124.', '256B.').replace('92.', '192B.').replace('60.', '128B.').replace('28.', '64B.').replace('12.', '32B.').replace('.sass', '') + '.sass'
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)

    if os.path.isfile(file) and os.access(file, os.X_OK) and '.out' in file:
        new_filename = file.replace('cuda::thread_scope_', '').replace('134217724.', '256MB.').replace('67108860.', '128MB.').replace('33554428.', '64MB.').replace('16777212.', '32MB.').replace('8388604.', '16MB.').replace('4194300.', '8MB.').replace('2097148.', '4MB.').replace('1048572.', '2MB.').replace('524284.', '1MB.').replace('262140.', '512KB.').replace('131068.', '256KB.').replace('65532.', '128KB.').replace('32764.', '64KB.').replace('16380.', '32KB.').replace('8188.', '16KB.').replace('4092.', '8KB.').replace('2044.', '4KB.').replace('1020.', '2KB.').replace('508.', '1KB.').replace('252.', '512B.').replace('188.', '384B.').replace('124.', '256B.').replace('92.', '192B.').replace('60.', '128B.').replace('28.', '64B.').replace('12.', '32B.')
        
        print("Renaming", file, "to", new_filename)
        os.rename(file, new_filename)