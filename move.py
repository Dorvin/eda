import os
import shutil

datas = [0, 1]
main_inv = ['_False_until' , '_True_until']
methods = [0, 1, 5]
tenfold = range(0, 10)
train_val = ['train', 'val']
norm_inv = ['norm', 'inv']

print('Moving files...')
for data in datas:
    for i in (0,):
        mid = main_inv[i]
        ni = norm_inv[i]
        for m in methods:
            for ten in tenfold:
                for tv in train_val:
                    path = f'./logs/{data}{mid}{m}/{ten}/{tv}'
                    file_list = os.listdir(path)
                    for fn in file_list:
                        try:
                            if not os.path.exists(f'./result/data{data}_{ni}/{tv}/d{m}'):
                                os.makedirs(f'./result/data{data}_{ni}/{tv}/d{m}')
                        except OSError:
                            print("Err: can not make directory "+ f'./result/data{data}_{ni}/{tv}/d{m}')
                        shutil.copy2(f'./logs/{data}{mid}{m}/{ten}/{tv}/{fn}', f'./result/data{data}_{ni}/{tv}/d{m}/{fn}')
print('Done!')



