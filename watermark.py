import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

def Injection(CFG, DATA, selected_attributes, domain_groups, indices, domain_gen):
    start_time = time.time()  

    for ex, row in tqdm(DATA.iterrows(), total=len(DATA)):
        for index in indices:
            random_values = domain_gen.generate_seed(CFG.PERSONAL_KEY, row.iloc[index])
            if random_values[0] % CFG.GAMMA == 0:
                i = random_values[1] % len(selected_attributes)
                j = random_values[2] % len(CFG.WM)
                v = row[selected_attributes[i]]
                if not pd.isna(v):
                    group = domain_gen.find_closest_value(v, domain_groups[selected_attributes[i]])
                    if group is not None and len(group) > 1:
                        k = domain_gen.Hash(CFG.PERSONAL_KEY, row.iloc[index], CFG.WM[j]) % len(group)
                        # print(f'Modifying row {ex}, column {selected_attributes[i]}: {v} -> {list(group)[k]}')

                        if isinstance(DATA[selected_attributes[i]].dtype, CategoricalDtype) and list(group)[k] not in DATA[selected_attributes[i]].cat.categories:
                            DATA[selected_attributes[i]] = DATA[selected_attributes[i]].cat.add_categories(list(group)[k])
                        DATA.at[ex, selected_attributes[i]] = list(group)[k]

    print('Injection completed.\n')
    end_time = time.time()  
    total_time = end_time - start_time  
    return DATA, total_time


def modify_dataframe(df, modify_percent):
    df_modified = df.copy()
    num_rows = len(df)
    num_modify = int(num_rows * modify_percent / 100)
    rows_to_modify = np.random.choice(df.index, size=num_modify, replace=False)
    for row in rows_to_modify:
        col_to_modify = np.random.choice(df.columns)
        random_value = np.random.choice(np.arange(1, 10000))  #
        df_modified.at[row, col_to_modify] = random_value
    
    return df_modified


def Detection(CFG, watermarked_data, selected_attributes, domain_groups, indices, domain_gen):
    DATA = pd.read_csv(watermarked_data)
    DATA = modify_dataframe(DATA, modify_percent=99)
    
    num_bits = len(CFG.WM)
    Count = np.zeros((num_bits, 2))
    WM_x = np.zeros(num_bits)
    start_time = time.time()
    
    for i, row in tqdm(DATA.iterrows(), total=len(DATA)):
        for index in indices:
            # print(f'index: {index}, row.iloc[index]: {row.iloc[index]}')
            random_values = domain_gen.generate_seed(CFG.PERSONAL_KEY, row.iloc[index])
            if random_values[0] % CFG.GAMMA == 0:
                i = random_values[1] % len(selected_attributes)
                j = random_values[2] % len(CFG.WM)
                v = row[selected_attributes[i]]
                if not pd.isna(v):
                    group = domain_gen.find_closest_value(v, domain_groups[selected_attributes[i]])
                    # print(seed_values, i, j, v, group)
                    if group is not None and len(group) > 1:
                        k = domain_gen.Hash(CFG.PERSONAL_KEY, row.iloc[index], CFG.WM[j]) % len(group)
                        if list(group)[k] == v:
                            Count[j][eval(CFG.WM[j])] += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    # print(f"Total detection time: {total_time:.2f} seconds")
    for b in range(num_bits):
            if Count[b][1] > Count[b][0]:
                WM_x[b] = 1
            elif Count[b][1] < Count[b][0]:
                WM_x[b] = 0
            else:
                print(f'watermark_x[{b}] not detected.')
    
    WM = np.array([eval(bit) for bit in CFG.WM])  
    correct_bits = np.sum(WM == WM_x)
    print(Count)  
    accuracy = (correct_bits / num_bits) * 100

    print(f"Watermark detection accuracy: {accuracy:.2f}%")
    
    return WM_x, accuracy, total_time