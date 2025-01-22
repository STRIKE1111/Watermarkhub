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

    # watermarked_data = 'data_output.csv'
    # DATA.to_csv(watermarked_data, index=False)
    print('Injection completed.\n')
    end_time = time.time()  
    total_time = end_time - start_time  
    # print(f"Total injection time: {total_time:.2f} seconds")
    return DATA, total_time


def modify_dataframe(df, modify_percent):
    """
    根据给定的修改比例，对 DataFrame 的部分行进行随机修改。
    
    参数:
    df: 原始 DataFrame
    modify_percent: 修改的百分比（取值范围: 5, 10, 15, ..., 95）
    
    返回:
    修改后的 DataFrame
    """
    # 复制 DataFrame，避免修改原始数据
    df_modified = df.copy()
    
    # 计算需要修改的行数
    num_rows = len(df)
    num_modify = int(num_rows * modify_percent / 100)
    
    # 随机选择要修改的行的索引
    rows_to_modify = np.random.choice(df.index, size=num_modify, replace=False)
    
    # 对选中的行进行修改
    for row in rows_to_modify:
        # 随机选择该行的一个列进行修改
        col_to_modify = np.random.choice(df.columns)
        
        # 生成随机值进行替换，可以根据实际情况改为更具体的值生成策略
        random_value = np.random.choice(np.arange(1, 10000))  # 生成一个随机数
        
        # 修改指定的列的值
        df_modified.at[row, col_to_modify] = random_value
    
    return df_modified


def Detection(CFG, watermarked_data, selected_attributes, domain_groups, indices, domain_gen):
    DATA = pd.read_csv(watermarked_data)
    DATA = modify_dataframe(DATA, modify_percent=99)
    # original_DATA= pd.read_csv(f'parse_data')
    # DATA = DATA.sample(frac=0.1, random_state=CFG.RANDOM_SEED)
    num_bits = len(CFG.WM)
    Count = np.zeros((num_bits, 2))
    WM_x = np.zeros(num_bits)
    start_time = time.time()
    # for _, row in tqdm(DATA.iterrows(), total=len(DATA)):
    #     for index in indices:
    #         random_values = domain_gen.generate_seed(CFG.PERSONAL_KEY, row.iloc[index])
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