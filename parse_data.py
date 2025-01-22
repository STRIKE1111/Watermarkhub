import logging
import random
import pandas as pd
import os
from config import CFG, frozen_seed, load_env_variables
from data_processing import (
    DataProcessor,
    FeatureSelector,
    DomainProcessor,
)
from watermark import (
    Injection,
    Detection
)
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)     
pd.set_option('display.width', 1000)


if __name__ == '__main__':
    api_key, base_url = load_env_variables()
    frozen_seed(CFG.RANDOM_SEED)
    
    filepath = f"parser_data\code1\dataset\logfiles.log"

    base_filename = os.path.basename(filepath)
    log_filename = f"{base_filename.split('.')[0]}.log"
    log_folder = 'parser_data\code1\en_de_time_Log'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logging.basicConfig(filename=os.path.join(log_folder, log_filename), 
                        level=logging.INFO,  
                        format='%(asctime)s - %(levelname)s - %(message)s',  
                        filemode='a+')  

    logging.info(f'Dataset name: {base_filename}')
    
    processor = DataProcessor(filepath)
    DATA, parse_time = processor.get_structured_data()
    # DATA = DATA.reset_index().rename(columns={'index': 'Index'})
    print(F'data head 2: \n{DATA.head(2)}')
    print(f"data info:\n{DATA.info()}")
    DATA.to_csv(f'parser_data\\code1\\csv\\original_{base_filename.split(".")[0]}.csv', index=False)
    
    logging.info(f'DATA length: {len(DATA)}')
    logging.info(f'Parse time: {parse_time}')
    DATA = processor.str2num(DATA)    
    DATA = processor.str2category(DATA)

    data_sample = DATA.sample(frac=0.01, random_state=CFG.RANDOM_SEED)
    non_null_cols = [col for col in data_sample.columns if data_sample[col].notnull().sum() == len(data_sample)]
    
    selector = FeatureSelector()
    valid_fields = selector.get_numeric_category_cols(data_sample)
    valid_fields1 = selector.get_candidate_indices1(data_sample, non_null_cols, P=4)
    logging.info(f'valie_fields1: {valid_fields1}')

    valid_fields2 = selector.get_candidate_indices2(data_sample, valid_fields, CFG.CANDIDATE_P)
    valid_fields2 = set(random.sample(valid_fields2, 2))
    logging.info(f'valie_fields2: {valid_fields2}')

    location_fields = list(valid_fields1 | valid_fields2)
    logging.info(f'location_fields:{location_fields}')

    Indices = random.sample([data_sample.columns.to_list().index(col) for col in location_fields], min(4, max(len(valid_fields1), len(valid_fields2))))
    logging.info(f'Indices:{Indices}')
    selected_cols = [data_sample.columns.to_list()[col] for col in Indices]
    nunique_values = DATA[selected_cols].nunique()

    logging.info(f"Selected columns: {selected_cols}")
    logging.info(f"Unique value counts:\n{nunique_values}")
    # print(f'Indices:{Indices}')
    # selected_cols = [data_sample.columns.to_list()[col] for col in Indices]
    # nunique_values = DATA[selected_cols].nunique()

    # print(f"Selected columns: {selected_cols}")
    # print(f"Unique value counts:\n{nunique_values}")

    Attributes = selector.get_candidate_attributes(data_sample, valid_fields)  
    logging.info(f'Attributes: {Attributes}')
    # print(Attributes)
    
    # DATA = DATA.reset_index().rename(columns={'index': 'Index'})
    domain_gen = DomainProcessor(DATA, Attributes, CFG.DELTA)
    print('doamin generating...')
    domain_groups = domain_gen.generate_domain(api_key, base_url)
    with open('output.txt', 'w') as txt_file:
        txt_file.write(str(domain_groups))

    DATA['Index'] = range(len(DATA))  # 添加Index列
    
    DATA = DATA.reset_index().rename(columns={'index': 'Index'})
    # DATA.to_csv(f'parser_data\\statis\\original_{base_filename.split(".")[0]}.csv', index=False)
    
    Indices = [index + 1 for index in Indices]
    Indices.append(0)
    print(f'Indices:{Indices}')
    cols = [DATA.columns.to_list()[index] for index in Indices]
    print(f'cols:{cols}')

    print('watermark injecting...')
    watermarked_data, injection_time = Injection(CFG, DATA, Attributes, domain_groups, Indices, domain_gen)
    watermarked_data_path = f'parser_data\\statis\\watermark_{base_filename.split(".")[0]}.csv'
    # watermarkee_data_path1 = f'parser_data\\statis\\watermarked_{base_filename.split(".")[0]}.csv'
    
    # DATA.to_csv(watermarkee_data_path, index=False)
    watermarked_data.to_csv(watermarked_data_path, index=False)

    

    logging.info(f'Injection time: {injection_time}')
    
    print('watermark detecting...')
    with open('output.txt', 'r') as txt_file:
        content = txt_file.read()
        domain_groups = eval(content)
        
    WM_x, accuracy, detection_time = Detection(CFG, watermarked_data_path, Attributes, domain_groups, Indices, domain_gen)
    
    logging.info(f'WM_x: {WM_x}')
    logging.info(f'Accuracy: {accuracy}')
    logging.info(f'Detection time: {detection_time}')