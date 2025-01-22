# data_processing.py
import pandas as pd
import numpy as np
from generated_code.convert2json import parseCode
from file_parsers import read_file
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import hashlib
import random
import ast
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def get_structured_data(self):
        start_time = time.time()
        objects = read_file(self.file_path)
        df = pd.DataFrame(objects, columns=['objects'])
        df = df.reset_index().rename(columns={'index': 'Index'})
        df['formatted_objects'] = df['objects'].apply(lambda x: {"data": x})
        df['json_objects'] = df['formatted_objects'].apply(parseCode)
        # structured_data = pd.json_normalize(df['json_objects'].apply(json.loads))
        structured_data = pd.json_normalize(df['json_objects'])
        end_time = time.time()
        total_time = end_time - start_time
        return structured_data, total_time

    def str2num(self, df):
        for column in df.columns:
            if df[column].dtype == 'object':
                try:
                    # df[column] = pd.to_numeric(df[column], errors='coerce')
                    df[column] = pd.to_numeric(df[column], errors='raise')
                
                except Exception as e:
                    print(f"Warning: Column '{column}' could not be fully converted. Error: {e}")
                    df[column] = df[column] 
        return df

    def str2category(self, df):
        for column in df.select_dtypes(include='object').columns:
            try:
                if df[column].apply(lambda x: isinstance(x, list)).any():
                    df[column] = df[column].apply(
                        lambda x: tuple(x) if isinstance(x, list) and all(isinstance(item, str) for item in x) else x
                    )
                unique_count = df[column].nunique()
                if 3 <= unique_count < 50:
                    df[column] = df[column].astype('category')
                else:
                    df[column] = df[column].astype('object')
            except Exception as e:
                print(f"Column '{column}' skipped due to error: {e}")
        return df

class FeatureSelector:
    @staticmethod
    def get_numeric_category_cols(df):
        categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # if len(numerical_cols) > 20:
        #     return numerical_cols
        return categorical_cols + numerical_cols

    @staticmethod
    def get_candidate_indices1(df, valid_fields, P):
        sorted_fields = df[valid_fields].nunique().sort_values(ascending=False).index.to_list()
        # valid_fields1 = set([item for item in sorted_fields[:int(len(sorted_fields) * P)]])
        # valid_fields1 = set([item for item in sorted_fields[:4]])

        if isinstance(P, int):
            valid_fields1 = set(sorted_fields[:P])
        elif isinstance(P, float) and 0 < P <= 1:
            valid_fields1 = set(sorted_fields[:int(len(sorted_fields) * P)])
        else:
            raise ValueError("P must be a float (0 < P <= 1) or an integer.")

        return valid_fields1

    @staticmethod
    def get_candidate_indices2(df, valid_cols, P):
        mi_matrix = pd.DataFrame(index=valid_cols, columns=valid_cols)
        categorical_cols = df.select_dtypes(include=['category']).columns.tolist()

        for col_x in valid_cols:
            for col_y in valid_cols:
                if col_x == col_y:
                    mi_matrix.loc[col_x, col_y] = None
                else:
                    try:
                        if col_x in categorical_cols or col_y in categorical_cols:
                            if col_y in categorical_cols:
                                mi = mutual_info_classif(df[[col_x]], df[col_y].astype('category').values.ravel())
                            else:
                                mi = mutual_info_classif(df[[col_y]], df[col_x].astype('category').values.ravel())
                        else:
                            mi = mutual_info_regression(df[[col_x]], df[[col_y]].values.ravel())

                        mi_value = mi[0] if isinstance(mi, (list, np.ndarray)) else mi
                        mi_matrix.loc[col_x, col_y] = mi_value
                        mi_matrix.loc[col_y, col_x] = mi_value  
                    except ValueError as e:
                        # print(f"Error calculating MI for columns '{col_x}' and '{col_y}': {e}")
                        mi_matrix.loc[col_x, col_y] = None
                        mi_matrix.loc[col_y, col_x] = None 

        mi_matrix.fillna(0, inplace=True)
        positive_counts = mi_matrix.sum()
        sorted_positive_counts = positive_counts.sort_values(ascending=False)
        # valid_fileds2 = set(sorted_positive_counts[:int(len(sorted_positive_counts) * P)].index.to_list())
        # valid_fileds2 = set(sorted_positive_counts[:4].index.to_list())

        if isinstance(P, int):
            valid_fields2 = set(sorted_positive_counts[:P].index.to_list())
        elif isinstance(P, float) and 0 < P <= 1:
            valid_fields2 = set(sorted_positive_counts[:int(len(sorted_positive_counts) * P)].index.to_list())
        else:
            raise ValueError("P must be a float (0 < P <= 1) or an integer.")


        return valid_fields2

    

    @staticmethod
    def get_candidate_attributes(sampled_data, valid_fields, max_fields=10):
        if len(valid_fields) >= max_fields:
            numeric_columns = sampled_data.select_dtypes(include=[int, float]).columns
            if len(numeric_columns) >= max_fields:
                # unique_counts = {col: sampled_data[col].nunique() for col in numeric_columns}
                # sorted_candidates = sorted(unique_counts.items(), key=lambda x: x[1], reverse=True)

                # Attributes = [col for col, count in sorted_candidates[:10]]
                unique_counts = sampled_data[numeric_columns].nunique().sort_values(ascending=False)
                Attributes = unique_counts.index[:max_fields].tolist()

            else:
                category_columns = sampled_data.select_dtypes(include=['category']).columns
                # sorted_category_columns = sorted(category_columns, key=lambda col: sampled_data[col].notnull().sum())
                # Attributes = list(numeric_columns)
                # Attributes += list(sorted_category_columns[:(10 - len(Attributes))])
                sorted_category_columns = category_columns[sampled_data[category_columns].notnull().sum().sort_values().index]
                Attributes = list(numeric_columns) + list(sorted_category_columns[:(max_fields - len(numeric_columns))])

        else:
            Attributes = valid_fields   

        return Attributes


class DictOutputParser(BaseOutputParser):
    def parse(self, text: str) -> set:
        # print(text)
        result_dict = ast.literal_eval(text.strip())
        # result_set = set(result_dict.keys())
        return result_dict

class DomainProcessor:
    def __init__(self, df, candidate_keys, delta):
        self.df = df
        self.candidate_keys = candidate_keys
        self.delta = delta

    def synonyms_set(self, data, api_key, base_url):
        synonyms_gen_template = """
        Now you are an expert in generating synonyms. I will provide you with data in the form of a Python list. 
        For each element in the list, you need to generate a set of exactly 3-5 synonyms, including the original element in the set. 
        If the original word is in all uppercase, ensure the generated synonyms are also in all uppercase. 
        Maintain the same case sensitivity as the original words (e.g., if the original word is capitalized, the synonyms should also be capitalized if applicable). 
        Additionally, ensure that the synonyms are contextually appropriate and maintain the same meaning as the original words. 
        The final return format should be a dictionary with the original element as the key, and a set of synonyms (including the original element) as the value. 
        **Do not include any code block markers such as triple backticks (` ``` `) or language identifiers like `python` in the response.** 
        Return only the result in this format: {{element: {{synonym1, synonym2, synonym3, synonym4, synonym5}}, ...}}.
        Ensure that all strings are properly escaped and enclosed in double quotes to avoid parsing errors.
        data: {data}
        """
        
        model_4_mini = ChatOpenAI(model="gpt-4o", temperature=0.8, top_p=0.95, api_key=api_key, base_url=base_url)

        synonyms_gen_prompt = PromptTemplate.from_template(synonyms_gen_template)
        synonyms_gen_chain = synonyms_gen_prompt | model_4_mini | DictOutputParser()
        final_result = synonyms_gen_chain.invoke({"data": data})
        return final_result


    def dp_grouping(self, partition):
        """
        对一个分区使用动态规划分组，最小化组内差异总和。
        """
        n = len(partition)
        if n < 2:
            return [partition]  
        
        
        dp = [float('inf')] * n 
        split_point = [-1] * n  
        dp[0] = float('inf')  
        dp[1] = partition[1] - partition[0]  
        
        
        for i in range(2, n):
            for j in range(i - 1):  
                cost = dp[j] + (partition[i] - partition[j + 1]) 
                if cost < dp[i]:
                    dp[i] = cost
                    split_point[i] = j
        
        
        groups = []
        i = n - 1
        while i > 0:
            start = split_point[i] + 1
            groups.append(partition[start:i + 1])
            i = split_point[i]
        return groups[::-1]  


    def generate_domain(self, api_key, base_url):
        partitions = {}
        for key in self.candidate_keys:
            # result[key] = {}
            partitions[key] = []
            if isinstance(self.df[key].dtype, pd.CategoricalDtype):
                partitions[key] = self.synonyms_set(self.df[key].dropna().unique().tolist(), api_key=api_key, base_url=base_url)
            if pd.api.types.is_integer_dtype(self.df[key]) or pd.api.types.is_float_dtype(self.df[key]):  # 检查列是否为数值型
                print(f"key:{key} \nvalues:{self.df[key].dtype}")
                print(f"example: {self.df[key].head().tolist()}")

                for value in self.df[key].unique():
                    assigned = False
                    
                    for partion in partitions[key]:
                        
                        if abs(value - partion[0]) <= partion[0] * self.delta:
                            partion.append(value)
                            assigned = True
                            break
                    
                    if not assigned:
                        partitions[key].append([value])
                
                for partition in partitions[key]:
                    partition.sort()
                
                column_groups = []
                for partition in partitions[key]:
                    if len(partition) > 1:
                        column_groups.extend(self.dp_grouping(partition))
                    else:
                        column_groups.append(partition)  # 单值分区直接添加
                
                partitions[key] = column_groups
            
        return partitions
            

    @staticmethod
    def Hash(key, index, extra_param=None):
        if extra_param is not None:
            combined = f"{key}{index}{extra_param}"
        else:
            combined = f"{key}{index}"
        num = int(hashlib.sha256(combined.encode('utf-8')).hexdigest(), 16)
        return num % (2**32)
    
    # def generate_seed(key, index):
    #     random.seed(DomainProcessor.Hash(key, index))
    #     random_values = [int(random.random() * 1e16) for _ in range(3)]
    #     return random_values
    
    @staticmethod
    def generate_seed(key, index, extra_param=None):
        if extra_param is not None:
            random.seed(DomainProcessor.Hash(key, index, extra_param))
        else:
            random.seed(DomainProcessor.Hash(key, index))
        random_values = [int(random.random() * 1e16) for _ in range(3)]
        return random_values


    @staticmethod        
        
    def find_closest_value(v, domain_group):
        if isinstance(v, (int, float)):
            for _, sublist in enumerate(domain_group):
                if v in sublist:
                    return sublist
        else:
            closest_key = None
            for key, value in domain_group.items():
                value.add(key)
                if v in value:
                    closest_key = key
                    return domain_group.get(closest_key)

def convert_int64_to_int(data):
    new_data = {}
    for key, value in data.items():
        new_value = {int(inner_key): inner_value for inner_key, inner_value in value.items()}
        new_data[key] = new_value
    return new_data

def convert_sets_to_lists(data):
    for key, value in data.items():
        for inner_key, inner_value in value.items():
            value[inner_key] = list(inner_value)
    return data

def convert_lists_to_sets(data):
    for key, value in data.items():
        for inner_key, inner_value in value.items():
            value[inner_key] = set(inner_value)
    return data