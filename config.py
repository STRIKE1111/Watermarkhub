import os
from dotenv import load_dotenv, find_dotenv

class CFG:
    WM = '100011100101100011100101'
    GAMMA = 100 #original 2
    RANDOM_SEED = 42
    N_SAMPLES = 5000
    CANDIDATE_P = 0.5
    DELTA = 0.01
    PERSONAL_KEY = 'example'
    MOD = 100
    KNOWLEDGE_DATABASE = {}

def frozen_seed(seed=CFG.RANDOM_SEED):
    import random, os, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_env_variables():
    _ = load_dotenv(find_dotenv())
    # open_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["https_proxy"] = os.getenv("https_proxy")
    os.environ["http_proxy"] = os.getenv("http_proxy")

    api_key = os.getenv("API_KEY")  
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise EnvironmentError("API_KEY not found in the environment.")
    if not base_url:
        raise EnvironmentError("BASE_URL not found in the environment.")    

    return api_key, base_url
