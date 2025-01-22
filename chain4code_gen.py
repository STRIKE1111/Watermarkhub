import time
import logging
import os
import subprocess
from config import CFG, frozen_seed, load_env_variables
from file_parsers import load_and_sample_data
from langchain_model import constrained_self_and_plan_chains
from langchain_openai import ChatOpenAI

def run_generated_code(code_path, retries=0, max_retries=5):
    try:
        result = subprocess.run(['python', code_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Error running the script: {result.stderr}")
            return False
        else:
            logging.info(f"Code ran successfully: {result.stdout}")
            return True
    except Exception as e:
        logging.error(f"Exception occurred while running the script: {e}")
        return False


def main():
    api_key, base_url = load_env_variables()
    frozen_seed(CFG.RANDOM_SEED)
    
    gpt = ChatOpenAI(model="gpt-4o", api_key=api_key, base_url=base_url, temperature=0.8, top_p=0.95)
    filepath = f"parser_data\code1\dataset\logfiles.log"
    base_filename = os.path.basename(filepath).split('.')[0]
    chain_function = constrained_self_and_plan_chains
    custom_chain = chain_function(model_name=gpt)  
    
    chain_function_name = chain_function.__name__
    print(chain_function_name)
    log_filename = f"{base_filename}-{chain_function_name}-gpt.log"
    
    log_folder = 'Lllog'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logging.basicConfig(filename=os.path.join(log_folder, log_filename), 
                        level=logging.INFO,  
                        format='%(asctime)s - %(levelname)s - %(message)s',  
                        filemode='w+')  

    data_piece = load_and_sample_data(filepath)
    
    function_gen, code, result_json, separators, nest, tags = custom_chain.invoke({"data": data_piece})

    # file_path = f"parser_data\\convert2json{i}.py"
    code_path = f"convert2json.py"
    with open(code_path, "w", encoding='utf-8') as file:
        file.write(code)
        logging.info(f"File '{code_path}' has been updated.")
    
    logging.info(f"random_object:\n {data_piece}")

    logging.info(f"separators:\n {separators}")
    logging.info(f"nest:\n {nest}")
    logging.info(f"tags:\n {tags}")
    logging.info(f"json_result:\n {result_json}")
    logging.info(f"function_gen:\n {function_gen}")
    logging.info(f"code:\n {code}")


    # Run the generated code and handle any errors, with a retry limit of 5 attempts
    success = run_generated_code(code_path, retries=0, max_retries=5)
    
    retries = 0
    while not success and retries < 5:
        logging.info(f"Attempt {retries + 1} failed. Regenerating code and trying again.")
        
        # Regenerate the code
        function_gen, code, result_json, separators, nest, tags = custom_chain.invoke({"data": data_piece})
        with open(code_path, "w", encoding='utf-8') as file:
            file.write(code)
            logging.info(f"File '{code_path}' has been updated.")

        # Run the new generated code
        success = run_generated_code(code_path, retries=retries + 1, max_retries=5)
        retries += 1

    # Log if all attempts failed
    if not success:
        logging.error("Max retries reached. Code execution failed after 5 attempts.")

if __name__ == '__main__':
    main()
