# Watermarkhub

This repository contains the code for the Paper: ***WatermarkHub: A Universal and Automated Watermarking Technique for Data Marketplace***.

<!-- Due to time constraints, this code submission was made hastily, but the core code is included. **We assure that we will update the code and documentation with more user-friendly versions as soon as possible.** -->

## 1 Requirements

We recommended the following dependencies.

- python==3.10.10
- langchain==0.2.9
- langchain-openai==0.1.17

For more recommended dependencies, please refer to the file [`[requirements.txt](requirements.txt)`].

``` bash
conda create -n watermark python=3.10.10 -y
conda activate watermark
pip install -r requirements.txt
```

## 2 How to use

### 2.1 Generating parse code

We use the SL.log dataset as an example.

Before generating the code, set up your api key in `.env`.

```python
API_KEY = ""
BASE_URL = ""
```

Run `chain4code_gen.py` to obtain the parse code: 

```bash
python chain4code_gen.py
```

### 2.2 Watermarking and Detecting

Run `parse_data.py` to process and watermark specified dataset : 

```bash
python parse_data.py
```
