# Watermarkhub

## 1 Requirements

We recommended the following dependencies.

* python==3.10
* langchain==0.2.9
* pandas==2.1.2

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
