from transformers import AutoTokenizer

def get_tokenizer():
    model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return tokenizer