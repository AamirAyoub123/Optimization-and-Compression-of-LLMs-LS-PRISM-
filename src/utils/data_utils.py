import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_calibration_data(model_id, n_samples=32, seq_len=512):
    # Load a small portion of WikiText
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Concatenate and tokenize
    import random
    random.seed(0)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    # Prepare batches
    batches = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        batches.append(inp)
    return batches