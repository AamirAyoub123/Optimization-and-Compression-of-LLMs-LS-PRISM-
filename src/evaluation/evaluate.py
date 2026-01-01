import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def evaluate_perplexity(model, tokenizer, device):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    seq_len = 2048
    nlls = []
    
    for i in tqdm(range(0, testenc.input_ids.size(1), seq_len)):
        j = min(i + seq_len, testenc.input_ids.size(1))
        inputs = testenc.input_ids[:, i:j].to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            nlls.append(outputs.loss * (j - i))
            
    perplexity = torch.exp(torch.stack(nlls).sum() / j)
    return perplexity.item()

