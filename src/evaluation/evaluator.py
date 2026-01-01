import torch
import time
import torch.nn as nn
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, text_sample, device):
    """
    Calculates Perplexity (PPL). 
    Lower PPL = The model understands language better.
    """
    encodings = tokenizer(text_sample, return_tensors="pt")
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    # Evaluate using sliding window
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Negative Log Likelihood
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def measure_speed(model, tokenizer, device):
    """
    Measures generation speed in Tokens Per Second (TPS).
    """
    prompt = "Write a python function to calculate the Fibonacci sequence."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, min_new_tokens=50)
    end = time.time()
    
    duration = end - start
    tokens_per_sec = 50 / duration
    return tokens_per_sec

def run_task_benchmark(model, tokenizer, device):
    """
    Tests the model on 3 specific tasks to see which one survived compression.
    """
    tasks = {
        "Coding": "Write a Python function to merge two sorted lists.",
        "Logic": "If I have 3 apples and eat one, how many do I have?",
        "Summary": "Summarize this: The quick brown fox jumps over the lazy dog."
    }
    
    results = {}
    print("\n   --- Task Output Samples ---")
    
    for task_name, prompt in tasks.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        #
        results[task_name] = decoded
        print(f"   [{task_name}]: {decoded[:60]}...") # Print preview
        
    return results

def evaluate_full(model, tokenizer, name="Model"):
    print(f"\n Evaluating: {name}...")
    model.eval()
    device = model.device
    
    # 1. Speed Test
    tps = measure_speed(model, tokenizer, device)
    print(f"   - Speed: {tps:.2f} tokens/sec")
    
    # 2. Perplexity Test 
    sample_text = "The study of artificial intelligence is the study of intelligent agents. " * 50
    ppl = calculate_perplexity(model, tokenizer, sample_text, device)
    print(f"   - Perplexity (PPL): {ppl:.2f}")

    # 3. Qualitative Tasks
    task_out = run_task_benchmark(model, tokenizer, device)
    
    return {"speed": tps, "ppl": ppl, "tasks": task_out}