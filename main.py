import torch
import gc
import os
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Import your modules
from src.utils.activation_hooks import ActivationCollector
from src.utils.data_utils import get_calibration_data
from src.compression.svd_decomp import decompose_layer 
from src.compression.wanda_prune import apply_wanda      
from src.evaluation.evaluator import evaluate_full # Using your evaluator.py


model_id = "meta-llama/Meta-Llama-3.1-8B" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./models", exist_ok=True)


BATTLE_TASKS = [
    {
        "category": "Summarization",
        "prompt": "Summarize this article in one sentence:\n'The Apache Hadoop software library is a framework that allows for the distributed processing of large data sets across clusters of computers using simple programming models.'\nSummary:",
    },
    {
        "category": "Extraction",
        "prompt": "Extract the technologies from this text: 'Elon Musk announced a new update for Tesla Autopilot involving PyTorch and CUDA.'\nAnswer:",
    },
    {
        "category": "Creative",
        "prompt": "Write a short, polite email declining a wedding invitation.\nEmail:",
    },
    {
        "category": "Logic (Hard)",
        "prompt": "Solve this: If 5 machines take 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\nAnswer:",
    }
]

def run_battle_phase(model, tokenizer, phase_name):
    """Runs the specific battle tasks for the comparative report."""
    print(f"\n Running Battle Tasks: {phase_name}...")
    results = []
    model.eval()
    
    for task in BATTLE_TASKS:
        inputs = tokenizer(task['prompt'], return_tensors="pt").to(model.device)
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=60, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        end = time.time()
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer = full_text.replace(task['prompt'], "").strip()
        
        results.append({
            "category": task['category'],
            "answer": answer,
            "time": end - start
        })
        print(f"   ✓ Finished: {task['category']}")
        
    return results


# 1. SETUP & LOADING

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto",
    low_cpu_mem_usage=True,
    use_cache=False
)


# 2. BASELINE EVALUATION (BEFORE)

print("\n🔍 Establishing Baseline (Original Model)...")

# A. Standard Metrics (PPL, Speed)
base_metrics = evaluate_full(model, tokenizer, name="Original Llama-8B")

# B. Battle Tasks (For Judge)
base_battle = run_battle_phase(model, tokenizer, "Original")



# 3. COMPRESSION PROCESS

print("\n--- Starting Calibration ---")
collector = ActivationCollector(model)
collector.register()


samples = get_calibration_data(model_id, n_samples=32, seq_len=512) 

with torch.no_grad():
    for i, batch in enumerate(samples):
        batch = batch.to(model.device) 
        model(batch)
collector.remove()

print("\n- Starting LS-PRISM Compression -")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        layer_device = module.weight.device 

        # Stage 1: SVD for Attention
        if "self_attn" in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            parent = dict(model.named_modules())[parent_name]
            
            new_layers = decompose_layer(module)
            new_layers.to(layer_device) 
            setattr(parent, child_name, new_layers)
            
        # Stage 2: Wanda for MLP
        elif "mlp" in name:
            if name in collector.activations:
                act_norm = collector.activations[name].to(layer_device)
                apply_wanda(module, act_norm, sparsity=0.5)

# Cleanup
gc.collect()
torch.cuda.empty_cache()


# 4. COMPRESSED EVALUATION (AFTER)

print("\n🔍 Evaluating Compressed Model (LS-PRISM)...")

# A. Standard Metrics
comp_metrics = evaluate_full(model, tokenizer, name="LS-PRISM Compressed")

# B. Battle Tasks
comp_battle = run_battle_phase(model, tokenizer, "Compressed")



# 5. FINAL REPORT & JUDGING

print("\n" + "="*60)
print("              FINAL PROJECT REPORT             ")
print("="*60)

# --- PART A: PHYSICAL & EFFECTIVE SIZE ---
orig_params = 8_030_000_000
current_params = sum(p.numel() for p in model.parameters())
active_params = sum(torch.count_nonzero(p).item() for p in model.parameters())

print(f"{'METRIC':<20} | {'ORIGINAL':<12} | {'COMPRESSED':<12} | {'CHANGE'}")
print("-" * 60)

# Size Disk
disk_change = (1 - current_params/orig_params) * 100
print(f"{'Size (Disk)':<20} | {orig_params/1e9:.1f}B         | {current_params/1e9:.1f}B         | {disk_change:+.2f}% (Space Saved)")

# Size Effective
eff_change = (1 - active_params/orig_params) * 100
print(f"{'Size (Effective)':<20} | {orig_params/1e9:.1f}B         | {active_params/1e9:.1f}B         | {eff_change:+.2f}%")

# Speed
s_orig = base_metrics['speed']
s_comp = comp_metrics['speed']
s_change = ((s_comp - s_orig) / s_orig) * 100
print(f"{'Speed (Tok/s)':<20} | {s_orig:.1f}         | {s_comp:.1f}         | {s_change:+.1f}%")

# Perplexity
p_orig = base_metrics['ppl']
p_comp = comp_metrics['ppl']
p_change = p_comp - p_orig
print(f"{'Perplexity':<20} | {p_orig:.2f}        | {p_comp:.2f}        | +{p_change:.2f} (Lower is Better)")
print("-" * 60)

# --- PART B: AI JUDGE (SEMANTIC SIMILARITY) ---
print("\n  AI JUDGE REPORT (Task Accuracy)")
try:
    judge = SentenceTransformer('all-MiniLM-L6-v2')
    has_judge = True
except:
    print("(Sentence Transformers not found, skipping similarity score)")
    has_judge = False

report_rows = []
for i in range(len(BATTLE_TASKS)):
    task_cat = base_battle[i]['category']
    ans_orig = base_battle[i]['answer']
    ans_comp = comp_battle[i]['answer']
    
    row = {"Task": task_cat}
    
    if has_judge:
        emb1 = judge.encode(ans_orig, convert_to_tensor=True)
        emb2 = judge.encode(ans_comp, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2).item()
        row["Similarity"] = f"{score:.2f} / 1.00"
        
        if score > 0.85: verdict = "✅ PASS"
        elif score > 0.60: verdict = "⚠️ OK"
        else: verdict = "❌ FAIL"
        row["Verdict"] = verdict
    
    report_rows.append(row)

df = pd.DataFrame(report_rows)
print(df.to_string(index=False))

#Optional: Save Model
print("\nSaving compressed model...")
model.save_pretrained("./models/llama-8b-ls-prism")
tokenizer.save_pretrained("./models/llama-8b-ls-prism")