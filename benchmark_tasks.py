import torch
import gc
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
ORIGINAL_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
COMPRESSED_MODEL_PATH = "./models/llama-8b-ls-prism"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TASKS = [
    {
        "category": "Summarization",
        "prompt": "Summarize this article in one sentence:\n'The Apache Hadoop software library is a framework that allows for the distributed processing of large data sets across clusters of computers using simple programming models. It is designed to scale up from single servers to thousands of machines, each offering local computation and storage.'\nSummary:",
        "type": "retention" 
    },
    {
        "category": "Extraction",
        "prompt": "Extract the names and technologies from this text: 'Elon Musk announced a new update for Tesla Autopilot involving PyTorch and CUDA.'\nAnswer:",
        "type": "retention"
    },
    {
        "category": "Creative",
        "prompt": "Write a short email declining a wedding invitation politely.\nEmail:",
        "type": "style"
    },
    {
        "category": "Logic (Hard)",
        "prompt": "Solve this: If 5 machines take 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\nAnswer:",
        "type": "reasoning"
    }
]

def generate_answers(model_path, task_list, is_original=False):
    """Generates answers for a list of tasks using a specific model."""
    print(f"\n🔄 Loading Model: {model_path}...")
    
    # Garbage Collection to ensure 8GB GPU doesn't explode
    gc.collect()
    torch.cuda.empty_cache()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Fix for some Llama tokenizers
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f" Failed to load {model_path}: {e}")
        return []

    answers = []
    model.eval()
    
    print(f" Running {len(task_list)} tasks...")
    for task in task_list:
        inputs = tokenizer(task['prompt'], return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=60, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer_only = text.replace(task['prompt'], "").strip()
        
        answers.append({
            "answer": answer_only,
            "time": end_time - start_time
        })
        print(f"   Finished: {task['category']}")

    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return answers

# --- PHASE 1: GENERATION ---
print("--- 🏁 PHASE 1: GENERATION BATTLE ---")

# 1. Run Original
orig_results = generate_answers(ORIGINAL_MODEL_ID, TASKS, is_original=True)

# 2. Run Compressed
comp_results = generate_answers(COMPRESSED_MODEL_PATH, TASKS)

if not orig_results or not comp_results:
    print("❌ Critical Error: Could not generate results.")
    exit()

# --- PHASE 2: JUDGING (Semantic Similarity) ---
print("\n--- ⚖️ PHASE 2: AI JUDGE (Semantic Similarity) ---")
print("Loading Judge Model (all-MiniLM-L6-v2)...")


judge_model = SentenceTransformer('all-MiniLM-L6-v2')

report_data = []

for i, task in enumerate(TASKS):
    orig_text = orig_results[i]['answer']
    comp_text = comp_results[i]['answer']
    
    # Calculate Similarity Score (0.0 to 1.0)
    embedding_1 = judge_model.encode(orig_text, convert_to_tensor=True)
    embedding_2 = judge_model.encode(comp_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    
    # Calculate Speedup
    time_orig = orig_results[i]['time']
    time_comp = comp_results[i]['time']
    speed_diff = ((time_orig - time_comp) / time_orig) * 100
    
    report_data.append({
        "Category": task['category'],
        "Similarity Score": f"{similarity:.4f}",
        "Orig Time": f"{time_orig:.2f}s",
        "Comp Time": f"{time_comp:.2f}s",
        "Speedup": f"{speed_diff:+.1f}%",
        "Original Answer": orig_text[:50] + "...",
        "Compressed Answer": comp_text[:50] + "..."
    })

# --- PHASE 3: REPORTING ---
df = pd.DataFrame(report_data)

print("\n" + "="*60)
print("             🏆 FINAL BATTLE REPORT             ")
print("="*60)
print(df[["Category", "Similarity Score", "Speedup"]].to_string(index=False))
print("-" * 60)
print("INTERPRETATION:")
print("1. Similarity > 0.85: The compressed model is 'Good Enough' for this task.")
print("2. Similarity < 0.50: The compressed model failed this task.")
print("3. Comparison: If Similarity is High and Speedup is Positive, YOU WON.")


df.to_csv("model_comparison_results.csv", index=False)
print("\n✅ Full results saved to 'model_comparison_results.csv'")