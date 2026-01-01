
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Path to your saved compressed model
MODEL_PATH = "./models/llama-8b-ls-prism"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading compressed model from {MODEL_PATH}...")
try:
    # Load the compressed model
    # We use float16 and device_map="auto" to fit on 8GB GPU + RAM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

# Test Prompts
prompts = [
    "The capital of France is",
    "Python is a programming language that",
    "To clean a dataset in SQL, you should"
]

print("\n--- GENERATION TEST ---")
model.eval()

for prompt in prompts:
    print(f"\n Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Measure generation time
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7
        )
    end = time.time()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Output: {generated_text}")
    print(f"⏱️ Time: {end - start:.2f}s")

print("\n--- TEST FINISHED ---")