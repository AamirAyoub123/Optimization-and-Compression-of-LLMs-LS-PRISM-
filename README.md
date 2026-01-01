# 📉 LS-PRISM: LLM Compression Pipeline 
**Hybrid Optimization for Llama-3.1-8B using SVD & Wanda Pruning**

## 🌐 Project Overview  
In the rapidly evolving landscape of Large Language Models (LLMs), deployment on consumer-grade hardware remains a significant bottleneck. Running state-of-the-art models like **Llama-3.1-8B** typically requires expensive, high-memory enterprise GPUs.

This project implements **LS-PRISM (Large Scale - Pruning & SVD Integration for Sparse Models)**, a hybrid compression pipeline designed to democratize access to powerful LLMs. By combining structured pruning with low-rank approximation, we drastically reduce memory footprint while maintaining linguistic intelligence.

Our system bridges the gap between **research-grade capability** and **hardware accessibility**, offering a viable path to run 8B+ parameter models on standard GPUs.

## 🎯 Objectives  
- **Analyze Sparsity Trade-offs**: Conduct a comparative study between **0.3 (Conservative)** and **0.5 (Aggressive)** sparsity levels.
- **Use-Case Adaptation**: Demonstrate how sparsity settings can be tailored to specific deployment constraints (Speed vs. Accuracy).
- **Democratize LLMs** by enabling inference on consumer-grade hardware.
- Implement a **hybrid pipeline** combining SVD Decomposition and Wanda Pruning.  
- Compare **Baseline vs Compressed** model performance using automated metrics.
- Validate robustness on standard linguistic tasks (Summarization, Text Generation).
  
# 🚀 Features Overview
### **1. Hybrid Compression Engine**
Combines two distinct mathematical approaches:
- **Wanda Pruning:** Removes weights based on the product of weights and input activation magnitudes.  
- **SVD Decomposition:** Factorizes weight matrices into low-rank components ($U, \Sigma, V^T$) to reduce redundancy.

### **2. Adaptive Sparsity Strategy**
The pipeline supports flexible sparsity targets to match the deployment use case:
- **Sparsity 0.3:** Priority on Precision (Minimal information loss).
- **Sparsity 0.5:** Priority on Efficiency (Maximum speed & memory savings).

### **3. Automated Evaluation Suite**
- **Semantic Similarity:** BERTScore analysis.
- **Perplexity Benchmarking:** Measures model uncertainty on WikiText-2.
- **Inference Profiling:** Real-time tracking of token generation speed (tokens/sec).

## ⚙️ Technical Stack  
| Category | Tools / Libraries |
|----------|-------------------|
| **Language Models** | Meta Llama-3.1-8B |
| **Deep Learning** | PyTorch, Transformers, Accelerate |
| **Math & Optimization** | SciPy (SVD), NumPy, Scikit-learn |
| **Data Processing** | Datasets (WikiText-2), Einops |
| **Evaluation** | Evaluate, BERTScore, Rouge_score |
| **Machine Learning** | Transformers, PyTorch, HuggingFace |
| **Environment** | Python 3.10+, CUDA|

## 🏗️ Architecture  
### Pipeline Steps  
1. **Model Loading & Calibration**  
   - Load Llama-3.1-8B in high precision. 
   - Process calibration data (WikiText-2) to capture activation statistics.

2. **Wanda Pruning (Unstructured)**  
   - Calculate importance metric: $|W| \cdot ||X||_2$. 
   - Apply distinct sparsity masks (0.3 or 0.5) based on configuration. 

3. **SVD Decomposition (Structured)**  
   - Decompose remaining dense matrices into low-rank approximations.
   - Reconstruct layers with reduced parameter counts.

4. **Inference & Evaluation**  
   - Load compressed weights. 
   - Run generation tasks and compare against baseline metrics.


## 📊 Performance & Trade-off Analysis 

| Métrique | Baseline (Llama-3.1) | Sparsity 0.3 (Conservative) | Sparsity 0.5 (Aggressive) |
|----------|----------|------------|------------|
| **Effective Size** | 8.0 B Params | 6.2 B Params | 5.1 B Params |
| **Parameter Reduction** | 0% | ~22% | 37% |
| **Inference Speed (Raw)** | 2.7 tok/s | 4.1 tok/s | 4.1 tok/s |
| **Inference Speedup (%)** | - | ⚡ +51% | ⚡ +51% |
| **Perplexity (PPL)** | 1.10 | 1.71 | 1.62 |
| **Semantic Score** | 1.0 (Ref) | 0.96 (Near Perfect) | 0.89 (High Retention) |



### Key Insights: Choosing the Right Sparsity
The choice of sparsity is dictated by the specific **Use Case:**
- **Use Case A: Critical NLP Tasks (Sparsity = 0.3)**
   - Scenario: Legal analysis, Medical summarization.
   - Decision: We prioritize semantic accuracy **(0.96 similarity)**. Although the Perplexity (PPL) increases to 1.71, the model retains nearly all its reasoning capabilities while fitting into smaller memory (6.2B params).
- **Use Case B: Real-Time / Edge Deployment (Sparsity = 0.5)** 
   - Scenario: Chatbots, Live translation, Mobile deployment.
   - Decision: We prioritize efficiency. At **0.5 Sparsity**, the model shrinks to **5.1B parameters**, achieving a steady **4.1 tok/s** on consumer hardware. The perplexity stabilizes at **1.62**, offering an excellent trade-off for speed-sensitive applications. 


---

## 📁 Repository Structure

```
Comp-LLM-Project/
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Git configuration
│
├── src/                          # Core source code
│   ├── compression/
│   │   ├── wanda_prune.py        # Wanda pruning logic
│   │   ├── svd_decomp.py         # SVD Matrix Decomposition
│   │   └── kmif.py               # KMIF compression utility
│   │
│   ├── layers/
│   │   └── low_rank_layer.py     # Custom PyTorch layers
│   │
│   ├── evaluation/
│   │   ├── evaluator.py          # Main evaluation pipeline
│   │   ├── evaluate_model.py     # Metrics calculation
│   │   └── evaluate.py           # Evaluation runner script
│   │
│   └── utils/
│       ├── activation_hooks.py   # Capture forward pass activations
│       └── data_utils.py         # Calibration data loader
│
├── benchmark_tasks.py            
├── check_stats.py                
└── main.py                       # CLI Entry point

```

---


## 🚀 Quick Start  


## 🧩 Prerequisites

Before running the application, ensure that your environment meets the following requirements:

### **System Requirements**
- Python **3.10 or higher** (fully compatible with **3.11**)
- CUDA-compatible GPU (Recommended)
- Minimum **16 GB RAM**

### **Internet Access**
- An active internet connection is required only during the first launch to download artifacts to your local cache:
  - **Base Model:** meta-llama/Meta-Llama-3.1-8B (Requires Hugging Face Token).
  - **Judge Model:** all-MiniLM-L6-v2 (SentenceTransformer used for semantic similarity evaluation).
  - **Calibration Data:** wikitext-2-raw-v1.
- **After the models are downloaded, the entire system can run offline.**



# 🚀 How to Run the Application From Scratch

Follow this step-by-step guide to set up and launch the entire application starting from a clean environment.

---

## 1. Setup Environment
```bash
# 1. Clone the repository
git clone https://github.com/AamirAyoub123/LS-PRISM.git
cd LS-PRISM

# 2. Create Conda environment 
conda create -n ls-prism python=3.10 -y

# 3. Activate the environment
conda activate ls-prism

# 4. Install dependencies
pip install -r requirements.txt
```
  
## 2. Run Compression (Select your Use Case)
 - **Option A: Conservative (High Accuracy)**
```bash
python main.py \
  --model_id "meta-llama/Meta-Llama-3.1-8B" \
  --method hybrid \
  --sparsity 0.3 \
  --save_path "./output/llama_conservative"
```

- **Option B: Aggressive (High Speed/Efficiency)**
```bash
python main.py \
  --model_id "meta-llama/Meta-Llama-3.1-8B" \
  --method hybrid \
  --sparsity 0.5 \
  --save_path "./output/llama_aggressive"
```

## 3. Evaluate Performance
Compare the compressed model against the baseline:

```bash
python benchmark_tasks.py \
  --model_path "./output/llama_aggressive" \
  --task summarization
```

---

## 📜 License

This project is open-source and available for educational and research purposes.

* **Llama 3.1 Community License applies to the base model weights.**

---

## 👨‍💻 Author
**Ayoub Aamir**  

🎓 **Master Big Data & IoT**  
📍 *ENSAM Casablanca*  
📧 [aamir.ayoub@ensam-casa.ma](mailto:aamir.ayoub@ensam-casa.ma)

🔗 **Connect with me:**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayoub-aamir)  
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AamirAyoub123)




