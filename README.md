# When Chain-of-Thought Falls Short: Enhancing LLM Causal Reasoning via Self-Synthesized Causal Maps

This repository contains the official implementation of the paper  
**"When Chain-of-Thought Falls Short: Enhancing LLM Causal Reasoning via Self-Synthesized Causal Maps."**

The project investigates the limitations of **Chain-of-Thought (CoT)** reasoning in causal inference tasks and proposes a new **test-time structural prompting framework** that enables large language models (LLMs) to generate **self-synthesized causal maps** before reasoning.  
Experiments on **CausalBench** (covering code and mathematical causal reasoning) demonstrate that this approach significantly improves accuracy and interpretability compared to standard CoT and Self-Consistency methods.

---

## 1. Environment Setup

Create a clean virtual environment for the project:
```
conda create -n mathllm python=3.11.9
conda activate mathllm
```
Install PyTorch
```
pip install torch==2.4.0
```
> Note: Qwen3 requires transformers > 4.51.0
Then install the project dependencies:
```
pip install -r requirements.txt
```


## 2. Dataset
We use [CausalBench](https://huggingface.co/datasets/CCLV/CausalBench), a benchmark designed to evaluate causal reasoning capabilities of LLMs across four dimensions:

Cause → Effect

Effect → Cause

Cause → Effect (with intervention)

Effect → Cause (with intervention)


## 3. Reproduction

├── analysis.py              # Main script for calculting results of each method
├── Map/                     # Synthesized causal maps
├── Results/                 # Directory to store experiment outputs, logs, and accuracy reports
├── utils/                   # Utility scripts for parsing model outputs and computing metrics
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── Data/                    # Local cache for downloaded datasets


```
python analysis.py --dataset_name causal_math
```