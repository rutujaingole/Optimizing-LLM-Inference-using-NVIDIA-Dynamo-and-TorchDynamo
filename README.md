# Optimizing LLM Inference using NVIDIA Dynamo and TorchDynamo

## Project Overview
This project explores methods to optimize inference performance for large language models, specifically BERT-base, using the following technologies:

- **Torch.compile (Inductor backend)**
- **NVIDIA Triton Inference Server**
- **NVIDIA Dynamo (from GTC 2024)**
- **PyTorch Profiler & Nsight Systems**

The task is binary sentiment classification using the **GLUE SST-2** dataset.

---

##  Problem Statement
Large LLMs like BERT are compute-heavy and often bottlenecked by inference latency and resource usage. Our goal was to optimize inference execution time and kernel efficiency across different serving backends.

---

##  Completed Milestones

- [x] Setup baseline inference using `bert-base-uncased`
- [x] Integrated TorchDynamo + Inductor backend
- [x] Served TorchScript model via Triton Inference Server
- [x] Profiled kernel- and op-level execution via PyTorch Profiler & Nsight
- [x] Compared latency across 3 settings
- [x] Analyzed top kernel bottlenecks and CPU ops

---

## Experimental Setup

- **Model**: `bert-base-uncased`
- **Task**: GLUE SST-2 (binary sentiment classification)
- **Dataset**: ~1.8k validation samples
- **Input Length**: 128 tokens (max padded)
- **Environment**: NYU HPC A100 GPU (80GB), CUDA 11.8
- **Software**: PyTorch 2.1, TorchDynamo, Triton Server v3.0, Nsight Systems, TorchProfiler

---

##  Results Summary

| Engine | Mean Latency (ms) | Speedup |
|--------|-------------------|---------|
| PyTorch Baseline | ~12.5 | 1× |
| Torch.compile (Inductor) | ~12.4 | ~1.01× |
| Triton (TorchScript + NVIDIA Dynamo) | ~9.5 | **1.30×** |

- Top kernels: attention_softmax, fused_layernorm, matmul_1
- Top CPU ops: `addmm`, `linear`, `view`

---

##  How to Run

```bash
# Clone the repo
$ git clone https://github.com/rutujaingole/Optimizing-LLM-Inference-using-NVIDIA-Dynamo-and-TorchDynamo.git
$ cd llm-inference-nvidia-dynamo

# Set up environment
$ conda create -n bertopt python=3.9
$ conda activate bertopt
$ pip install -r requirements.txt

# Run benchmarking notebook
$ jupyter notebook inferenceNVIDIA.ipynb

# Run profiler script
$ python profiler_runner.py

# (Optional) Launch Triton server
$ bash start_triton.sh
```

---

##  Repo Structure

```
.
├── inferenceNVIDIA.ipynb             # Benchmarking, profiling, plots
├── profiler_runner.py                # PyTorch Profiler driver
├── export_model.py                   # TorchScript export for Triton
├── start_triton.sh                   # Containerized Triton launcher
├── tables/                           # CSV outputs (latency, profiler)
├── figures/                          # Plots and charts
├── log/                              # Profiler traces
├── requirements.txt                  # Required packages
└── README.md                         # This file
```

---

##  Observations

- Torch.compile gave minimal improvement for BERT on single input batch
- Triton + TorchScript + NVIDIA Dynamo gave best results (~30% speedup)
- Nsight helped identify time-heavy kernels for future optimization

---
