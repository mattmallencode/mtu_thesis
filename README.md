# Parameter-Efficient Domain Adaptation for RAG

This repository contains the implementation for the thesis "Reducing the Resources Required for Domain Adaptation of Pre-Trained RAG Models: A Parameter-Efficient Approach" by Matt Mallen.

## Overview

This project introduces parameter-efficient approaches for domain adaptation of Retrieval-Augmented Generation (RAG) models. The implementation includes:

- LoRA adaptation for RAG components
- Custom P-tuning v2 implementation for DPR
- Novel document adapter approach
- Stratified random sampling for resource-constrained validation

## Setup Instructions

1. Clone this repository
2. Download and unzip the RAG-end2end dependency in the project directory:
   ```
   wget https://github.com/huggingface/transformers-research-projects/archive/refs/heads/main.zip
   unzip main.zip
   cp -r transformers-research-projects-main/rag-end2end-retriever ./ragend2end
   ```
3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1up3yKcJFArBQ6e0F_6n_mfW1VPHxA20A)

## Usage

Use the provided shell scripts to run experiments:

- `finetune.sh`: Train a parameter-efficient RAG model
- `eval.sh`: Evaluate a trained model

## Key Components

- `peft_module.py`: PEFT integration for RAG
- `process_dataset.py`: Stratified sampling implementation
- `p_tuningv2.py`: Custom P-tuning v2 for DPR
- `document_adapter.py`: Novel document adaptation approach
- `eval.py`: Enhanced evaluation utilities
- `finetune_rag.py`: Main training pipeline for parameter-efficient RAG fine-tuning

## Results

The parameter-efficient approach using LoRA achieves significant improvements across all datasets (1.75–3.06× for exact match and 1.69–3.02× for F1) while modifying only 0.12% of model parameters.

## Citation

If you use this code in your research, please cite:
```
Mallen, M. (2025). Reducing the Resources Required for Domain Adaptation of Pre-Trained RAG Models: A Parameter-Efficient Approach. Master's thesis, Munster Technological University Cork.
```
