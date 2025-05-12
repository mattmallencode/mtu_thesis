#!/bin/bash

handle_error() {
    echo "Error: $1"
    ray stop 2>/dev/null
    exit 1
}

export PYTHONPATH="../":"${PYTHONPATH}"

# Set common parameters
KB="news_qa/news_dump_splitted.csv"
PASSAGES="news_qa/knowledge_base/shards/data_0"
INDEX="news_qa/knowledge_base/news_qa.faiss"
EVAL_SET="news_qa/test.source"
GOLD_PATH="news_qa/test.target"
BASE_MODEL="facebook/rag-token-nq"
MODELS_DIR="news_qa_models/"
PREDICTIONS_DIR=predictions_${MODELS_DIR}

# Start ray
ray start --head || handle_error "Failed to start Ray cluster"

mkdir -p ${PREDICTIONS_DIR}

echo "===== PHASE 1: E2E Evaluation (EM and F1 scores) ====="

# Baseline e2e evaluation
echo "Evaluating baseline model for EM/F1..."
python eval.py \
   --model_name_or_path ${BASE_MODEL} \
   --model_type rag_token \
   --index_name custom \
   --passages_path ${PASSAGES} \
   --index_path ${INDEX} \
   --evaluation_set ${EVAL_SET} \
   --gold_data_path ${GOLD_PATH} \
   --gold_data_mode ans \
   --predictions_path predictions_baseline_e2e.txt \
   --eval_mode e2e \
   --k 5 \
   --n_docs 5 \
   --recalculate \
   --num_beams 4 \
   --min_length 1 \
   --max_length 50 \

# LoRA models e2e evaluation
for model in seed_21 seed_42 seed_84; do
 echo "Evaluating ${MODELS_DIR}${model} for EM/F1..."
 python eval.py \
   --model_name_or_path ${MODELS_DIR}${model} \
   --model_type rag_token \
   --is_peft_model \
   --base_model_path ${BASE_MODEL} \
   --index_name custom \
   --passages_path ${PASSAGES} \
   --index_path ${INDEX} \
   --evaluation_set ${EVAL_SET} \
   --gold_data_path ${GOLD_PATH} \
   --gold_data_mode ans \
   --predictions_path predictions_${MODELS_DIR}${model}_e2e.txt \
   --eval_mode e2e \
   --k 5 \
   --n_docs 5 \
   --recalculate \
   --num_beams 4 \
   --min_length 1 \
   --max_length 50
done

echo "===== PHASE 2: Retrieval Evaluation (Precision@5 and Precision@20) ====="

# Evaluate baseline and models for both k values
for k_value in 5 20; do
 echo "Evaluating baseline model for Precision@${k_value}..."
 python eval.py \
   --model_name_or_path ${BASE_MODEL} \
   --model_type rag_token \
   --index_name custom \
   --passages_path ${PASSAGES} \
   --index_path ${INDEX} \
   --evaluation_set ${EVAL_SET} \
   --gold_data_path ${GOLD_PATH} \
   --gold_data_mode ans \
   --predictions_path predictions_baseline_retrieval_k${k_value}.txt \
   --eval_mode retrieval \
   --k ${k_value} \
   --n_docs ${k_value} \
   --recalculate \
   --num_beams 4 \
   --kb_path ${KB}

 # LoRA models retrieval evaluation
 for model in seed_21 seed_42 seed_84; do
   echo "Evaluating ${MODELS_DIR}${model} for Precision@${k_value}..."
   python eval.py \
     --model_name_or_path ${MODELS_DIR}${model} \
     --model_type rag_token \
     --is_peft_model \
     --base_model_path ${BASE_MODEL} \
     --index_name custom \
     --passages_path ${PASSAGES} \
     --index_path ${INDEX} \
     --evaluation_set ${EVAL_SET} \
     --gold_data_path ${GOLD_PATH} \
     --gold_data_mode ans \
     --predictions_path predictions_${MODELS_DIR}${model}_retrieval_k${k_value}.txt \
     --eval_mode retrieval \
     --k ${k_value} \
     --n_docs ${k_value} \
     --recalculate \
     --num_beams 4 \
     --kb_path ${KB}
 done
done

echo "===== PHASE 3: Bootstrap Significance Testing ====="

# 1. EM Metric
echo "Performing bootstrap testing for EM metric..."
python eval.py \
   --compare_models \
   --model_type rag_token \
   --baseline_model_path ${BASE_MODEL} \
   --comparison_model_paths ${PREDICTIONS_DIR} \
   --index_name custom \
   --passages_path ${PASSAGES} \
   --index_path ${INDEX} \
   --evaluation_set ${EVAL_SET} \
   --gold_data_path ${GOLD_PATH} \
   --metric em \
   --gold_data_mode ans \
   --eval_mode e2e \
   --k 5 \
   --n_docs 5 \
   --bootstrap_samples 1000 \

# 2. F1 Metric
echo "Performing bootstrap testing for F1 metric..."
python eval.py \
   --compare_models \
   --model_type rag_token \
   --baseline_model_path ${BASE_MODEL} \
   --comparison_model_paths ${PREDICTIONS_DIR} \
   --index_name custom \
   --passages_path ${PASSAGES} \
   --index_path ${INDEX} \
   --evaluation_set ${EVAL_SET} \
   --gold_data_path ${GOLD_PATH} \
   --gold_data_mode ans \
   --eval_mode e2e \
   --metric f1 \
   --k 5 \
   --n_docs 5 \
   --bootstrap_samples 1000 \

#  3. Retrieval@5
echo "Performing bootstrap testing for Precision@5..."
python eval.py \
   --compare_models \
   --model_type rag_token \
   --baseline_model_path ${BASE_MODEL} \
   --comparison_model_paths ${PREDICTIONS_DIR} \
   --index_name custom \
   --passages_path ${PASSAGES} \
   --index_path ${INDEX} \
   --evaluation_set ${EVAL_SET} \
   --gold_data_path ${GOLD_PATH} \
   --gold_data_mode ans \
   --eval_mode retrieval \
   --k 5 \
   --n_docs 5 \
   --bootstrap_samples 1000 \
   --kb_path ${KB}

# 4. Retrieval@20
echo "Performing bootstrap testing for Precision@20..."
python eval.py \
    --compare_models \
    --model_type rag_token \
    --baseline_model_path ${BASE_MODEL} \
    --comparison_model_paths ${PREDICTIONS_DIR} \
    --index_name custom \
    --passages_path ${PASSAGES} \
    --index_path ${INDEX} \
    --evaluation_set ${EVAL_SET} \
    --gold_data_path ${GOLD_PATH} \
    --gold_data_mode ans \
    --eval_mode retrieval \
    --k 20 \
    --n_docs 20 \
    --bootstrap_samples 1000 \
    --kb_path ${KB}
