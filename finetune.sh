#!/bin/bash

handle_error() {
    echo "Error: $1"
    ray stop 2>/dev/null
    exit 1
}

export PYTHONPATH="../":"${PYTHONPATH}"

if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init || handle_error "Failed to initialize git repository"

    if ! git config --get user.email >/dev/null; then
        git config --local user.email "rag@.com"
        git config --local user.name "RAG Training"
    fi

    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*.so

# Directories
wandb/
outputs/
logs/
checkpoints/

# Environment
.env
venv/
.venv/

# Model artifacts
*.ckpt
*.pt
*.bin

# Data
*.csv
*.json
*.jsonl
*.txt
!requirements.txt

# IDE
.vscode/
.idea/
EOL
    fi

    git add .gitignore
    git commit -m "Initial commit with .gitignore" || handle_error "Failed to make initial commit"
    if ! git show-ref --verify --quiet refs/heads/main; then
        git checkout -b main || handle_error "Failed to create main branch"
    fi

    echo "Git repository initialized successfully"
fi

ray start --head || handle_error "Failed to start Ray cluster"

python finetune_rag.py \
    --model_name_or_path facebook/rag-token-nq \
    --model_type rag_token \
    --seed 84 \
    --fp16 \
    --gpus 1  \
    --do_predict \
    --do_train \
    --n_val 500 \
    --data_dir news_qa \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --max_source_length 128 \
    --output_dir checkpoints \
    --save_top_k 1 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 1.0 \
    --lr_scheduler polynomial \
    --learning_rate 2e-04 \
    --num_train_epochs 10 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 16 \
    --distributed_retriever ray \
    --num_retrieval_workers 1  \
    --index_name custom \
    --passages_path news_qa/knowledge_base/shards/data_0 \
    --shards_dir news_qa/knowledge_base/shards/data_0 \
    --index_path news_qa/knowledge_base/news_qa.faiss \
    --context_encoder_name facebook/dpr-ctx_encoder-single-nq-base \
    --csv_path news_qa/knowledge_base/news_dump_splitted.csv \
    --index_gpus 1 \
    --gpu_order [0] \
    --indexing_freq 500 || handle_error "Training failed"

ray stop
