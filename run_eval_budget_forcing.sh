#!/bin/bash
#$ -q gpu@qa-h100-001
#$ -l gpu=1
#$ -pe smp 16
#$ -M nphuc@nd.edu
#$ -m abe
#$ -N eval-tts

export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_BATCH_INVARIANT=1

conda activate pre_rlvr
module load gcc/15.2.0
module load cuda/12.1

max_budget_tokens=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7680 8192 8704 9216 9728 10240 10752 11264 11776 12288 12800 13312 13824 14336 14848 15360 15872 16384)
steps=(6400 5600 4800 4000 3200 2400 1600 800)
benchmarks=

for step in "${steps[@]}"; do
for max_budget_token in "${max_budget_tokens[@]}"; do
    # python evaluation/eval_tts.py --model_path checkpoints/medgemma-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 2 --max_budget_tokens $max_budget_token --benchmark GPQA_Medical_test --is_reasoning
    # python evaluation/eval_tts.py --model_path checkpoints/medgemma-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 4 --max_budget_tokens $max_budget_token --benchmark GPQA_Medical_test --is_reasoning
    python evaluation/eval_tts.py --model_path checkpoints/medgemma-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 10 --max_budget_tokens $max_budget_token --benchmark GPQA_Medical_test --is_reasoning
    python evaluation/eval_tts.py --model_path checkpoints/medgemma-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 10 --max_budget_tokens $max_budget_token --benchmark GPQA_Medical_test --no-is_reasoning

    # python evaluation/eval_tts.py --model_path checkpoints/medgemma-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 8 --max_budget_tokens $max_budget_token --benchmark hle_biomed --no-is_reasoning

    # python evaluation/eval_tts.py --model_path checkpoints/gemma-3-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 10 --max_budget_tokens $max_budget_token --benchmark GPQA_Medical_test --is_reasoning
    # python evaluation/eval_tts.py --model_path checkpoints/gemma-3-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 10 --max_budget_tokens $max_budget_token --benchmark GPQA_Medical_test --no-is_reasoning
    # python evaluation/eval_tts.py --model_path checkpoints/medgemma-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 10 --max_budget_tokens $max_budget_token --benchmark GPQA_Medical_test --no-is_reasoning
    # python evaluation/eval_tts.py --model_path checkpoints/gemma-3-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 10 --max_budget_tokens $max_budget_token

    # python evaluation/eval_budget_forcing.py --model_path checkpoints/medgemma-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 10 --max_budget_tokens $max_budget_token
    # python evaluation/eval_budget_forcing.py --model_path checkpoints/gemma-3-4b-pt-m1/global_step_6400/huggingface --seed 1 --n_trials 10 --max_budget_tokens $max_budget_token
done
done
    # python evaluation/eval_tts.py --model_path models/m1-7B-23K --seed $seed --n_trials $n_trial
# done
# done
# done
