#!/bin/bash
#$ -q gpu@@lucy
#$ -l gpu=4
#$ -pe smp 16
#$ -M nphuc@nd.edu
#$ -m abe
#$ -N SFT-Gemma-3-4B-PT-M1

conda activate pre_rlvr
module load cuda/12.1
module load gcc/15.2.0

# python sft.py --dataset_name m1 --epochs 10 --model_name gemma-3-4b-pt
# python sft.py --dataset_name m1 --epochs 10 --model_name medgemma-4b-pt

set -x

nproc_per_node=4
model_name=gemma-3-4b-pt
save_path=checkpoints/${model_name}-m1

# Shift the arguments so $@ refers to the rest

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/m1/train.parquet \
    data.val_files=data/m1/test.parquet \
    data.train_batch_size=1 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=1 \
    +model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[Gemma3DecoderLayer]' \
    model.partial_pretrain=models/${model_name} \
    data.max_length=16384\
    ulysses_sequence_parallel_size=4 \
    optim.lr=1e-5 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=medidcal-underthink-sft \
    trainer.experiment_name=medidcal-underthink-sft-${model_name} \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]'
