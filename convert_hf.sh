# steps=(800 1600 2400)
steps=(800)
# model_name=gemma-3-4b-pt-m1
model_name=medgemma-4b-pt-m1

for step in "${steps[@]}";do
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/${model_name}/global_step_${step}\
    --target_dir checkpoints/${model_name}/global_step_${step}/huggingface\

done