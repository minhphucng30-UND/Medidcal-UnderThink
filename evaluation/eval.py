import argparse
import torch
import numpy as np
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import os
import json
from data import *
from collect_eval_data import load_huatuo_eval
from extract_format import extract_answer, huatuo_match_choice


def compute_score(output, answer_string, answer_letter, option_str):
    extracted_answer = extract_answer(output)
    huatuo_extracted_answer = None
    correct = False
    if extracted_answer.lower() == answer_string.lower():
        correct = True
    elif extracted_answer.lower() == f"{answer_letter}. {answer_string}".lower():
        correct = True
    elif (
        len(extracted_answer) == 1
        and extracted_answer.lower() == answer_letter.lower()
    ):
        correct = True
    else:
        huatuo_extracted_answer = huatuo_match_choice(extracted_answer, option_str)
        if huatuo_extracted_answer.lower() == answer_letter.lower():
            correct = True
    return correct



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/gemma-3-4b-pt/m1-sft-epoch9")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_budget_tokens", type=int, default=16384)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens
    seed = args.seed
    eval_data = load_huatuo_eval()
    metrics = {}
    metrics['model_path'] = "/".join(model_path.split("/")[1:])
    print(metrics['model_path'])
    n_gpus = torch.cuda.device_count()
    llm = LLM(model_path, enforce_eager=False, max_num_seqs=1024, max_model_len=32768, tensor_parallel_size=n_gpus, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    metrics['temperature'] = temperature
    metrics['top_p'] = top_p
    metrics['max_tokens'] = max_tokens
    metrics['seed'] = seed

    for k in eval_data.keys():
        prompts = [ele['question'] for ele in eval_data[k]]
        answer_idxs = [ele['answer_idx'] for ele in eval_data[k]]
        answer_strs= [ele['answer'] for ele in eval_data[k]]
        option_strs = [ele['options'] for ele in eval_data[k]]

        outputs = llm.generate(prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        total_scores = 0
        for output, answer_idx, answer_str, option_str in zip(outputs, answer_idxs, answer_strs, option_strs):
            score = float(compute_score(output, answer_str, answer_idx, option_str))
            total_scores += score
        accuracy = total_scores / len(outputs)
        metrics[k] = accuracy

    for k in eval_data.keys():
        print("Accuracy for ", k, ": ", metrics[k])
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "a") as f:
        json.dump(metrics, f)
        f.write("\n")