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

def budget_forcing_generate(prompt, max_budget_tokens, llm, n_trials: int = 1):
    stop_token_ids = tokenizer("</think>")["input_ids"]
    ignore_str = "\nWait"
    max_tokens_thinking_tmp = max_budget_tokens
    if max_tokens_thinking_tmp > 0:
        for i in range(n_trials):
            sampling_params = SamplingParams(max_tokens=max_tokens_thinking_tmp, temperature=0.0, min_tokens=1, stop_token_ids=stop_token_ids, skip_special_tokens=False, repetition_penalty=1.0)
            o = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
            max_tokens_thinking_tmp -= min([len(o[i].outputs[0].token_ids) for i in range(len(o))])
            if max_tokens_thinking_tmp > 0:
                prompt = prompt + o[0].outputs[0].text + ignore_str
            else:
                break
        prompt += o[0].outputs[0].text
    return prompt


def final_answer_generate(prompts, llm):
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0, min_tokens=0, skip_special_tokens=False)
    o = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    return [prompt + output.outputs[0].text for prompt, output in zip(prompts, o)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/gemma-3-4b-pt/m1-sft-epoch9")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_budget_tokens", type=int, default=512)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model_path = args.model_path
    temperature = args.temperature
    n_trials = args.n_trials
    seed = args.seed
    # eval_data = {k: eval_data[k] for k in ['GPQA_Medical_test', 'PubMedQA_test']}
    eval_data = load_huatuo_eval()
    eval_data = {k: eval_data[k] for k in ['GPQA_Medical_test']}
    metrics = {}
    metrics['model_path'] = "/".join(model_path.split("/")[1:])
    print(metrics['model_path'])
    n_gpus = torch.cuda.device_count()
    llm = LLM(model_path, enforce_eager=False, max_num_seqs=2048, max_model_len=32768, tensor_parallel_size=n_gpus, gpu_memory_utilization=0.8, max_num_batched_tokens=16384)
    tokenizer = llm.get_tokenizer()
    metrics['temperature'] = 0.0
    metrics['max_budget_tokens'] = args.max_budget_tokens
    metrics['n_trials'] = n_trials
    metrics['seed'] = seed
    MAX_TOKENS_THINKING = args.max_budget_tokens
    NUM_TRIALS = args.n_trials

    for k in eval_data.keys():
        chat_prompts = [ele['question'] for ele in eval_data[k]]
        answer_idxs = [ele['answer_idx'] for ele in eval_data[k]]
        answer_strs= [ele['answer'] for ele in eval_data[k]]
        option_strs = [ele['options'] for ele in eval_data[k]]
        
        instruction_following = 'Please reason step by step, and put your final answer within \\boxed{}.'
        progress_bar = tqdm(range(len(chat_prompts)))
        total_scores = 0.0
        total_count = 0
        if tokenizer.chat_template is not None:
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": chat_prompts[i] + "\n" + instruction_following}], tokenize=False, add_generation_prompt=True) + "\n<think>" for i in range(len(chat_prompts))]
        else:
            prompts = [chat_prompts[i] + "\n" + instruction_following + "\n<think>" for i in range(len(chat_prompts))]
        
        ignore_str = "\nWait"
        stop_token_ids = tokenizer("</think>")["input_ids"]
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS_THINKING, temperature=0.0, min_tokens=0, stop_token_ids=stop_token_ids, skip_special_tokens=False)
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        prompts = [prompt + output.outputs[0].text for prompt, output in zip(prompts, outputs)]
        budget_tokens = [MAX_TOKENS_THINKING - len(output.outputs[0].token_ids) for output in outputs]

        progress_bar = tqdm(range(len(prompts)))
        all_outputs = []
        for i in progress_bar:
            prompt = prompts[i]
            if budget_tokens[i] > 0:
                output = budget_forcing_generate(prompt + ignore_str, budget_tokens[i], llm, NUM_TRIALS)
            else:
                output = prompt
            all_outputs.append(output)

        answer_txt = "\nTime is up. Given the time I’ve spent and the approaches I’ve tried, I should stop thinking and formulate a final answer based on what I already have.\n</think>"
        all_outputs = [output + answer_txt for output in all_outputs]
        all_outputs = final_answer_generate(all_outputs, llm)
        all_scores = [float(compute_score(all_outputs[i], answer_strs[i], answer_idxs[i], option_strs[i])) for i in range(len(all_outputs))]
        print(f"Accuracy for {k}: {sum(all_scores) / len(all_scores)}")
        metrics[k] = sum(all_scores) / len(all_scores)

    for k in eval_data.keys():
        print("Accuracy for ", k, ": ", metrics[k])
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/results_budget_forcing.json", "a") as f:
        json.dump(metrics, f)
        f.write("\n")