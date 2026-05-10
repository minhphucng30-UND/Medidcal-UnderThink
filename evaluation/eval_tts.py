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
    parser.add_argument("--benchmark", type=str, default="GPQA_Medical_test")
    parser.add_argument("--is_reasoning", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_budget_tokens", type=int, default=512)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model_path = args.model_path
    temperature = args.temperature
    n_trials = args.n_trials
    seed = args.seed
    eval_data = load_dataset("zou-lab/BioMed-R1-Eval", args.benchmark, split="test")
    if args.is_reasoning:
        eval_data = eval_data.filter(lambda x: x['is_reasoning'])
    else:
        eval_data = eval_data.filter(lambda x: not x['is_reasoning'])
    
    print("Number of data: ", len(eval_data))
    metrics = {}
    metrics['model_path'] = "/".join(model_path.split("/")[1:])
    print(metrics['model_path'])
    n_gpus = torch.cuda.device_count()
    llm = LLM(model_path, enforce_eager=False, max_num_seqs=2048, max_model_len=32768, tensor_parallel_size=n_gpus, gpu_memory_utilization=0.8, max_num_batched_tokens=16384)
    tokenizer = llm.get_tokenizer()
    metrics['max_budget_tokens'] = args.max_budget_tokens
    metrics['n_trials'] = n_trials
    metrics['benchmark'] = args.benchmark
    metrics['is_reasoning'] = args.is_reasoning
    MAX_TOKENS_THINKING = args.max_budget_tokens
    NUM_TRIALS = args.n_trials

    chat_prompts = [ele['question'] for ele in eval_data]
    answer_idxs = [ele['answer_idx'] for ele in eval_data]
    answer_strs= [ele['answer'] for ele in eval_data]
    option_strs = [json.loads(ele['options']) for ele in eval_data]
    
    instruction_following = 'Please reason step by step, and put your final answer within \\boxed{}.'
    total_scores = 0.0
    total_count = 0
    if tokenizer.chat_template is not None:
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": chat_prompts[i] + "\n" + instruction_following}], tokenize=False, add_generation_prompt=True) + "\n<think>" for i in range(len(chat_prompts))]
    else:
        prompts = [chat_prompts[i] + "\n" + instruction_following + "\n<think>" for i in range(len(chat_prompts))]
    
    all_outputs = []

    for prompt in tqdm(prompts):
        stop_token_ids = tokenizer("</think>")["input_ids"]
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS_THINKING, temperature=0.0, min_tokens=0, stop_token_ids=stop_token_ids, skip_special_tokens=False)
        o = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
        ignore_str = "\nWait"
        max_tokens_thinking_tmp = MAX_TOKENS_THINKING
        for i in range(NUM_TRIALS):
            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            if max_tokens_thinking_tmp > 0:
                prompt += o[0].outputs[0].text + ignore_str
                sampling_params = SamplingParams(max_tokens=max_tokens_thinking_tmp, temperature=0.0, min_tokens=1, stop_token_ids=stop_token_ids, skip_special_tokens=False)
                o = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)

        prompt += o[0].outputs[0].text + "\n</think>"
        sampling_params = SamplingParams(max_tokens=1024, temperature=0.0, min_tokens=0, skip_special_tokens=False)
        o = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
        prompt += o[0].outputs[0].text
        all_outputs.append(prompt)
    
    all_scores = [float(compute_score(all_outputs[i], answer_strs[i], answer_idxs[i], option_strs[i])) for i in range(len(all_outputs))]
    metrics[args.benchmark] = sum(all_scores) / len(all_scores)
    print("Accuracy for ", args.benchmark, ": ", metrics[args.benchmark])
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/results_tts.json", "a") as f:
        json.dump(metrics, f)
        f.write("\n")