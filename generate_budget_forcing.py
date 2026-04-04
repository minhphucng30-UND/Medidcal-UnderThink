from vllm import LLM, SamplingParams
import os
from tqdm import tqdm
from datasets import Dataset
import random
import numpy as np
from data import load_mix_dataset

if __name__ == "__main__":
    ds = load_mix_dataset()
    llm = LLM("models/DeepSeek-R1-0528-Qwen3-8B", enforce_eager=False, max_num_seqs=1024)
    tokenizer = llm.get_tokenizer()
    stop_token_ids = tokenizer("</think>")["input_ids"]

    random.seed(42)
    prompts = list(ds['query'])
    random.shuffle(prompts)
    prompts = prompts[:1000]
    chat_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) for prompt in prompts]
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        min_tokens=0,
    )

    MAX_TOKENS_THINKING = 32768
    NUM_TRIALS = 10
    outputs = llm.generate(chat_prompts, sampling_params=sampling, use_tqdm=True)
    ignore_str = "\nWait"
    max_tokens_thinking_tmp = MAX_TOKENS_THINKING

    for i in tqdm(range(NUM_TRIALS)):
        max_tokens_thinking_tmp -= min([len(outputs[i].outputs[0].token_ids) for i in range(len(outputs))])
        if max_tokens_thinking_tmp > 0:
            chat_prompts = [chat_prompts[i] + outputs[i].outputs[0].text + ignore_str for i in range(len(chat_prompts))]
            sampling_params = SamplingParams(
                max_tokens=max_tokens_thinking_tmp,
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.6,
                top_p=0.95,
            )
            outputs = llm.generate(
                chat_prompts,
                sampling_params=sampling_params
            )

    chat_prompts = [chat_prompts[i] + outputs[i].outputs[0].text for i in range(len(chat_prompts))]
    sampling_params = SamplingParams(
        max_tokens=32768,
        min_tokens=0,
        skip_special_tokens=False,
        temperature=0.6,
        top_p=0.95,
    )
    outputs = llm.generate(
        chat_prompts,
        sampling_params=sampling_params,
    )
    chat_prompts = [chat_prompts[i] + outputs[i].outputs[0].text for i in range(len(chat_prompts))]
    dataset = Dataset.from_dict({"query": prompts, "solution": chat_prompts})
    os.makedirs("data/budget_forcing", exist_ok=True)
    dataset.save_to_disk(f"data/budget_forcing/Med-R1-0528-Qwen3-8B-1k-ntrial-{NUM_TRIALS}")