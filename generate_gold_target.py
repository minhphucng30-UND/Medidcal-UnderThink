from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import Dataset
import random
import numpy as np
from data import load_medical_reasoning_dataset
ds = load_medical_reasoning_dataset()

if __name__ == "__main__":
    llm = LLM("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", enforce_eager=False, max_num_seqs=1024)
    tokenizer = llm.get_tokenizer()
    stop_token_ids = tokenizer("</think>")["input_ids"]

    random.seed(42)
    prompts = list(ds['query'])
    random.shuffle(prompts)
    prompts = prompts[:1000]
    n_tokens = 4096
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
    )
    import os
    os.makedirs("data/gold_target", exist_ok=True)
    for n_tokens in [2048, 4096, 8192, 16384, 32768]:
        chat_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt + f"\nThink for {n_tokens} tokens."}], add_generation_prompt=True, tokenize=False) for prompt in prompts]
        outputs = llm.generate(chat_prompts, sampling_params=sampling, use_tqdm=True)
        outputs = [output.outputs[0].text for output in outputs]
        dataset = Dataset.from_dict({"query": prompts, "solution": outputs})
        dataset.save_to_disk(f"data/gold_target/Med-R1-0528-Qwen3-8B-1k-ntokens-{n_tokens}")