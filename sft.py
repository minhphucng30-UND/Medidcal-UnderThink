from data import load_medical_reasoning_dataset, load_medical_r1_dataset
import time
import wandb
from collections import deque
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import argparse
import os
torch.backends.cuda.matmul.allow_tf32 = True


def _collate(tokenizer, batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        padding_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_len)
        attention_mask.append(item["attention_mask"] + [0] * padding_len)
        labels.append(item["labels"] + [-100] * padding_len)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }

def process_sample(tokenizer, sample):
    input_ids = []
    labels = []
    query_messages = [
        {"role": "user", "content": sample["query"], "type": "text"},
    ]
    query_text = tokenizer.apply_chat_template(
        query_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    response_text = sample["solution"] + tokenizer.eos_token
    prompt_enc = tokenizer(
        query_text,
        truncation=True,
        add_special_tokens=False,
        max_length=1024,
    )["input_ids"]
    input_ids.extend(prompt_enc)
    labels.extend([-100] * len(prompt_enc))

    response = tokenizer(
        response_text,
        truncation=True,
        add_special_tokens=False,
        max_length=16384,
    )["input_ids"]
    input_ids.extend(response)
    labels.extend(response)
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["medical-reasoning", "medical-r1"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.dataset_name == "medical-reasoning":
        dataset = load_medical_reasoning_dataset()
    elif args.dataset_name == "medical-r1":
        dataset = load_medical_r1_dataset()
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    
    wandb.init(project="medical-reasoning-sft", name=f"{args.dataset_name}-sft", config=vars(args))

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
    }
    model = AutoModelForCausalLM.from_pretrained("models/Qwen3-1.7B", **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-1.7B")
    dataset = dataset.map(lambda x: process_sample(tokenizer, x), batched=False, num_proc=10, remove_columns=dataset.column_names)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: _collate(tokenizer, x), num_workers=8, pin_memory=True)
    total_steps = len(dataloader) * args.epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    progress_bar = tqdm(dataloader, smoothing=0.01)
    rolling_loss = deque(maxlen=100)

    for epoch in range(args.epochs):
        metrics = None
        for step_i, batch in enumerate(progress_bar):
            start_time = time.time()
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
                loss.backward()
            l2_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                _loss = loss.item()
                rolling_loss.append(_loss)
                _rolling_loss = sum(rolling_loss) / len(rolling_loss)
                progress_bar.set_description(
                    f"epoch: {epoch} step: {step_i} loss: {_rolling_loss:.4f} lr: {scheduler.get_last_lr()[0]:.2e}"
                )
                wandb.log({
                    "train/loss": _loss,
                    "epoch": epoch,
                    "train/rolling_loss": _rolling_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_l2_norm": l2_norm.detach().cpu().item(),
                    "train/time": time.time() - start_time,
                })
        model.save_pretrained(f"models/Qwen3-1.7B/{args.dataset_name}-sft-epoch{epoch}")
        tokenizer.save_pretrained(f"models/Qwen3-1.7B/{args.dataset_name}-sft-epoch{epoch}")