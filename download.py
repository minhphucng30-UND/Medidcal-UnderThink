from huggingface_hub import snapshot_download

if __name__ == "__main__":
    snapshot_download(
        repo_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        local_dir="models/DeepSeek-R1-0528-Qwen3-8B",
    )
