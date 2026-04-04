from datasets import load_dataset

def get_medqa():
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    return ds

def get_medmcqa():
    ds = load_dataset("araag2/MedMCQA", "processed",split="test")
    return ds

def get_pubmedqa():
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    return ds

def get_gpqa():
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    return ds

def get_mmlu():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    return ds








