from datasets import load_dataset

def load_medical_reasoning_dataset():
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train", cache_dir="data/medical-o1")
    ds = ds.map(
        lambda x: {
            "query": x["Question"],
            "solution": "<think>\n" + x["Complex_CoT"] + "\n</think>\n" + x['Response']
        }
    )
    return ds


def load_medical_r1_dataset():
    ds = load_dataset("FreedomIntelligence/Medical-R1-Distill-Data", "en", split="train", cache_dir="data/medical-r1")
    ds = ds.map(
        lambda x: {
            "query": x["question"],
            "solution": "<think>\n" + x["reasoning (reasoning_content)"] + "\n</think>\n" + x['response (content)']
        }
    )
    return ds