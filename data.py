from datasets import load_dataset
from datasets import concatenate_datasets

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


def load_mix_dataset():
    medqa_dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", cache_dir="data/medqa")
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    medqa_dataset = medqa_dataset.map(
        lambda x: {
            "query": x["question"] +  "\n" + "\n".join(f"{key}: {value}" for key, value in x['options'].items()) + "\n" + instruction_following,
            "reward_model": {"answer": x['answer'], "answer_idx": x['answer_idx']}
        },
        remove_columns=medqa_dataset.column_names
    )

    medmcqa_dataset = load_dataset("araag2/MedMCQA", "processed",split="train", cache_dir="data/medmcqa")
    def format_choices(x):
        return "A: " + x['Option_A'] + "\nB: " + x['Option_B'] + "\nC: " + x['Option_C'] + "\nD: " + x['Option_D']

    medmcqa_dataset = medmcqa_dataset.map(
        lambda x: {
            "query": x["Question"] +  "\n" + format_choices(x) + "\n" + instruction_following,
            "reward_model": {"answer": x['Option_'+x['Label']], "answer_idx": x['Label']}
        },
        remove_columns=medmcqa_dataset.column_names
    )
    dataset = concatenate_datasets([medqa_dataset, medmcqa_dataset])
    pubmed_dataset =  load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train', cache_dir="data/pubmedqa")
    instruction_following = "Please reason step by step, and put your final answer ('Yes' or 'No') within \\boxed{}."
    pubmed_dataset = pubmed_dataset.map(
        lambda x: {
            "query": x["question"] +  "\n" + instruction_following,
            "reward_model": {"answer": x['final_decision'], "answer_idx": None}
        },
        remove_columns=pubmed_dataset.column_names
    )
    dataset = concatenate_datasets([dataset, pubmed_dataset])
    return dataset
