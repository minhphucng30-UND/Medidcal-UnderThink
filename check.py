from datasets import load_from_disk
from evaluation.qwen_math_eval_toolkit.parser import extract_answer, choice_answer_clean
from data import load_mix_dataset
import numpy as np
import pickle

reward_model = pickle.load(open('data/budget_forcing/Med-R1-0528-Qwen3-8B-1k-ntrial-0/reward_model.pkl', 'rb'))
all_accs = []
for n_trial in [0, 3, 5, 10]:
    ds = load_from_disk(f"data/budget_forcing/Med-R1-0528-Qwen3-8B-1k-ntrial-{n_trial}")
    solution = ds['solution']
    lens = [len(solution[i]) for i in range(len(solution))]
    accs = []
    for i in range(len(solution)):
        answer_idx = reward_model[i]['answer_idx']
        if answer_idx is not None:
            pred= extract_answer(solution[i], 'multiple_choice')
            score = 1 if pred == answer_idx else 0
            accs.append(score)
        else:
            pred= extract_answer(solution[i], 'med')
            answer = reward_model[i]['answer']
            score = 1 if pred.strip().lower() == answer.strip().lower() else 0
            accs.append(score)
        
    print(f"Acc: {sum(accs)/len(accs)}, Mean length: {np.mean(lens)}")
    all_accs.append(sum(accs)/len(accs))
import matplotlib.pyplot as plt
plt.plot([0, 3, 5, 10], all_accs, marker='o', linestyle='-', color='b')
plt.xlabel('Number of trials')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of trials')
plt.savefig('acc_vs_ntrial.pdf')