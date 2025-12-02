import os
from typing import List, Tuple

import numpy as np
import cornac
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, NDCG, MAP
from cornac.models import BiVAECF
import pickle

def load_interactions_from_txt(path: str) -> List[Tuple[str, str, float]]:
    uir = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user = parts[0]  
            for item in parts[1:]:
                uir.append((user, item, 1.0))
    return uir


DATA_PATH = "train.txt"  
OUTPUT_PATH = "bivae_recommendations_top20.txt"
MODEL_PATH = "bivae_model.pkl"
TRAINSET_PATH = "train_set.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find {DATA_PATH}. Put train.txt in this folder.")

data = load_interactions_from_txt(DATA_PATH)
print(f"Loaded {len(data)} user-item interactions")

eval_method = RatioSplit(
    data=data,
    test_size=0.2,          
    rating_threshold=0.5,   
    exclude_unknowns=True,
    verbose=True,
    seed=42,
)


bivae = BiVAECF(
    k=50,                   
    encoder_structure=[100],
    act_fn="tanh",          
    likelihood="pois",      
    n_epochs=200,
    batch_size=128,
    learning_rate=0.001,
    beta_kl=1.0,            
    cap_priors={"user": False, "item": False},  
    use_gpu=True,           
    verbose=True,
    seed=42,
)


metrics = [
    Recall(k=10),
    Recall(k=20),
    NDCG(k=10),
    NDCG(k=20),
    MAP(),   
]


print("Starting BiVAE experiment...")
experiment = cornac.Experiment(
    eval_method=eval_method,
    models=[bivae],
    metrics=metrics,
    user_based=True,
)

experiment.run()   
print("BiVAE training & evaluation finished.")


train_set = eval_method.train_set

bivae.save(MODEL_PATH)
print(f"Saved trained BiVAE model to: {MODEL_PATH}")


with open(TRAINSET_PATH, "wb") as f_ts:
    pickle.dump(train_set, f_ts)
print(f"Saved train_set to: {TRAINSET_PATH}")


print(f"Generating top-20 recommendations for each user into: {OUTPUT_PATH}")

inv_uid_map = {v: k for k, v in train_set.uid_map.items()}
inv_iid_map = {v: k for k, v in train_set.iid_map.items()}

num_users = train_set.num_users
num_items = train_set.num_items
all_items = np.arange(num_items)

with open(OUTPUT_PATH, "w") as f_out:
    for u_idx in range(num_users):
        raw_user_id = inv_uid_map[u_idx]

        scores = bivae.score(u_idx) 

        user_interactions = set(train_set.matrix[u_idx].indices)

        candidates = [
            (i, scores[i]) for i in range(num_items)
            if i not in user_interactions
        ]

        candidates.sort(key=lambda x: x[1], reverse=True)

        top_k = 20
        top_items_internal = [i for i, _ in candidates[:top_k]]

        top_item_ids = [str(inv_iid_map[i]) for i in top_items_internal]

        line = " ".join([str(raw_user_id)] + top_item_ids)
        f_out.write(line + "\n")

print("Done. Recommendation file written:")
print(f"  {OUTPUT_PATH}")
