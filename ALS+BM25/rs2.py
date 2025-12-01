import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  

import math
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from tqdm import tqdm

import implicit
from implicit.nearest_neighbours import bm25_weight

# ============================================
# CONFIG
# ============================================
INPUT_PATH = "train.txt"                    
OUTPUT_PATH = "top20_recommendations_bm25_als_pop.txt"
TOP_K = 20

ALPHA = 30.0       # confidence scaling for implicit feedback (try 20/30/40)
FACTORS = 256      # latent dimensions (try 128/256)
REG = 0.02         # regularization (try 0.01/0.02/0.05)
ITERS = 40         # ALS iterations (try 30/40)
POP_LAMBDA = 0.15  # weight for popularity in final score (try 0.1–0.3)

# ============================================
# 1. LOAD DATA (user_id item1 item2 ...)
# ============================================
user_interactions = defaultdict(set)
all_users = set()
all_items = set()
item_pop_counts = defaultdict(int)

with open(INPUT_PATH, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) <= 1:
            continue

        u = int(parts[0])
        items = list(map(int, parts[1:]))

        all_users.add(u)
        for it in items:
            user_interactions[u].add(it)
            all_items.add(it)
            item_pop_counts[it] += 1

# Map raw IDs -> consecutive indices
user2idx = {u: i for i, u in enumerate(sorted(all_users))}
idx2user = {i: u for u, i in user2idx.items()}

item2idx = {it: i for i, it in enumerate(sorted(all_items))}
idx2item = {i: it for it, i in item2idx.items()}

num_users = len(user2idx)
num_items = len(item2idx)

print(f"Users: {num_users}  Items: {num_items}")

# Convert popularity to array aligned with item indices
item_pop = np.zeros(num_items, dtype=np.float32)
for it, cnt in item_pop_counts.items():
    item_pop[item2idx[it]] = cnt

# ============================================
# 2. BUILD USER–ITEM SPARSE MATRIX (users x items)
# ============================================
rows, cols, data = [], [], []

for u, items in user_interactions.items():
    u_idx = user2idx[u]
    for it in items:
        i_idx = item2idx[it]
        rows.append(u_idx)
        cols.append(i_idx)
        data.append(1.0)  # implicit positive interaction

X_ui = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

# ============================================
# 3. APPLY BM25 + IMPLICIT CONFIDENCE
# ============================================
# bm25_weight on user–item, then scale by ALPHA
X_bm25 = bm25_weight(X_ui).tocsr()
X_conf = (X_bm25 * ALPHA).astype("double").tocsr()

# ============================================
# 4. TRAIN ALS MODEL ON USER–ITEM MATRIX
# ============================================
model = implicit.als.AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REG,
    iterations=ITERS,
    random_state=42,
)

print(f"Training ALS (BM25 + implicit confidence) with "
      f"factors={FACTORS}, reg={REG}, iters={ITERS}, alpha={ALPHA}...")
model.fit(X_conf)
print("Training complete.")

# ============================================
# 5. GENERATE TOP-20 RECOMMENDATIONS PER USER
#    with popularity-blended re-ranking
# ============================================
print(f"Generating Top-{TOP_K} recommendations for each user...")

with open(OUTPUT_PATH, "w") as out:
    for u_idx in tqdm(range(num_users)):
        u_id = idx2user[u_idx]
        seen_items = user_interactions[u_id]
        seen_indices = {item2idx[it] for it in seen_items}

        # Ask for more than TOP_K so we can filter & re-rank
        N_raw = TOP_K + len(seen_indices) + 100

        # Get candidate ids & ALS scores (no internal filtering)
        ids, als_scores = model.recommend(
            userid=u_idx,
            user_items=None,
            N=N_raw,
            filter_already_liked_items=False,
            recalculate_user=False,
        )

        ids = np.asarray(ids)
        als_scores = np.asarray(als_scores)

        # Popularity scores for these candidate items
        cand_pop = item_pop[ids]
        pop_scores = np.log1p(cand_pop)  # log(1 + pop)

        # Blend ALS + popularity
        final_scores = als_scores + POP_LAMBDA * pop_scores

        # Build list of (item_idx, final_score), filter seen items
        candidates = []
        for item_idx, score in zip(ids, final_scores):
            if item_idx in seen_indices:
                continue
            candidates.append((item_idx, score))

        # Sort by blended score desc, pick top K
        candidates.sort(key=lambda x: x[1], reverse=True)
        topk_indices = [it for (it, _) in candidates[:TOP_K]]

        # Map back to original item IDs
        rec_items = [idx2item[i] for i in topk_indices]

        line = " ".join([str(u_id)] + [str(it) for it in rec_items])
        out.write(line + "\n")

print("Done. Recommendations saved to:", OUTPUT_PATH)
