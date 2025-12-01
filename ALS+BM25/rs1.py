import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  

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
OUTPUT_PATH = "top20_recommendations_bm25_als.txt"
TOP_K = 20
ALPHA = 40.0  # confidence scaling for implicit feedback

# ============================================
# 1. LOAD DATA (user_id item1 item2 ...)
# ============================================
user_interactions = defaultdict(set)
all_users = set()
all_items = set()

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

# Map raw IDs -> consecutive indices
user2idx = {u: i for i, u in enumerate(sorted(all_users))}
idx2user = {i: u for u, i in user2idx.items()}

item2idx = {it: i for i, it in enumerate(sorted(all_items))}
idx2item = {i: it for it, i in item2idx.items()}

num_users = len(user2idx)
num_items = len(item2idx)

print(f"Users: {num_users}  Items: {num_items}")

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
#    (still on user–item; that's fine)
# ============================================
# bm25_weight returns a COO; convert back to CSR
X_bm25 = bm25_weight(X_ui).tocsr()

# Scale by alpha (confidence)
X_conf = (X_bm25 * ALPHA).astype("double").tocsr()

# ============================================
# 4. TRAIN ALS MODEL ON USER–ITEM MATRIX
# ============================================
model = implicit.als.AlternatingLeastSquares(
    factors=128,        # try 64 / 128 / 256
    regularization=0.05,
    iterations=30,
    random_state=42,
)

print("Training ALS (BM25 + implicit confidence) on user–item matrix...")
# For ALS in recent implicit versions, .fit expects user–item CSR
model.fit(X_conf)
print("Training complete.")

# ============================================
# 5. GENERATE TOP-20 RECOMMENDATIONS PER USER
#    (MANUAL FILTERING OF SEEN ITEMS)
# ============================================

print(f"Generating Top-{TOP_K} recommendations for each user...")

with open(OUTPUT_PATH, "w") as out:
    for u_idx in tqdm(range(num_users)):
        u_id = idx2user[u_idx]
        seen_items = user_interactions[u_id]
        seen_indices = {item2idx[it] for it in seen_items}

        # Ask for a bit more than TOP_K so we can drop seen items safely
        N_raw = TOP_K + len(seen_indices) + 50

        # IMPORTANT: we do NOT pass user_items here; we use stored user_factors
        ids, scores = model.recommend(
            userid=u_idx,
            user_items=None,
            N=N_raw,
            filter_already_liked_items=False,  # we'll filter ourselves
            recalculate_user=False,           # use learned user_factors
        )

        # Manually filter out already-seen items
        filtered = [i for i in ids if i not in seen_indices]

        # Take top-K after filtering
        topk_indices = filtered[:TOP_K]

        # Map internal indices back to original item IDs
        rec_items = [idx2item[i] for i in topk_indices]

        # In case something weird happens and we have < TOP_K, pad nothing extra
        line = " ".join([str(u_id)] + [str(it) for it in rec_items])
        out.write(line + "\n")

print("Done. Recommendations saved to:", OUTPUT_PATH)
