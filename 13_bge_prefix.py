from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import faiss

RESULT_DIR = Path('result/14_bge_prefix_allfields')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder : {RESULT_DIR}/ -- ready')

print('[Load] Loading production results...')
results_df = pd.read_excel('dataset/production_results.xlsx')
all_companies = results_df.drop_duplicates(subset='domain').reset_index(drop=True)
print(f'[Load] Unique companies : {len(all_companies):,}')

with open('dataset/goi_search_results.json', 'r') as f:
    queries_data = json.load(f)
print(f'[Load] Queries : {len(queries_data)}')

ORIGINAL_RESULT_DIR = Path('result/02_baseline_bge')

print('[FAISS] Loading cached company embeddings (no re-encoding needed)...')
embeddings = np.load(ORIGINAL_RESULT_DIR / 'company_embeddings.npy').astype('float32')
print(f'[FAISS] Embeddings shape : {embeddings.shape}')

print('[FAISS] Loading cached IndexFlatIP...')
index = faiss.read_index(str(ORIGINAL_RESULT_DIR / 'company_faiss.index'))
print(f'[FAISS] Vectors in index : {index.ntotal:,}')
assert index.ntotal == len(all_companies), 'Corpus/index size mismatch'

print("[Note] Text representation used to build these embeddings: all fields -- name, country, state, city, district, organization_type, organization_size, nace_code, summary, summary_keywords (matches 02_baseline_bge.ipynb's corpus)")

import torch

print(f'[GPU] CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[GPU] Device         : {torch.cuda.get_device_name(0)}')
    DEVICE = 'cuda'
else:
    print('[GPU] No GPU -- query encoding will be slower but still fine (only 101 queries)')
    DEVICE = 'cpu'


from sentence_transformers import SentenceTransformer

BGE_QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '

print('[Encode] Loading BGE-large model...')
t0    = time.time()
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=DEVICE)
print(f'[Encode] Model loaded in {time.time()-t0:.1f}s on {model.device}')

print(f'[Run] Starting BGE retrieval for {len(queries_data)} queries (WITH instruction prefix)...')
print('-' * 55)

all_bge_results = []
query_times     = []
total_start     = time.time()

for i, item in enumerate(queries_data):
    query_id = item['query_id']
    query    = item['query']
    query_with_instruction = BGE_QUERY_INSTRUCTION + query

    t0           = time.perf_counter()
    query_emb    = model.encode(
        [query_with_instruction],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype('float32')
    scores, idxs = index.search(query_emb, 1000)
    query_ms     = (time.perf_counter() - t0) * 1000
    query_times.append(query_ms)

    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0])):
        company = all_companies.iloc[idx]
        all_bge_results.append({
            'query_id': query_id,
            'query':    query,
            'rank':     rank + 1,
            'score':    float(score),
            'domain':   company['domain'],
            'name':     company.get('name', ''),
            'country':  company.get('country', ''),
        })

    if (i + 1) % 20 == 0 or (i + 1) == len(queries_data):
        elapsed   = time.time() - total_start
        remaining = (len(queries_data) - i - 1) * elapsed / (i + 1)
        print(f'[Run] {i+1:3d}/{len(queries_data)}  |  '
              f'avg {sum(query_times)/len(query_times):.1f}ms/query  |  '
              f'~{remaining:.0f}s remaining')

bge_df = pd.DataFrame(all_bge_results)
bge_df.to_csv(RESULT_DIR / 'bge_results_prefix.csv', index=False)

AVG_LATENCY_MS = sum(query_times) / len(query_times)
print('-' * 55)
print('[Run] Done!')
print(f'[Run] Avg query latency : {AVG_LATENCY_MS:.1f}ms')
print(f'[Run] Saved to          : {RESULT_DIR}/bge_results_prefix.csv')

print('[Eval] Loading production labels...')
production_df = pd.read_excel('dataset/production_results.xlsx')

K_VALUES = [10, 50, 100, 500, 1000]

def get_relevant(query_id, top_k=1000):
    return set(production_df[
        (production_df['query_id'] == query_id) &
        (production_df['rank'] <= top_k)
    ]['domain'].tolist())

def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / k if k else 0

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0

def f1_at_k(retrieved, relevant, k):
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def dcg_at_k(retrieved, relevant, k):
    return sum(
        1 / np.log2(i + 2)
        for i, d in enumerate(retrieved[:k]) if d in relevant
    )

def ndcg_at_k(retrieved, relevant, k):
    ideal = dcg_at_k(list(relevant), relevant, k)
    return dcg_at_k(retrieved, relevant, k) / ideal if ideal else 0

print('[Eval] Computing metrics for all queries...')
eval_rows = []

for i, item in enumerate(queries_data):
    qid       = item['query_id']
    query     = item['query']
    relevant  = get_relevant(qid)
    retrieved = (
        bge_df[bge_df['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )
    for k in K_VALUES:
        eval_rows.append({
            'query_id':  qid,
            'query':     query,
            'k':         k,
            'precision': precision_at_k(retrieved, relevant, k),
            'recall':    recall_at_k(retrieved, relevant, k),
            'f1':        f1_at_k(retrieved, relevant, k),
            'ndcg':      ndcg_at_k(retrieved, relevant, k),
        })

    if (i + 1) % 25 == 0:
        print(f'[Eval] {i+1}/101 queries evaluated...')

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(RESULT_DIR / 'evaluation_bge_prefix.csv', index=False)
print(f'[Eval] Saved to {RESULT_DIR}/evaluation_bge_prefix.csv')

print('[Compare] Loading original (no-prefix) BGE evaluation for comparison...')
original_eval_path = Path('result/02_baseline_bge/evaluation_bge.csv')
if original_eval_path.exists():
    original_eval_df = pd.read_csv(original_eval_path)
    print()
    print(f'  {"k":<6} {"Recall (no prefix)":>20} {"Recall (with prefix)":>22} {"Delta":>8}')
    print('  ' + '-' * 62)
    for k in K_VALUES:
        orig_recall = original_eval_df[original_eval_df['k'] == k]['recall'].mean()
        new_recall  = eval_df[eval_df['k'] == k]['recall'].mean()
        delta       = new_recall - orig_recall
        print(f'  {k:<6} {orig_recall:>20.3f} {new_recall:>22.3f} {delta:>+8.3f}')
else:
    print(f'[Compare] Original evaluation not found at {original_eval_path} -- skipping comparison')

print('[Summary] ============================================================')
print('[Summary] BGE-large RESULTS (All-Fields, WITH query instruction prefix)')
print('\n[Summary] Query encoding note : "Represent this sentence for searching relevant passages: " prefix added')
print(f'[Summary] Avg query latency  : {AVG_LATENCY_MS:.1f}ms')
print(f'[Summary] Companies (cached) : {len(all_companies):,}')
print('[Summary] Embedding dims     : 1024')
print(f'[Summary] Queries evaluated  : {len(queries_data)}')
print()
print(f'  {"k":<6} {"NDCG":>8} {"Precision":>10} {"Recall":>8} {"F1":>8}')
print('  ' + '-' * 46)
for k in K_VALUES:
    sub = eval_df[eval_df['k'] == k]
    print(f'  {k:<6} '
          f'{sub["ndcg"].mean():>8.3f} '
          f'{sub["precision"].mean():>10.3f} '
          f'{sub["recall"].mean():>8.3f} '
          f'{sub["f1"].mean():>8.3f}')
