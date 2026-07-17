#!/usr/bin/env python
# coding: utf-8

# # ColBERTv2 Baseline Retrieval (own brute-force MaxSim implementation)
# ## Late-interaction dense retrieval over 98,716 companies, no PLAID indexing needed
# 
# **Why this route instead of the official `colbert-ai`/PLAID engine:** `colbert-ir/colbertv2.0`'s config declares a custom `HF_ColBERT` architecture class, which needs `trust_remote_code=True` and the `colbert-ai` package -- the same class of custom-modeling-code failure that already broke `Alibaba-NLP/gte-large-en-v1.5` and `nvidia/NV-Embed-v2` in this repo against `transformers==5.9.0`. Inspecting the checkpoint's raw safetensors header shows it is actually just a standard BERT-base (`bert.embeddings.*`, `bert.encoder.layer.*`) plus one extra `linear.weight [128, 768]` token-projection layer -- no custom code needed at all. We load it with plain `BertModel.from_pretrained` and apply the projection ourselves.
# 
# **Why brute-force MaxSim instead of PLAID:** PLAID exists to make ColBERT's late-interaction scoring cheap at tens-of-millions-of-documents scale. Notebook 11's own ANN scaling results already showed exact/brute-force search is completely fine at our corpus size (98,716) -- IVF only started mattering as a latency optimization, never as a necessity. So we precompute per-token embeddings for the whole corpus once, then score every query against the full corpus via a batched MaxSim matmul -- no compressed index, no PLAID dependency, same underlying similarity function ColBERT was trained for.
# 
# **Known simplification vs. the official implementation:** we don't do ColBERT's query augmentation (padding queries with `[MASK]` tokens to a fixed length) or the standard practice of also masking out punctuation tokens -- we keep it to the `attention_mask` as returned by the tokenizer (real tokens incl. `[CLS]`/`[SEP]`, excluding padding). Worth flagging as a caveat if this baseline is written up.
# 
# **MaxSim scoring:** for query token embeddings Q (n_q x 128) and a document's token embeddings D (n_d x 128), `score(Q,D) = sum over q in Q of max over d in D of (q . d)`. Both are L2-normalized so dot products are cosine similarities.
# 
# **Checkpoint/resume:** both the corpus encoding step and the per-query MaxSim retrieval loop save progress periodically and self-exit gracefully before the 30-min SLURM wall time, so a timed-out run can simply be resubmitted (`sbatch run14.sh` again) to continue exactly where it left off -- no lost work, no restart from zero.

# In[ ]:


from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULT_DIR = Path('result/27_baseline_colbert')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder : {RESULT_DIR}/ -- ready')

MAX_DOC_LEN  = 192   # our rich text averages ~160 tokens per earlier baselines' notes
MAX_QUERY_LEN = 32   # standard ColBERT query length
DOC_CHUNK_SIZE = 10000   # companies scored per MaxSim matmul chunk
ENCODE_CHUNK_SIZE = 5000  # companies per encoding checkpoint interval


# In[ ]:


import sys, os, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_START = time.time()  # marks total job elapsed time, used to stop safely before the 30-min SLURM wall time
TIME_BUDGET_MINUTES = 26    # job wall-time is 30 min -- leaves a buffer before the hard kill so a checkpoint always gets saved

def robust_save(save_fn, path, retries=3, delay_seconds=5):
    """Writes to a temp file then atomically renames -- avoids leaving a corrupted checkpoint if the write is interrupted (seen: Lustre iostream errors truncating torch.save mid-write). Retries transient I/O failures. Keeps the original suffix on the temp name since np.save auto-appends .npy to any path that doesn't already end with it."""
    p = Path(path)
    tmp_path = str(p.with_suffix('.tmp' + p.suffix))
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            save_fn(tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            last_err = e
            print(f'[Checkpoint] Save attempt {attempt}/{retries} to {path} failed: {e} -- retrying in {delay_seconds}s...')
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            time.sleep(delay_seconds)
    raise last_err

print('[Imports] All packages loaded successfully')


# ## 1. Load dataset & build all-fields rich text (same convention as every other baseline)

# In[ ]:


print('[Load] Loading production results...')
results_df = pd.read_excel('dataset/production_results.xlsx')
all_companies = results_df.drop_duplicates(subset='domain').reset_index(drop=True)
print(f'[Load] Unique companies : {len(all_companies):,}')

with open('dataset/goi_search_results.json', 'r') as f:
    queries_data = json.load(f)
print(f'[Load] Queries : {len(queries_data)}')

def build_rich_text(row):
    parts = []
    for field, prefix in [
        ('name',              'Company:'),
        ('country',           'Country:'),
        ('state',             'State:'),
        ('municipality',      'City:'),
        ('district',          'District:'),
        ('organization_type', 'Type:'),
        ('organization_size', 'Size:'),
        ('nace_code',         'Industry:'),
        ('summary',           ''),
    ]:
        val = row.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(f'{prefix} {val}'.strip() if prefix else val)
    kw = row.get('summary_keywords', '')
    if isinstance(kw, str) and kw.strip():
        kw_clean = kw.replace("'", '').replace('[', '').replace(']', '')
        parts.append(f'Keywords: {kw_clean}')
    return ' | '.join(parts)

print('[Load] Building all-fields rich text for each company...')
rich_texts = [build_rich_text(row) for _, row in all_companies.iterrows()]
print('[Load] Sample rich text (first company):')
print(f'  {rich_texts[0][:300]}...')


# ## 2. GPU check

# In[ ]:


print(f'[GPU] CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[GPU] Device         : {torch.cuda.get_device_name(0)}')
    print(f'[GPU] VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    DEVICE = 'cuda'
else:
    print('[GPU] WARNING: No GPU -- this baseline needs a GPU to be practical')
    DEVICE = 'cpu'


# ## 3. Load ColBERTv2 as a plain BertModel + manual linear projection
# 
# Bypasses the custom `HF_ColBERT` architecture class entirely -- loads the standard `bert.*` weights via `BertModel.from_pretrained`, then separately reads the `linear.weight [128, 768]` token-projection tensor straight out of the checkpoint's safetensors file.

# In[ ]:


from transformers import AutoTokenizer, BertModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

REPO = 'colbert-ir/colbertv2.0'
print('[Load] Loading ColBERTv2 tokenizer + BertModel backbone...')
t0        = time.time()
tokenizer = AutoTokenizer.from_pretrained(REPO)
model     = BertModel.from_pretrained(REPO, add_pooling_layer=False).to(DEVICE)
model.eval()

print('[Load] Loading token-projection linear layer from checkpoint...')
ckpt_path       = hf_hub_download(REPO, 'model.safetensors')
state_dict      = load_safetensors(ckpt_path)
linear_weight   = state_dict['linear.weight'].to(DEVICE)  # [128, 768], no bias
print(f'[Load] Done in {time.time()-t0:.1f}s on {DEVICE} -- linear projection shape {tuple(linear_weight.shape)}')

@torch.no_grad()
def encode_tokens(texts, max_length, batch_size=64):
    """Returns (embeddings [N, max_length, 128] fp16, mask [N, max_length] bool), both on CPU."""
    all_embs, all_masks = [], []
    for i in range(0, len(texts), batch_size):
        batch   = texts[i:i+batch_size]
        encoded = tokenizer(batch, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)
        outputs = model(**encoded)
        proj    = F.linear(outputs.last_hidden_state, linear_weight)   # [B, L, 128]
        proj    = F.normalize(proj, p=2, dim=-1)
        all_embs.append(proj.half().cpu())
        all_masks.append(encoded['attention_mask'].bool().cpu())
    return torch.cat(all_embs, dim=0), torch.cat(all_masks, dim=0)


# ## 4. Encode all companies (token-level embeddings, kept in memory for brute-force MaxSim)
# 
# Checkpointed every `ENCODE_CHUNK_SIZE` companies -- if the job hits its time budget mid-encode, it saves progress and exits cleanly; resubmitting the same job resumes from the last completed chunk instead of starting over.

# In[ ]:


CHECKPOINT_EMBS_PATH  = RESULT_DIR / 'doc_embs_checkpoint.pt'
CHECKPOINT_MASKS_PATH = RESULT_DIR / 'doc_masks_checkpoint.pt'
FINAL_EMBS_PATH        = RESULT_DIR / 'doc_embs.pt'
FINAL_MASKS_PATH       = RESULT_DIR / 'doc_masks.pt'

print(f'[Encode] Encoding {len(rich_texts):,} companies (max_length={MAX_DOC_LEN})...')
t0 = time.time()

if FINAL_EMBS_PATH.exists():
    print('[Encode] Final embeddings already on disk -- loading, skipping corpus encoding')
    doc_embs  = torch.load(FINAL_EMBS_PATH)
    doc_masks = torch.load(FINAL_MASKS_PATH)
else:
    if CHECKPOINT_EMBS_PATH.exists():
        doc_embs  = torch.load(CHECKPOINT_EMBS_PATH)
        doc_masks = torch.load(CHECKPOINT_MASKS_PATH)
        start_idx = doc_embs.shape[0]
        pct = start_idx / len(rich_texts) * 100
        print(f'[Encode] Resuming from checkpoint -- {start_idx:,}/{len(rich_texts):,} ({pct:.1f}%) companies already encoded')
    else:
        doc_embs, doc_masks = None, None
        start_idx = 0

    for chunk_start in range(start_idx, len(rich_texts), ENCODE_CHUNK_SIZE):
        chunk_end   = min(chunk_start + ENCODE_CHUNK_SIZE, len(rich_texts))
        chunk_texts = rich_texts[chunk_start:chunk_end]
        chunk_embs, chunk_masks = encode_tokens(chunk_texts, max_length=MAX_DOC_LEN, batch_size=64)
        doc_embs  = chunk_embs if doc_embs is None else torch.cat([doc_embs, chunk_embs], dim=0)
        doc_masks = chunk_masks if doc_masks is None else torch.cat([doc_masks, chunk_masks], dim=0)
        done = doc_embs.shape[0]
        pct  = done / len(rich_texts) * 100
        robust_save(lambda p: torch.save(doc_embs, p), CHECKPOINT_EMBS_PATH)
        robust_save(lambda p: torch.save(doc_masks, p), CHECKPOINT_MASKS_PATH)
        print(f'[Encode]   {done:,}/{len(rich_texts):,} companies encoded ({pct:.1f}%)... checkpoint saved')
        if (time.time() - SCRIPT_START) / 60 > TIME_BUDGET_MINUTES:
            print(f'[Encode] Time budget ({TIME_BUDGET_MINUTES} min) reached at {done:,}/{len(rich_texts):,} ({pct:.1f}%) -- checkpoint saved. Rerun this same job to resume.')
            sys.exit(0)

    robust_save(lambda p: torch.save(doc_embs, p), FINAL_EMBS_PATH)
    robust_save(lambda p: torch.save(doc_masks, p), FINAL_MASKS_PATH)
    if CHECKPOINT_EMBS_PATH.exists():
        CHECKPOINT_EMBS_PATH.unlink()
    if CHECKPOINT_MASKS_PATH.exists():
        CHECKPOINT_MASKS_PATH.unlink()

ENCODE_TIME = time.time() - t0
print(f'[Encode] Done in {ENCODE_TIME/60:.1f} minutes (this session)')
print(f'[Encode] doc_embs shape : {tuple(doc_embs.shape)}  ({doc_embs.element_size()*doc_embs.nelement()/1e9:.2f} GB)')

doc_embs  = doc_embs.to(DEVICE)
doc_masks = doc_masks.to(DEVICE)
print(f'[Encode] Moved to {DEVICE} for scoring')


# ## 5. MaxSim retrieval -- brute-force over the full corpus, chunked for memory

# In[ ]:


@torch.no_grad()
def maxsim_search(query_text, top_k=1000):
    q_emb, q_mask = encode_tokens([query_text], max_length=MAX_QUERY_LEN, batch_size=1)
    q_emb  = q_emb.to(DEVICE).squeeze(0)[q_mask.squeeze(0)]   # [n_q, 128], real tokens only

    all_scores = []
    for start in range(0, doc_embs.shape[0], DOC_CHUNK_SIZE):
        end       = min(start + DOC_CHUNK_SIZE, doc_embs.shape[0])
        d_chunk   = doc_embs[start:end].float()          # [N, L, 128]
        mask_chunk = doc_masks[start:end]                 # [N, L]
        sim = torch.einsum('qd,nld->qnl', q_emb.float(), d_chunk)   # [n_q, N, L]
        sim = sim.masked_fill(~mask_chunk.unsqueeze(0), float('-inf'))
        max_sim = sim.max(dim=-1).values      # [n_q, N]
        chunk_scores = max_sim.sum(dim=0)     # [N] -- ColBERT score per document
        all_scores.append(chunk_scores.cpu())

    scores = torch.cat(all_scores)
    top_scores, top_idxs = torch.topk(scores, k=min(top_k, len(scores)))
    return top_idxs.numpy(), top_scores.numpy()

print('[Search] MaxSim search function ready')


# ## 6. Run all 101 queries
# 
# Checkpointed every 10 queries -- if the time budget runs out mid-loop, already-scored queries are skipped on resume (matched by `query_id`), so no query gets re-scored unnecessarily.

# In[ ]:


CHECKPOINT_RESULTS_PATH = RESULT_DIR / 'colbert_results_checkpoint.csv'
FINAL_RESULTS_PATH      = RESULT_DIR / 'colbert_results.csv'

print(f'[Run] Starting ColBERT MaxSim retrieval for {len(queries_data)} queries...')

if FINAL_RESULTS_PATH.exists():
    print('[Run] Final results already on disk -- loading, skipping retrieval')
    results_df_out = pd.read_csv(FINAL_RESULTS_PATH)
    query_times = []
else:
    if CHECKPOINT_RESULTS_PATH.exists():
        prior_df  = pd.read_csv(CHECKPOINT_RESULTS_PATH)
        done_qids = set(prior_df['query_id'].unique())
        all_results = prior_df.to_dict('records')
        print(f'[Run] Resuming -- {len(done_qids)}/{len(queries_data)} queries already scored')
    else:
        all_results = []
        done_qids   = set()

    query_times = []
    total_start = time.time()

    for i, item in enumerate(queries_data):
        query_id = item['query_id']
        if query_id in done_qids:
            continue
        query = item['query']

        t0 = time.perf_counter()
        idxs, scores = maxsim_search(query, top_k=1000)
        query_ms = (time.perf_counter() - t0) * 1000
        query_times.append(query_ms)

        for rank, (idx, score) in enumerate(zip(idxs, scores)):
            company = all_companies.iloc[int(idx)]
            all_results.append({
                'query_id': query_id, 'query': query, 'rank': rank + 1,
                'score': float(score), 'domain': company['domain'],
                'name': company.get('name', ''), 'country': company.get('country', ''),
            })

        if len(query_times) % 10 == 0 or (i + 1) == len(queries_data):
            elapsed   = time.time() - total_start
            done_now  = len(set(r['query_id'] for r in all_results))
            print(f'[Run] {done_now:3d}/{len(queries_data)}  |  avg {sum(query_times)/len(query_times):.0f}ms/query (this session)')
            robust_save(lambda p: pd.DataFrame(all_results).to_csv(p, index=False), CHECKPOINT_RESULTS_PATH)

        if (time.time() - SCRIPT_START) / 60 > TIME_BUDGET_MINUTES:
            robust_save(lambda p: pd.DataFrame(all_results).to_csv(p, index=False), CHECKPOINT_RESULTS_PATH)
            done_now = len(set(r['query_id'] for r in all_results))
            print(f'[Run] Time budget ({TIME_BUDGET_MINUTES} min) reached at {done_now}/{len(queries_data)} queries -- checkpoint saved. Rerun this same job to resume.')
            sys.exit(0)

    results_df_out = pd.DataFrame(all_results)
    robust_save(lambda p: results_df_out.to_csv(p, index=False), FINAL_RESULTS_PATH)
    if CHECKPOINT_RESULTS_PATH.exists():
        CHECKPOINT_RESULTS_PATH.unlink()

AVG_LATENCY_MS = (sum(query_times) / len(query_times)) if query_times else float('nan')
print(f'[Run] Done! Avg latency (this session): {AVG_LATENCY_MS:.0f}ms')


# ## Evaluation -- NDCG, Precision, Recall, F1 @ k=10,50,100,300,500,1000

# In[ ]:


print('[Eval] Loading production labels...')
production_df = pd.read_excel('dataset/production_results.xlsx')

K_VALUES = [10, 50, 100, 300, 500, 1000]

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
        results_df_out[results_df_out['query_id'] == qid]
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
eval_df.to_csv(RESULT_DIR / 'evaluation_colbert.csv', index=False)
print(f'[Eval] Saved to {RESULT_DIR}/evaluation_colbert.csv')


# ## Final Summary

# In[ ]:


print('[Summary] ============================================================')
print('[Summary] ColBERTv2 (MaxSim, brute-force) RESULTS')
print(f'\n[Summary] Avg query latency : {AVG_LATENCY_MS:.0f}ms')
print(f'[Summary] Companies encoded : {len(all_companies):,}')
print('[Summary] Token embedding dim: 128')
print(f'[Summary] Queries evaluated : {len(queries_data)}')
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

print()
print('[Summary] Comparison against models already tested (Recall@1000):')
print('  E5-Mistral        : 0.274')
print('  BM25              : 0.635')
print('  Nomic             : 0.684')
print('  BGE-M3            : 0.704')
print('  BGE summary-only  : 0.707  (prefix-fixed)')
print('  BGE all-fields    : 0.727  (prefix-fixed)')
print('  OpenAI large      : 0.728')
print('  MiniLM            : 0.741')
print('  SFR-Mistral       : 0.746')
print('  Linq-Mistral      : 0.749')
print('  GTE-large         : 0.768  (best solo so far)')
r1000 = eval_df[eval_df["k"] == 1000]["recall"].mean()
print(f'  ColBERTv2 (MaxSim): {r1000:.3f}')

