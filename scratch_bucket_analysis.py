import pandas as pd
import numpy as np

production_df = pd.read_excel('dataset/production_results.xlsx')

def get_relevant(qid, top_k=1000):
    return set(production_df[(production_df['query_id']==qid) & (production_df['rank']<=top_k)]['domain'].tolist())

METHODS = {
    'MiniLM':                    'result/03_baseline_minilm/minilm_results.csv',
    'GTE_large':                 'result/17_baseline_gte_large/gte_large_results.csv',
    'Linq_Mistral':              'result/20_baseline_linq_mistral/20_baseline_linq_mistral_results.csv',
    'Hybrid_RRF_top50_rerank':   'result/23_hybrid_minilm_linq_reranker/rrf_reranked_results.csv',
    'Hybrid_full_pool_rerank':   'result/24_hybrid_minilm_linq_full_rerank/reranked_results.csv',
}

BUCKETS = [(0,10),(10,50),(50,100),(100,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,800),(800,900),(900,1000)]

for name, path in METHODS.items():
    df = pd.read_csv(path, usecols=['query_id','rank','domain'])
    query_ids = sorted(df['query_id'].unique())
    relevant_cache = {qid: get_relevant(qid) for qid in query_ids}
    print(f'=== {name} ===')
    print(f'  {"bucket":<12}{"avg_hits_in_bucket":>20}{"bucket_precision":>18}')
    for lo, hi in BUCKETS:
        bucket_hits = []
        for qid in query_ids:
            relevant = relevant_cache[qid]
            retrieved = df[(df['query_id']==qid) & (df['rank']>lo) & (df['rank']<=hi)]['domain'].tolist()
            hits = len(set(retrieved) & relevant)
            bucket_hits.append(hits)
        avg_hits = np.mean(bucket_hits)
        bucket_size = hi - lo
        precision = avg_hits / bucket_size
        print(f'  {lo}-{hi:<8}{avg_hits:>20.2f}{precision:>18.3f}')
    print()
