# rag-assistant

Naive RAG Retriever and Generator

Retrieval Eval

GT generation
GT eval
GT filtering
Short GT (manually selected)
Generation eval

Evaluated with a synthetic dataset:
- retriever evaluation: ["precision", "recall", "mrr", "hit_rate"]
- generation evaluation: LLM-as-a-judge, or semantic similarity, faithfulness/completeness to context, answer relevancy



UI Added with Streamlit

Run the UI from this folder with:

```bash
streamlit run app.py
```

`python app.py` also redirects to Streamlit, but using `streamlit run` is the
native command and avoids Streamlit's bare-mode warnings.

Average Metrics Retrieval:
    retrievers  hit_rate       mrr  precision    recall        ap      ndcg
0  top-10 eval  0.950207  0.705088   0.095021  0.950207  0.705088  0.765139

Average Metrics Generation:
faithfulness_avg_score: 0.26666666666666666
relevancy_avg_score: 0.26666666666666666
correctness_avg_score: 3.7666666666666666
llm_judge_avg_score: 4.0
semantic_similarity_avg_score: 0.764165259591577


To create the vector db based on pdf files in the data folder

python '''python.exe vector_database.py'''

The idea: Naive -> Reranker/Adaptive-k -> Metadata/chunking strategy -> Hybrid search

To eval the retrieval part: '''python eval_retrieval.py'''


Done:
- Chunking strategies Document-Aware + Semantic split
- Embedding: BAAI/bge-large-en-v1.5   Financial/legal textStrong on domain text
- Reranking https://colemurray.medium.com/enhancing-rag-with-baai-bge-reranker-a-comprehensive-guide-fe994ba9f82a
- Adaptive k with SimilarityPostprocessor(similarity_cutoff=0.60) or with LLM
- Hybrid search dense vector + sparse keyword(BM25)
- filters by metadata
- review retrieval evaluation: not hard match + LLM as judge

## RAG variants

Le varianti RAG non sono codificate come preset Python: si ottengono impostando i singoli flag nel file `.env`.

| Variante | Reranker | Adaptive-k | Metadata filtering | Enhanced chunking | Hybrid search | Collection consigliata |
|---|---:|---:|---:|---:|---:|---|
| RAG naive | No | No | No | No | No | `rag_collection_simple` |
| RAG reranker + adaptive-k | Si | Si | No | No | No | `rag_collection_simple` |
| RAG reranker + adaptive-k + metadata + chunking | Si | Si | Si | Si | No | `rag_collection_enhanced` |
| RAG full | Si | Si | Si | Si | Si | `rag_collection_enhanced` |

### Flag disponibili

NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=False
USE_RERANKER=False
USE_ADAPTIVE_K=False
USE_METADATA_FILTERING=False
USE_HYBRID_SEARCH=False

COLLECTION_NAME=rag_collection_simple
DB_PATH=./chroma_db

FINAL_TOP_K=5
CANDIDATE_TOP_K=20
SIMILARITY_CUTOFF=0.60

### Esempi `.env`

RAG naive:
NEW_EMBED_MODEL=False
ENHANCED_CHUNKING=False
USE_RERANKER=False
USE_ADAPTIVE_K=False
USE_METADATA_FILTERING=False
USE_HYBRID_SEARCH=False
COLLECTION_NAME=rag_collection

RAG reranker + adaptive-k:
NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=False
USE_RERANKER=True
USE_ADAPTIVE_K=True
USE_METADATA_FILTERING=False
USE_HYBRID_SEARCH=False
COLLECTION_NAME=rag_collection_v1

RAG reranker + adaptive-k + metadata + chunking:
NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=True
USE_RERANKER=True
USE_ADAPTIVE_K=True
USE_METADATA_FILTERING=True
USE_HYBRID_SEARCH=False
COLLECTION_NAME=rag_collection_v2

RAG full:
NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=True
USE_RERANKER=True
USE_ADAPTIVE_K=True
USE_METADATA_FILTERING=True
USE_HYBRID_SEARCH=True
COLLECTION_NAME=rag_collection_v2
