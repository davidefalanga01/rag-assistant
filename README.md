# rag-assistant

## Descrizione del progetto

`rag-assistant` e un assistente RAG per interrogare documenti PDF, con focus su filing finanziari 10-K. Il progetto indicizza i documenti in un vector database Chroma, recupera i chunk piu rilevanti tramite LlamaIndex e genera risposte usando un modello LLM via Groq.

Il progetto include:

- creazione e caricamento di un vector database persistente;
- chunking semplice oppure chunking avanzato document-aware per documenti 10-K;
- retrieval denso basato su embeddings HuggingFace;
- retrieval ibrido dense vector + BM25;
- reranking con BAAI/bge-reranker;
- adaptive-k tramite soglia di similarita;
- filtri su metadati inferiti dalla query;
- UI Streamlit per interrogare i documenti;
- pipeline di generazione ground truth e valutazione;
- metriche di retrieval e generation evaluation.

## Architettura ad alto livello

Flusso principale:

1. I PDF vengono letti dalla cartella `data/`.
2. `vector_database.py` divide i documenti in chunk e li indicizza in Chroma.
3. `rag.py` carica l'indice, costruisce il retriever configurato e crea il query engine.
4. `generation.py` seleziona il modello LLM Groq per generazione o valutazione.
5. `app.py` espone una UI Streamlit per fare domande e visualizzare fonti e metadati.
6. Gli script `eval_retrieval.py` ed `eval_generation.py` misurano rispettivamente retrieval e generazione.

Componenti principali:

| File | Ruolo |
|---|---|
| `app.py` | Interfaccia Streamlit. Mostra configurazione attiva, input utente, risposta e documenti sorgente. |
| `config.py` | Carica variabili `.env`, flag RAG, modelli, path DB e parametri top-k. |
| `vector_database.py` | Lettura PDF, chunking, creazione/caricamento Chroma, dense/hybrid retriever. |
| `rag.py` | Costruzione del retriever configurato, postprocessor, query engine ed entrypoint CLI. |
| `generation.py` | Factory dei modelli Groq usati per generazione, giudice e reference answer. |
| `retrieval_filters.py` | Inferenza di filtri metadata-based dalla query naturale. |
| `eval_retrieval.py` | Valutazione del retrieval con metriche hard, soft e opzionalmente LLM judge. |
| `eval_generation.py` | Valutazione delle risposte generate con faithfulness, relevancy, correctness, judge e semantic similarity. |
| `gt_generation.py` | Generazione delle reference answer a partire dal dataset di retrieval. |
| `eval_gt.py` | Valutazione della qualita delle ground truth generate. |
| `filter_gt.py` | Filtraggio del dataset GT e costruzione di un dataset breve per evaluation rapida. |
| `plot_results.ipynb` | Notebook per analisi e visualizzazione dei risultati. |

## Requisiti

- Python 3.10+ consigliato.
- Ambiente virtuale Python.
- Dipendenze in `requirements.txt`.
- Chiave API Groq in `.env`.
- PDF da indicizzare nella cartella `data/`.

Il progetto usa principalmente:

- `llama-index`;
- `chromadb`;
- `sentence-transformers` / embeddings HuggingFace;
- `BAAI/bge-large-en-v1.5` per embeddings avanzati;
- `BAAI/bge-reranker-base` per reranking;
- `llama-index-retrievers-bm25` per hybrid search;
- `streamlit` per la UI;
- `pandas`, `numpy`, `scikit-learn` per evaluation e metriche.

## Setup

Dalla cartella `rag-assistant`:

```bash
python -m venv venv
```

Attivazione su Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Installazione dipendenze:

```bash
pip install -r requirements.txt
```

Creare o aggiornare il file `.env`:

```env
GROQ_API_KEY=your_groq_api_key

NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=False
USE_RERANKER=False
USE_ADAPTIVE_K=False
USE_METADATA_FILTERING=False
USE_HYBRID_SEARCH=False

DB_PATH=./chroma_db
COLLECTION_NAME=rag_collection

FINAL_TOP_K=5
CANDIDATE_TOP_K=20
SIMILARITY_CUTOFF=0.60
```

## Preparazione dei dati

Inserire i PDF nella cartella:

```text
rag-assistant/data/
```

Lo script di indicizzazione legge i file `.pdf` presenti in `data/`.

Per creare o aggiornare il vector database:

```bash
python vector_database.py
```

Il database viene salvato in `DB_PATH`, di default:

```text
./chroma_db
```

Se la collection contiene gia nodi, lo script evita duplicati usando un hash del contenuto del chunk.

## Avvio della UI

Comando consigliato:

```bash
streamlit run app.py
```


`app.py` rilancia automaticamente il processo con Streamlit se viene eseguito direttamente con Python.

La UI mostra:

- collection Chroma attiva;
- flag RAG attivi;
- campo domanda;
- risposta generata;
- source nodes con score, metadati e contenuto.

## Configurazione `.env`

### Modello embedding

```env
NEW_EMBED_MODEL=True
```

Se `True`, usa:

```text
BAAI/bge-large-en-v1.5
```

Se `False`, usa:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Importante: il modello di embedding usato in query deve essere coerente con quello usato per creare la collection. Se si cambia `NEW_EMBED_MODEL`, conviene usare una nuova `COLLECTION_NAME` o rigenerare l'indice.

### Chunking

```env
ENHANCED_CHUNKING=False
```

Se `False`, usa `SentenceSplitter` con:

- `chunk_size=1024`;
- `chunk_overlap=100`.

Se `True`, usa una strategia specifica per filing 10-K:

- split rigido sugli header SEC `ITEM`;
- rilevamento sezioni come `ITEM 1A`, `ITEM 7`, `ITEM 8`;
- metadati finanziari e documentali;
- preservazione dei blocchi tabellari;
- split semantico sulle sezioni narrative lunghe;
- fallback sentence splitter in caso di errore.

### Retrieval

```env
USE_RERANKER=False
USE_ADAPTIVE_K=False
USE_METADATA_FILTERING=False
USE_HYBRID_SEARCH=False
```

`USE_RERANKER=True` abilita `FlagEmbeddingReranker` con:

```text
BAAI/bge-reranker-base
```

`USE_ADAPTIVE_K=True` applica una soglia di similarita tramite `SimilarityPostprocessor`. Se nessun nodo supera la soglia, viene mantenuto il miglior candidato come fallback.

`USE_METADATA_FILTERING=True` inferisce filtri dalla query e li applica al retrieval. Esempi di filtri:

- ticker, ad esempio Apple/AAPL;
- fiscal year;
- form type 10-K;
- SEC item;
- section group;
- chunk type table;
- financial statement type.

`USE_HYBRID_SEARCH=True` combina:

- dense retrieval su Chroma;
- sparse retrieval BM25;
- reciprocal rank fusion;
- deduplicazione dei risultati.

### Parametri top-k

```env
FINAL_TOP_K=5
CANDIDATE_TOP_K=20
SIMILARITY_CUTOFF=0.60
```

## Varianti RAG consigliate

### Naive RAG

Configurazione minima, utile come baseline:

```env
NEW_EMBED_MODEL=False
ENHANCED_CHUNKING=False
USE_RERANKER=False
USE_ADAPTIVE_K=False
USE_METADATA_FILTERING=False
USE_HYBRID_SEARCH=False
COLLECTION_NAME=rag_collection
```

### Dense RAG con embedding avanzato

```env
NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=False
USE_RERANKER=False
USE_ADAPTIVE_K=False
USE_METADATA_FILTERING=False
USE_HYBRID_SEARCH=False
COLLECTION_NAME=rag_collection_v1
```

### Reranker + adaptive-k

```env
NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=False
USE_RERANKER=True
USE_ADAPTIVE_K=True
USE_METADATA_FILTERING=False
USE_HYBRID_SEARCH=False
COLLECTION_NAME=rag_collection_v1
```

### Metadata + enhanced chunking

```env
NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=True
USE_RERANKER=True
USE_ADAPTIVE_K=True
USE_METADATA_FILTERING=True
USE_HYBRID_SEARCH=False
COLLECTION_NAME=rag_collection_v2
```

### Full RAG

```env
NEW_EMBED_MODEL=True
ENHANCED_CHUNKING=True
USE_RERANKER=True
USE_ADAPTIVE_K=True
USE_METADATA_FILTERING=True
USE_HYBRID_SEARCH=True
COLLECTION_NAME=rag_collection_v2
```

## Modelli LLM

I modelli sono definiti in `generation.py`:

| Size | Modello Groq | Uso previsto |
|---|---|---|
| `small` | `llama-3.1-8b-instant` | default leggero |
| `extra` | `openai/gpt-oss-20b` | generazione nella query engine di default |
| `big` | `openai/gpt-oss-120b` | reference answer / valutazioni piu robuste |
| `judge` | `llama-3.3-70b-versatile` | LLM-as-a-judge |

`build_query_engine()` usa `generation_model("extra")` se non viene passato un LLM esplicito.

## Metadati prodotti dal chunking avanzato

Quando `ENHANCED_CHUNKING=True`, i nodi possono includere metadati come:

- `company_name`;
- `ticker`;
- `form_type`;
- `fiscal_year`;
- `period_end_date`;
- `filing_date`;
- `source_file`;
- `source_page`;
- `sec_item`;
- `section_title`;
- `section_group`;
- `statement_type`;
- `chunk_type`;
- `section_index`;
- `chunk_index`;
- `document_index`.

Per Apple 2024, il parser riconosce automaticamente:

- company: `Apple Inc.`;
- ticker: `AAPL`;
- fiscal year: `2024`;
- period end date: `2024-09-28`;
- filing date: `2024-11-01`.

## Esempi di query supportate dai filtri metadata

Con `USE_METADATA_FILTERING=True`, query come queste possono restringere automaticamente la ricerca:

```text
What are Apple's risk factors in the 2024 10-K?
```

```text
Show the cash flow statement for AAPL fiscal year 2024.
```

```text
What does Item 7 say about liquidity and capital resources?
```

```text
Find tables about Apple's share repurchases.
```

## Valutazione retrieval

Esecuzione:

```bash
python eval_retrieval.py
```


Metriche hard:

- `hit_rate`;
- `mrr`;
- `precision`;
- `recall`;
- `ap`;
- `ndcg`.

Metriche soft basate su similarita embedding:

- `soft_hit_rate`;
- `soft_precision`;
- `soft_recall`;
- `soft_mrr`;
- `soft_avg_retrieved_max_similarity`;
- `soft_avg_reference_max_similarity`;
- `soft_relevant_retrieved_count`;
- `soft_covered_reference_count`.


## Valutazione generation

Esecuzione:

```bash
python eval_generation.py
```

Metriche:

- faithfulness;
- relevancy;
- correctness;
- LLM-as-a-judge;
- semantic similarity.

Nel file `eval_generation.py`, `SHORT_EVAL=True` usa un sottoinsieme manuale di query easy/medium/hard definito in `filter_gt.py`. Per valutare tutto il dataset, impostare:

```python
SHORT_EVAL = False
```

Anche la generation evaluation supporta resume: se trova risultati per-query gia salvati, salta le query completate.

## Pipeline consigliata end-to-end

1. Inserire i PDF in `data/`.
2. Configurare `.env`.
3. Creare la collection:

```bash
python vector_database.py
```

4. Provare la UI:

```bash
streamlit run app.py
```

5. Valutare retrieval:

```bash
python eval_retrieval.py
```

6. Generare reference answer:

```bash
python gt_generation.py
```

7. Valutare generazione:

```bash
python eval_generation.py
```

8. Analizzare risultati con `plot_results.ipynb`.

## Output e file generati

| Path | Descrizione |
|---|---|
| `chroma_db/` | Database Chroma persistente. |
| `data/retrieval_eval_results.csv` | Medie delle metriche retrieval. |
| `data/retrieval_eval_results_per_query.csv` | Dettaglio retrieval per query. |
| `data/eval_rag_dataset_with_refs.json` | Dataset con reference answer generate. |
| `data/generation_eval_results.csv` | Medie metriche generation. |
| `data/generation_eval_results_per_query.csv` | Dettaglio generation per query. |
| `generation_eval_results.json` | File legacy o risultato precedente in formato JSON. |
| `gt_eval_results.json` | Risultati di valutazione ground truth. |

Average Metrics Retrieval:  
    retrievers  hit_rate    mrr precision   recall  ap  ndcg  
top-10 eval 0.950207    0.705088    0.095021    0.950207    0.705088    0.765139