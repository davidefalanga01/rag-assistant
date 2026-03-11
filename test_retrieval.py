'''Evaluate the Retrieval capabilities of the RAG assistant on a set of test queries.'''
import asyncio

from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

from rag import create_vector_db


async def test_retrieval(retriever):
    metrics = ["precision", "recall", "mrr", "hit_rate"]
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics,
        retriever=retriever,
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    return eval_results

if __name__ == "__main__":
    index = create_vector_db()
    retriever = index.as_retriever(similarity_top_k=2)

    qa_dataset = EmbeddingQAFinetuneDataset.from_json("data/eval_rag_dataset.json")
    print("Type:", type(qa_dataset))
    print("Queries:", next(iter(qa_dataset.queries.items())))
    print("Corpus:", next(iter(qa_dataset.corpus.items())))
    print("Relevant docs:", next(iter(qa_dataset.relevant_docs.items())))
    print("Number of queries:", len(qa_dataset.queries))
    #eval_results = asyncio.run(test_retrieval(retriever))
    #print(eval_results)