'''Evaluate the overall RAG assistant performance on a set of test queries.'''
import os

from llama_index.core.evaluation import generate_question_context_pairs, RetrieverEvaluator
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from dotenv import load_dotenv

from generation import generation_model
from vector_database import load_raw_nodes

load_dotenv()

def generate_dataset(
    nodes,
    llm,
    num_questions_per_chunk: int = 2,
):
    
    print(
        f"Generating {num_questions_per_chunk} question(s) per chunk "
        f"across {len(nodes)} nodes …"
    )
    dataset = generate_question_context_pairs(
        nodes,
        llm=llm,
        num_questions_per_chunk=num_questions_per_chunk,
    )
    print(f"Generated {len(dataset.queries)} Q&A pairs.")
    return dataset


def main():
    llm = generation_model()
 
    # 1. Load nodes from Chroma
    nodes = load_raw_nodes(
        db_path="./chroma_db",
        collection_name="rag_collection",
    )
 
    # 2. Generate Q&A pairs
    dataset = generate_dataset(
        nodes=nodes,
        llm=llm,
        num_questions_per_chunk=2,
    )
 
    # 3. Persist
    dataset.save_json("data/eval_rag_dataset.json")
    print(f"Generated question-context pairs for evaluation.")
 
 
if __name__ == "__main__":
    main()