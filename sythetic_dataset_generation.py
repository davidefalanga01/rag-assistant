#'''Evaluate the overall RAG assistant performance on a set of test queries.'''
import os

from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Load documents
documents = SimpleDirectoryReader("data/").load_data()

parser = SentenceSplitter(
    chunk_size=800,
    chunk_overlap=100
)


# Split data in nodes
nodes = parser.get_nodes_from_documents(documents)
print(f"Loaded {len(nodes)} nodes from documents.")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="moonshotai/kimi-k2-instruct", api_key=groq_api_key)

testset = generate_question_context_pairs(nodes=nodes, 
                                          llm=llm,
                                          num_questions_per_chunk=2)

testset.save_json("data/eval_rag_dataset.json")

print(testset)