from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import os
import time

load_dotenv()

def generation_model():
    groq_api_key = os.getenv("GROQ_API_KEY")

    llm = Groq(model="moonshotai/kimi-k2-instruct", api_key=groq_api_key)

    return llm