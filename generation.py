from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()

def generation_model(size="small"):
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if size == "big":
        llm = Groq(model="openai/gpt-oss-120b", api_key=groq_api_key)
    elif size == "judge":
        llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
    elif size == "extra":
        llm = Groq(model="openai/gpt-oss-20b", api_key=groq_api_key)
    else:
        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    return llm

if __name__ == "__main__":
    model = generation_model()
    prompt = "Spiega in poche righe cos'è il machine learning."
    
    start_time = time.time()
    response = model.complete(prompt)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    print(f"Tempo di risposta: {elapsed_time:.2f} secondi")
    print("Risposta:")
    print(response.text)