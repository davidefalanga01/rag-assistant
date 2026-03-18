from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os
import time

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

#llm = Groq(model="moonshotai/kimi-k2-instruct", api_key=groq_api_key)
llm = Ollama(model="qwen3.5:0.8b",
             #temperature=1.0,
             #top_p=1.0,
             #top_k=20,
             #min_p=0.0,
             #presence_penalty=2.0, 
             #repetition_penalty=1.0,
             request_timeout=120
             )

start_time = time.time()

response = llm.complete("Explain what is a Retrieval-Augmented Generation (RAG)")
end_time = time.time()
print(f"Response time: {end_time - start_time} seconds")
print(response)

messages = [
    ChatMessage(
        role="system", content="You are a pirate that are not able to park his vessel"
    ),
    ChatMessage(
        role="user", content="What is your name?"
    ),
]

resp = llm.chat(messages)

print(resp)