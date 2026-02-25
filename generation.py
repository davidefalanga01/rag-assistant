from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="moonshotai/kimi-k2-instruct", api_key=groq_api_key)

response = llm.complete("Explain what is a GPU")
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