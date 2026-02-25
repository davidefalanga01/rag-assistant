from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_completition = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the RAG system",
        }
    ],
    model="moonshotai/kimi-k2-instruct"
)

print(chat_completition.choices[0].message.content)


