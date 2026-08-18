from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

response = client.chat_completion(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print(response)