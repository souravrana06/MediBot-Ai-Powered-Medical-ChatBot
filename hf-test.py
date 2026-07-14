import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HF_TOKEN")
)

response = client.text_generation(
    "What is diabetes?",
    max_new_tokens=50
)

print(response)