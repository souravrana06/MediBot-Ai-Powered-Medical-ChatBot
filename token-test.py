from huggingface_hub import whoami
import os

print(whoami(token=os.environ.get("HF_TOKEN")))