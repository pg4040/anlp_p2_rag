import time
from transformers import pipeline



# Prepare your batch of prompts
prompts = []
with open('rag_prompt_file.txt', 'r') as file:
    for line in file:
        prompts.append(line.strip())  # Remove newline characters
print(len(prompts))

model_name = "bigscience/llama-2"
generator = pipeline("text-generation", model=model_name)

