from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from prompt_utils import question_prompt_reader

# Load the tokenizer and model for the specified LLaMA version
#model_path  = 'meta-llama/Llama-2-7b-chat-hf'
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModelForCausalLM.from_pretrained(model_path)



from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "TheBloke/Llama-2-7B-GPTQ"
model_basename = "gptq_model-4bit-128g"
use_triton = False
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
model_basename=model_basename,
use_safetensors=True,
trust_remote_code=True,
device="cuda:0",
use_triton=use_triton,
quantize_config=None)


# Move model to GPU if available, for faster processing
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

questions, prompts = question_prompt_reader()

# Function to generate responses in batches
def generate_responses_in_batches(prompts, batch_size=2):
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size].to(device)
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        batch_outputs = model.generate(**batch_inputs, max_length=100, num_return_sequences=1, temperature=0.7)
        batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in batch_outputs]
        responses.extend(batch_responses)
    return responses

# Generate responses
responses = generate_responses_in_batches(prompts, batch_size=2)

# Print or process the responses
for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")

