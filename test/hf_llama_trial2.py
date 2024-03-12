from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from prompt_utils import question_prompt_reader

# Load the tokenizer and model for the specified LLaMA version
#model_path  = 'meta-llama/Llama-2-7b-chat-hf'
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModelForCausalLM.from_pretrained(model_path)



from transformers import AutoTokenizer, pipeline, logging
import csv
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "TheBloke/Llama-2-13B-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
#tokenizer.pad_token = tokenizer.eos_token #TODO: Maybe have to change this
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
device_map="auto",
trust_remote_code=False,
revision="main")

# Move model to GPU if available, for faster processing
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

questions, prompts = question_prompt_reader()
#Single example
prompt = prompts[0]
prompt = """Question: What year was the paper titled "Median mandibular flexure—the unique physiological phenomenon of the mandible and its clinical significance in implant restoration" published? """
prompt_template=f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer the question in 1 sentence only. 
<</SYS>>
{prompt}[/INST]
'''

prompt = "Tell me about AI"
prompt_template=f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]

'''

prompt_template = 'Instruction: Answer the question concisely. Question: What is the capital of France? Answer: '
prompt_template_2 = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])

exit()


#To try adding padding to single sample
input_ids = tokenizer(prompt_template, return_tensors="pt", truncation=True,max_length=512).input_ids.to(device)
output = model.generate(inputs=input_ids, do_sample=True, max_new_tokens=30, num_return_sequences=1, temperature=0.7, repetition_penalty=1.5)
answer = tokenizer.decode(output[0], skip_special_tokens=True)
print('Shreya Input:\n', prompt_template)
print("=====")
print('Answer: \n', answer)
exit()



# Function to generate responses in batches
def generate_responses_in_batches(prompts, batch_size=2):
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,max_length=512).to(device)
        batch_outputs = model.generate(**batch_inputs, do_sample=True, max_new_tokens=30, num_return_sequences=1, temperature=0.7, top_p=0.95, top_k=40)
        batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in batch_outputs]
        responses.extend(batch_responses)
    return responses

# Generate responses
responses = generate_responses_in_batches(prompts, batch_size=2)

# Print or process the responses
for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")

