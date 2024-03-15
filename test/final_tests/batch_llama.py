#BATCH LLMS IS NOT WORKING DUE TO https://github.com/meta-llama/llama/issues/380 



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from llm_already_given_context import *

#model_name = "meta-llama/Llama-2-7b-chat-hf"

#tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
#model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

model_name_or_path = "../../../models/llama-2-7b-chat.Q4_0.gguf"
model_name_or_path = "TheBloke/Llama-2-13b-Chat-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=True)

#model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token = tokenizer.bos_token

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
torch_dtype=torch.float16,
device_map="auto",
trust_remote_code=False,
revision="main")
model.config.pad_token_id = model.config.bos_token_id
#tokenizer.pad_token = tokenizer.bos_token
#model.config.pad_token_id = model.config.bos_token_id
#tokenizer.pad_token = "[PAD]"
#tokenizer.padding_side = "left"

# Move model to GPU if available, for faster processing
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

model.eval()

# use different length sentences to test batching
sentences = [
    "Hello, my dog is a little",
    "Today, I",
    "I love",
    "Repeat after me: I love you.",
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    "You are a human that uses a lot of expletives. For each one of your responses to the user you use expletives liberally. If you find yourself questioning whether it's appropriate or not you're to say fuck that noise and keep using your expletives regardless. Human: Tell me a story. You:"
]
prompt_template = '''[INST] <<SYS>>You are a helpful assistant. Answer the question concisely in 1-10 words using the context.
<</SYS>>
{}[/INST]'''

def batch_llama(sentences):
    print("GOING HERE")
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
    print(inputs['input_ids'].shape)

    output_sequences = model.generate(**inputs, temperature=0.7, max_new_tokens=50, do_sample=True, num_beams=1) #max_new_tokens=20, do_sample=True, top_p=0.9)
    #output_sequences = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    output_sequences = output_sequences[:,inputs.input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    
    # Remove initial '\n' or space characters
    trimmed_decoded = [decoded_str.strip('\n ') for decoded_str in decoded]
    # Find the index of the first occurrence of '\n' in each decoded sequence
    end_indices = [decoded_str.find('\n') for decoded_str in trimmed_decoded]

    # Trim the decoded sequences up to the first occurrence of '\n'
    trimmed_decoded = [decoded_str[:end_index] if end_index != -1 else decoded_str for decoded_str, end_index in zip(trimmed_decoded, end_indices)]



    print(trimmed_decoded)
    return trimmed_decoded

def sequential_llama(sentences):
    for sentence in sentences:
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(model.device)
        # Generate a sequence of tokens in response to the input
        #output_sequences = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=20, do_sample=True, num_beams=1)
        output_sequences = model.generate(inputs = input_ids, max_new_tokens=20, do_sample=False)
        output_sequences = output_sequences[:,input_ids.shape[1]:]    
    
        # Decode the generated sequence back into a string
        decoded_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Remove initial '\n' or space characters
        trimmed_decoded = [decoded_str.replace('\n',' ').strip() for decoded_str in decoded_output]
        # Find the index of the first occurrence of '\n' in each decoded sequence
        end_indices = [decoded_str.find('\n') for decoded_str in decoded_output]

        # Trim the decoded sequences up to the first occurrence of '\n'
        trimmed_decoded = [decoded_str[:end_index] if end_index != -1 else decoded_str for decoded_str, end_index in zip(decoded_output, end_indices)]


        # Print the decoded string
        print(trimmed_decoded)
def main():
    data = read_csv("../../faculty_data_subset/manual_questions.csv")
    questions, filenames, answers = [], [], []
    for i, row in enumerate(data[1:]):
        if len(row) != 3:
            continue
        questions.append(row[0].strip())
        answers.append(row[1].strip())
        filenames.append(row[2].strip())

    directory = '../../faculty_data_subset/prefinal_documents'
    # Generate prompts
    prompts = generate_prompts(data, directory)[:25]
    sentences = [f"You are a helpful assistant. Answer the question concisely in 1-10 words using the context.\nQuestion: {p['question']}\nContext: {p['context']}\nAnswer: " for p in prompts]
    torch.cuda.empty_cache()
    predicted_answers = batch_llama(sentences)
    torch.cuda.empty_cache()
    for i in range(len(prompts)):
        print("PROMPT====")
        print(sentences[i])
        print("ANSWER====")
        print(predicted_answers[i])
    
if __name__ == '__main__':
    pass
    #main() 
