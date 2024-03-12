from transformers import T5ForConditionalGeneration, T5Tokenizer
from prompt_utils import question_prompt_reader
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Prepare the input text
question = "What is the capital of France?"
context = "France is a country in Europe. Paris is the capital of France."
input_text = f"question: {question} context: {context}"

# Tokenize and prepare inputs for the model
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate the answer
output = model.generate(input_ids)
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print(answer)

questions, prompts = question_prompt_reader()
#Single example
for i in range(100):
    prompt = prompts[i]
    print("Question======")
    print(prompt)

    input_ids = tokenizer.encode(prompt, max_length=1024, return_tensors="pt")

    # Generate the answer
    output = model.generate(input_ids)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    print("====Answer=====")
    print(answer)


