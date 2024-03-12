import torch
from transformers import RobertaForQuestionAnswering, RobertaTokenizer

model_name = "deepset/roberta-base-squad2"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForQuestionAnswering.from_pretrained(model_name)

# Define question and context
question = "What is the capital of France?"
context = "France is a country in Europe. Paris is the capital of France."

# Tokenize input
inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]

# Get model's answer
output = model(**inputs)
answer_start_scores, answer_end_scores = output.start_logits, output.end_logits

# Find the position of the start and end of the answer
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1

# Convert tokens to answer
answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
answer = tokenizer.convert_tokens_to_string(answer_tokens)

print(answer)

