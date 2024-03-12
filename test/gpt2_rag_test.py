from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Instruction: Answer the question concisely in 1 sentence.\nQuestion:  When was oxygen discovered?\nAnswer: "
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

