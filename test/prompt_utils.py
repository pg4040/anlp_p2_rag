import csv
from transformers import pipeline

def question_prompt_reader(filename='questions_rag_prompts.csv'):
    questions_list = []
    rag_prompts_list = []

    # Open the file in read mode ('r') and create a csv.reader object
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader, None)

        for row in reader:
            rag_prompt = row[1].replace('\\n', '\n')
            rag_prompts_list.append(rag_prompt)
            questions_list.append(row[0]) #later use this to get the reference answer or store the reference answer in the same csv
    return questions_list, rag_prompts_list

if __name__ == '__main__':
    # Load the model and tokenizer
    generator = pipeline('text-generation', model='gpt-2')
    questions, prompts = question_prompt_reader()

    # Generate responses using batching
    # Adjust batch_size depending on your system's memory capacity and the complexity of the model
    batch_size = 5
    responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = generator(batch_prompts, max_length=50, num_return_sequences=1)
        responses.extend(batch_responses)

    # Print or process the responses
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}")
        print(f"Response: {response['generated_text']}\n")

