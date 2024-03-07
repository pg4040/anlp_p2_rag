from openai import OpenAI
import openai
import time

client = OpenAI(api_key='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')

def safe_call_openai(prompt):
    try:
        response = client.completions.create(model="gpt-3.5-turbo-instruct", 
        prompt=prompt,
        temperature=0.5,
        n=1,
        max_tokens=1024)
        return response
    except openai.RateLimitError:
        print("Rate limit exceeded, waiting before retrying...")
        time.sleep(60)  # Wait for 60 seconds before retrying
        return safe_call_openai(prompt)  # Retry the request

# Assume you have set up OpenAI API key
def generate_qa_pairs(text):
    """
    Generates simple QA pairs from a given text using an instruction-finetuned LLM.
    
    Parameters:
    - text (str): The input text from which to generate QA pairs.
    
    Returns:
    - List of tuples, where each tuple is a question-answer pair.
    """
    
    prompt=f"Generate simple QA pairs based on the following text, focusing on key details such as names, positions, research areas, office locations, contact information, and websites:\n\n{text}"
    safe_call_openai(prompt) 
    qa_text = response.choices[0].text.strip()
    
    # Assuming the model returns QA pairs in a simple format, split them into a list.
    # This is a simplistic approach; you might need to adapt parsing based on the actual output format.
    qa_pairs = qa_text.split('\n')
    parsed_qa_pairs = []
    for pair in qa_pairs:
        if pair:
            question, answer = pair.split('?')
            parsed_qa_pairs.append((question.strip() + '?', answer.strip()))
    
    return parsed_qa_pairs

if __name__ == '__main__':
    prof_name = 'bisk-yonatan'
    file_path = f"../{prof_name}.txt"  # Update this to your actual file path
    with open(file_path, "r") as file:
        text = file.read()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(text)

    for question, answer in qa_pairs:
        print(f"Q: {question}\nA: {answer}\n")

