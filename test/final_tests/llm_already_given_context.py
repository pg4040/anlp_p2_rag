#TODO
#Read the manual_questions.csv - format question, answer, filename
#For each record, prepare a prompt with question, filename
#use LLM, generate an answer - store it in predicted answers

#Using ground truth answer and predicted answer, evaluate the 3 metrics - F1 score, exact match accuracy and recall

import csv
from transformers import pipeline
from sklearn.metrics import f1_score, accuracy_score, recall_score
import os
from collections import Counter
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Function to read manual_questions.csv
def read_csv(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def generate_prompts(data, file_directory):
    prompts = []
    for i, row in enumerate(data[1:]):
        if len(row)!=3: continue
        question, _, filename = row
        if not filename.endswith('.txt'):
            filename+='.txt'
        try:
            context = read_file_content(filename, file_directory)
        except:
            print(i, row)
            exit()
        prompt = {"question": question, "context": context}
        prompts.append(prompt)
    return prompts

def read_file_content(file_path, file_directory):
    if ',' in file_path:
        content = ''
        paths = file_path.split(',')
        for p in paths:
            p = os.path.join(file_directory, p.strip())
            with open(p.strip(), 'r', encoding='utf-8') as file:
                content += file.read() + '\n'
        return content
    file_path = os.path.join(file_directory, file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def t5_answers(prompts):
    #Exact Match: 0.00%
    #F1 Score: 31.64
    #Recall Score: 59.31
    #Partial Match (Jaccard Similarity): 0.59%

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    answers = []
    for prompt in prompts:
        print("PROMPT=====")
        print(prompt)
        prompt = f"Question: {prompt['question']}\n Context:{prompt['context']}"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Batch size 1
        outputs = model.generate(input_ids, max_length=30, num_return_sequences=1)
        
        # Decode the output
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers.append(answer)
        print("T5 PREDICTED ANSWER=====")
        print(answer)
    return answers

from llama_cpp import Llama
def llama_cpp_trial(prompts):
    llama = Llama(model_path="../../../models/llama-2-7b-chat.Q4_0.gguf", n_ctx=2048, n_gpu_layers=30)
    answers = []

    for prompt in prompts:
        print("PROMPT=====")
        print(prompt)
        prompt = f"Question: {prompt['question']}\n Context:{prompt['context']}"
        # Generate a response
        response = llama.create_chat_completion(messages=[{"role": "system", "content": "You are a helpful assistant. Answer the question concisely in 1-10 words using the context."}, {"role": "user", "content": prompt}])
        
        # Extract the generated text
        answer = response['choices'][0]['message']['content']
        print("LLAMA-7B PREDICTED ANSWER=====")
        print(answer)
        answers.append(answer)
    return answers        


#Function to generate predicted answers using Hugging Face model
def generate_predicted_answers(prompts):
    #qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    #Exact Match: 20.00%
    #F1 Score: 61.37
    #Recall Score: 64.45
    #Partial Match (Jaccard Similarity): 0.20%
    #qa_pipeline = pipeline("question-answering", model="mrm8488/t5-base-finetuned-question-generation-ap", tokenizer="t5-base")
    #Exact Match: 0.00%
    #F1 Score: 40.25
    #Recall Score: 50.26
    #Partial Match (Jaccard Similarity): 1.19%

    predicted_answers = []
    for prompt in prompts:
        answer = qa_pipeline(prompt)["answer"]
        predicted_answers.append(answer)
        print("PROMPT=====")
        print(prompt)
        print("ROBERTA PREDICTED ANSWER=====")
        print(answer)
    return predicted_answers

def jaccard_similarity(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union != 0 else 0

def evaluation_metrics(ground_truth, predictions):
    exact_match = 0
    total = len(predictions)
    f1_total = 0
    partial_match_total = 0
    recall_total = 0
    for pred_answer, truth in zip(predictions, ground_truth):
        
        # Exact match
        if pred_answer == truth:
            exact_match += 1
        
        # F1 score
        common = Counter(pred_answer) & Counter(truth)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = num_same / len(pred_answer)
            recall = num_same / len(truth)
            f1 = (2 * precision * recall) / (precision + recall)
            recall_total+= recall
        f1_total += f1
        
        # Partial match (Jaccard similarity)
        partial_match_score = max(jaccard_similarity(pred_answer, answer) for answer in truth)
        partial_match_total += partial_match_score
    
    exact_match_score = 100.0 * exact_match / total
    f1_score = 100.0 * f1_total / total
    partial_match_score = 100.0 * partial_match_total / total
    recall_score = 100.0 * recall_total / total

    print(f"Exact Match: {exact_match_score:.2f}%")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Recall Score: {recall_score:.2f}")
    print(f"Partial Match (Jaccard Similarity): {partial_match_score:.2f}%")
    return f1_score, exact_match_score, recall_score

def read_from_log(filename):
    answers = []
    cur_answer = ''
    start = 'LLAMA-7B PREDICTED ANSWER====='
    end = "PROMPT====="
    start_found  = False
    with open(filename, 'r') as f:
        for line in f:
            #print(line)
            if line.strip().strip('\n') == start and not start_found:
                start_found=True
            elif line.strip().strip('\n') == end:
                start_found = False
                if cur_answer != '':
                    answers.append(cur_answer.strip().strip('\n'))
                cur_answer = ''
            elif start_found:
                cur_answer+=line.strip().strip('\n')+'\n'
                #print("cur_answer: ", cur_answer)
    if not start_found:
        start_found = False
        answers.append(cur_answer.strip().strip('\n'))
        cur_answer = ''
        
    return answers

from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity

def compute_answer_BERT_similarity(question, answers, predicted_answers):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    print(len(question), len(answers), len(predicted_answers))
    print(question[-1], answers[-1], predicted_answers[-1])
    # Storage for cosine similarities
    similarities = []

    for i in range(len(predicted_answers)):
        # Format the ground truth and predicted strings
        ground_truth = f"Q: {question[i]}\nA: {answers[i].strip().strip('\n')}"
        predicted = f"Q: {question[i]}\nA: {predicted_answers[i].strip().strip('\n')}"

        # Tokenize and encode both strings to BERT's format
        tokens_truth = tokenizer(ground_truth, return_tensors='pt', padding=True, truncation=True)
        tokens_pred = tokenizer(predicted, return_tensors='pt', padding=True, truncation=True)

        # Get BERT embeddings
        with torch.no_grad():
            embeddings_truth = model(**tokens_truth).last_hidden_state.mean(dim=1)
            embeddings_pred = model(**tokens_pred).last_hidden_state.mean(dim=1)

        # Compute cosine similarity
        similarity = cosine_similarity(embeddings_truth, embeddings_pred).item()
        similarities.append(similarity)

    # Compute overall similarity score
    overall_similarity = sum(similarities) / len(similarities) * 100  # Scale to 100

    return overall_similarity

# Main function
def main():
    # Read data
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
    prompts = generate_prompts(data, directory)

    # Generate predicted answers
    #predicted_answers = llama_cpp_trial(prompts)
    predicted_answers = read_from_log('answer_log.txt')
    print("predicted answers===")
    print(predicted_answers)

    # Evaluate metrics
    f1, accuracy, recall = evaluation_metrics(answers, predicted_answers)

    semantic_similarity = compute_answer_BERT_similarity(questions, answers, predicted_answers)
    print("semantic similarity :", semantic_similarity)
    # Print metrics
    #print("F1 Score:", f1)
    #print("Exact Match Accuracy:", accuracy)
    #print("Recall:", recall)

if __name__ == "__main__":
    main()

