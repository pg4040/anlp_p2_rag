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

# Function to generate predicted answers using Hugging Face model
def generate_predicted_answers(prompts):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
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
    predicted_answers = generate_predicted_answers(prompts)

    # Evaluate metrics
    f1, accuracy, recall = evaluation_metrics(answers, predicted_answers)

    # Print metrics
    #print("F1 Score:", f1)
    #print("Exact Match Accuracy:", accuracy)
    #print("Recall:", recall)

if __name__ == "__main__":
    main()

