import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
from collections import Counter
from sklearn.metrics import average_precision_score

#Recall@K (K=1,2,3) #retrieval metric
def recall_at_k(retrieved, ground_truth, k=1):
    recall_scores = []
    for retrieved_docs, true_docs in zip(retrieved, ground_truth):
        hit_count = len(set(retrieved_docs[:k]) & set(true_docs))
        total_relevant = len(set(true_docs))
        recall = hit_count / total_relevant if total_relevant > 0 else 0
        recall_scores.append(recall)
    return 100.0* np.mean(recall_scores)

#MAP@K (K=1,2,3) #retrieval metric
def map_at_k(retrieved, ground_truth, k=1):
    average_precisions = []
    for retrieved_docs, true_docs in zip(retrieved, ground_truth):
        y_true = [1 if doc_id in true_docs else 0 for doc_id in retrieved_docs[:k]]
        if len(true_docs) == 0:  # Handle case with no relevant documents
            continue
        average_precisions.append(average_precision_score(y_true, [1] * len(y_true)))
    return 100.0 * np.mean(average_precisions) if len(average_precisions) else 0


#Answer - semantic similarity
def compute_answer_BERT_similarity(question, answers, predicted_answers):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Storage for cosine similarities
    similarities = []

    for i in range(len(predicted_answers)):
        # Format the ground truth and predicted strings
        true_ans, pred_ans = str(answers[i]).strip().strip('\n'), str(predicted_answers[i]).strip().strip('\n')
        #print(question[i], true_ans, pred_ans)
        ground_truth = f"Q: {question[i]}\nA: {true_ans}"
        predicted = f"Q: {question[i]}\nA: {pred_ans}"

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

#Answer - exact match
def answer_exact_match(true_answers, pred_answers):
    match_count = sum(1 for pred, true in zip(pred_answers, true_answers) if str(pred).strip().lower() == str(true).strip().lower())
    return 100.0* match_count / len(true_answers) if len(true_answers) else 0

#Answer recall - No. of overlapping words/ no.of words in gt answer
def answer_recall(true_answers, pred_answers):
    recalls = []
    for pred, true in zip(pred_answers, true_answers):
        pred_words = Counter(str(pred).lower().split())
        true_words = Counter(str(true).lower().split())
        common_words = pred_words & true_words
        total_true_words = sum(true_words.values())
        recall = sum(common_words.values()) / total_true_words if total_true_words > 0 else 0
        recalls.append(recall)
    return 100.0* np.mean(recalls)

def listify(str_list_full):
    full = []
    for str_list in str_list_full:
        ls = str_list.strip()[1:-1].split(",")
        full.append([p.strip() for p in ls])
    return full

if __name__ == '__main__':
    results_path = 'Colbert_llama_pred_answers@1.csv'
    df = pd.read_csv(results_path)
    questions = df['question'].tolist()
    retrieved_docs = listify(df['retrieved_docs'].tolist())
    gt_docs = listify(df['gt_docs'].tolist())
    pred_answer = df['llm_answer'].tolist()
    gt_answer = df['gt_answer'].tolist()

    print(f"Metrics for results : {results_path}")
    print(f"Retrieval Metrics")
    recall_1 = recall_at_k(retrieved_docs, gt_docs, k=1)
    recall_2 = recall_at_k(retrieved_docs, gt_docs, k=2)
    map_1 = map_at_k(retrieved_docs, gt_docs, k=1)
    map_2 = map_at_k(retrieved_docs, gt_docs, k=2)

    print(f"Recall@1:{recall_1}\nRecall@2:{recall_2}\nMAP@1:{map_1}\nMAP@2:{map_2}")

    print("Answer match metrics")
    EM_prec = answer_exact_match(gt_answer, pred_answer)
    ans_recall = answer_recall(gt_answer, pred_answer)
    ans_sem_prec = compute_answer_BERT_similarity(questions, gt_answer, pred_answer)
    print(f"Exact match prec:{EM_prec}\nAnswer recall:{ans_recall}\nSemantic similarity:{ans_sem_prec}") 

