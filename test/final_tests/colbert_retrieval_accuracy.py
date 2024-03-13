#prepare rag index on the prefinal_documents folder
#for every question, do the k=2 rag search
# store the predicted filenames in a list
# compare it with ground truth filenames and calculate retrieval accuracy, f1 score and recall
import os
from ragatouille import RAGPretrainedModel
import time
#from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
import csv
from retrieval_metrics import *

def prepare_documents(directory_path):
    documents = []
    count = 0
    filenames = []
    for i, filename in enumerate(sorted(os.listdir(directory_path)), start=1):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                #TODO modify ID from integer to filename
                #documents.append({"id": i, "text": text})
                documents.append(text)
                filenames.append(filename)
    return documents, filenames


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row2 = {k.strip():v.strip() for k,v in row.items()}
            data.append(row2)
    return data

file_path = '../../faculty_data_subset/manual_questions.csv'
question_data = read_csv_file(file_path)

# Usage example:
directory_path = '../../faculty_data_subset/prefinal_documents'
documents, filenames = prepare_documents(directory_path)
print(len(documents))

st = time.time()
#RAG = RAGPretrainedModel.from_pretrained('../../../models/colbertv2.0')
#index_path = RAG.index(index_name="my_faculty_small_index", collection=documents, document_ids = filenames)
index_path = '.ragatouille/colbert/indexes/my_faculty_small_index'
RAG = RAGPretrainedModel.from_index(index_path)


questions_list = []
documents_list = []
for item in question_data:
    print(item)
    questions_list.append(item['question'])
    true_docs = []
    if ',' in item['document']:
        true_docs = item['document'].split(',')
    elif item['document'].endswith('.txt'):
        true_docs = [item['document']]
    else:
        true_docs = [item['document']+'.txt']
    documents_list.append(true_docs)
results = RAG.search(questions_list, k=2)

queries = []
for i in range(len(documents_list)):
    print(documents_list[i], results[i][0]['document_id'], results[i][1]['document_id'])
    query = {'retrieved_documents': [results[i][0]['document_id'], results[i][1]['document_id']],
                'relevant_documents': documents_list[i]}
    queries.append(query)

#TODO calculation
metrics = compute_metrics_overall(queries)
print(metrics)



