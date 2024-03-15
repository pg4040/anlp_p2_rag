
#Process - Workflow
#1. Rag index creation - onetime (including split_documents=True), reading from there
#2. Rag search @k = 1
#3. Prompt csv preparation
#4. Rag search @k=2 prompt csv preparation. 
#5. Reaad prompt csv and integrate with batch_llama to get final answers.
import os
from ragatouille import RAGPretrainedModel
import time
#from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
import csv
from retrieval_metrics import *
import pandas as pd
import torch
from batch_llama import *

######### HYPERPARAMETERS ##########

questions_filepath = '../../data/questions/combined_data.csv' #This can be changed to new combined test set path
document_directory = '../../data/documents' #This can also be changed to new combined documents path
index_name = 'combined_index' #This can be changed based on which kind of index we are creating
retrieval_k = 3 #This is no. of top_k documents to retrieve. 
prompt_file_name = f'combined_colbert_prompts@{retrieval_k}.csv' #question, retrieved_documents, prompt, [answer, relevant_documents]
output_path = f'combined_colbert_llama_pred_answers@{retrieval_k}.csv'
#Other hyperparameters like max_document_length, llama_context, split_documents = True
prompt_file_name = f'test_questions@{retrieval_k}.csv'
test_qs_file = f'test_questions.txt'
test_as_file = f'test_answers2@{retrieval_k}.txt'

#########################
def prepare_documents(directory_path):
    documents = []
    count = 0
    filenames = []
    for i, filename in enumerate(sorted(os.listdir(directory_path)), start=1):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    #TODO modify ID from integer to filename
                    #documents.append({"id": i, "text": text})
                    documents.append(text)
                    filenames.append(filename)
        except Exception as e:
            print(f"Could not prepare filepath : {file_path}", e)
            exit() 
    return documents, filenames

def prepare_index():
    documents, filenames = prepare_documents(document_directory)
    RAG = RAGPretrainedModel.from_pretrained('../../../models/colbertv2.0')
    index_path = RAG.index(index_name=index_name, collection=documents, document_ids = filenames, split_documents=True, max_document_length=512)


def write_prompts_file(index_name, questions_list, data=None):
    instruction = 'You are a helpful assistant. Answer the question concisely in 1-10 words using the context.'
    index_path = f'.ragatouille/colbert/indexes/{index_name}'
    RAG = RAGPretrainedModel.from_index(index_path)
    results = RAG.search(questions_list, k=retrieval_k)
    if not data:
        prompt_csv_cols= ['question','retrieved_docs', 'prompt']
        rows = []
        for i in range(len(questions_list)):
            q = questions_list[i]
            content_list = [r['content'] for r in results[i]] 
            docs_list = [r['document_id'] for r in results[i]]
            context = '\n'.join(content_list)
            prompt= f"{instruction}\nQuestion: {q}\nContext: {context}\nAnswer: "
            rows.append([q, docs_list, prompt])
        with open(prompt_file_name, 'w') as f:
            #TODO: writerow - cols, write each row in rows
            csv_writer = csv.writer(f)
            csv_writer.writerow(prompt_csv_cols)
            for row in rows:
                csv_writer.writerow(row)
    
def generate_answers_in_batches(prompts_list, batch_size=6):
    answers = []
    for i in range(0,len(prompts_list), batch_size):
        plist = prompts_list[i:i+batch_size]
        torch.cuda.empty_cache()
        ans = batch_llama(plist)
        answers.extend(ans)
    return answers

def prepare_prompts(questions_list):
    write_prompts_file(index_name, questions_list)

def cleaned_doc_list(document_list):
    cleaned_docs_list = []
    for item in document_list:
        if ',' in item:
            true_docs = item.split(',')
        elif item.endswith('.txt'):
            true_docs = [item]
        else:
            true_docs = [item+'.txt']
        cleaned_docs_list.append(true_docs)
    return cleaned_docs_list

def final_pipeline():
    #TODO: generate new index
    data_df = pd.read_csv(questions_filepath)
    questions_list = data_df['question'].tolist()
    prepare_prompts(questions_list)
    answer_list = data_df['answer'].tolist()
    prompt_df = pd.read_csv(prompt_file_name)
    prompts_list = prompt_df['prompt'].tolist()

    #Split prompts into N batches of batch_size 25; call batch_llama for each batch. 
    pred_answers = generate_answers_in_batches(prompts_list) #is a list
    prompt_df['llm_answer'] = pred_answers
    prompt_df['gt_answer'] = answer_list
    documents_list_updated = cleaned_doc_list(data_df['document'].tolist())
    prompt_df['gt_docs'] = documents_list_updated
    prompt_df.to_csv(output_path, index=False)

    for i in range(5):
        print(prompts_list[i])
        print(pred_answers[i])

def test_pipeline():
    test_qs = []
    with open(test_qs_file, 'r') as f:
        test_qs = f.readlines()
    #prepare_prompts(test_qs)
    prompt_df = pd.read_csv(prompt_file_name)
    prompts_list = prompt_df['prompt'].tolist()

    #Split prompts into N batches of batch_size 25; call batch_llama for each batch. 
    pred_answers = generate_answers_in_batches(prompts_list) #is a list
    with open(test_as_file, 'w') as f:
        f.write('\n'.join(pred_answers))
    
if __name__ == '__main__':
    #prepare_index()
    test_pipeline()
