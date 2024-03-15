import os
from ragatouille import RAGPretrainedModel
import time
#from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
import csv
import glob
from langchain.docstore.document import Document

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

def get_all_docs(path):
    all_files = glob.glob(path+'/*.txt')
    all_docs =  []
    for fi in all_files:
        with open(fi, 'r') as f:
            txt = f.read()
        if txt == '':
            continue
        doc = Document(page_content=txt.strip(), metadata={"id":fi})
        all_docs.append(doc)
    return all_docs


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row2 = {k.strip():v.strip() for k,v in row.items()}
            data.append(row2)
    return data

file_path = '../../../data/questions/combined_data.csv'
question_data = read_csv_file(file_path)

# Usage example:
directory_path = '../../../data/documents'
#documents, filenames = prepare_documents(directory_path)
documents = get_all_docs(directory_path)

#RAG index creation
#RAG = RAGPretrainedModel.from_pretrained('../../../models/colbertv2.0')
#index_path = RAG.index(index_name="my_faculty_small_index", collection=documents, document_ids = filenames)

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=documents, embedding=GPT4AllEmbeddings(), persist_directory='./chroma_db_all')

#docs = vectorstore.similarity_search(question)




#ChromaDB index creation
#HyDE index creation
