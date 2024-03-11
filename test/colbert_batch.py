import os
from ragatouille import RAGPretrainedModel
import time
from llm_inf import prepare_data_faculty
#from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp

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
                filenames.append(file_path)
    return documents, filenames

# Usage example:
directory_path = '../Faculty_data/documents'
documents, filenames = prepare_documents(directory_path)
print(len(documents))

st = time.time()
# Initialize RAGatouille with a pretrained ColBERT model
RAG = RAGPretrainedModel.from_pretrained("./colbertv2.0")
#RAG.index_documents(documents)
#index_path = RAG.index(index_name="my_large_index", collection=documents, document_ids = filenames)
index_path = '.ragatouille/colbert/indexes/my_large_index'
RAG = RAGPretrainedModel.from_index(index_path)
et = time.time()
print(f'Faculty index creation time: {et-st}')
#queries = ["Who wrote the paper 'Zero-Shot and Few-Shot Stance Detection on Varied Topics via Conditional Generation'?", "Which year was the paper 'Appropriateness is all you need!' published?"]

combined_df = prepare_data_faculty()
#ref_ans = combined_df['Ref_Answer'].iloc[0]
questions_list = combined_df['Question'].tolist()
print(len(questions_list))

# Perform batch inference
st = time.time()
results = RAG.search(questions_list, k=2) # top_k is the number of documents to retrieve for each query

print(f'Retrieval time:, {et-st}')

#rag_prompt = hub.pull("rlm/rag-prompt")
#print(rag_prompt.messages)

template = """
[INST] <<SYS>>
You are an expert extraction algorithm. Only extract relevant information from the text. If you do not know the answer for the question, given the context, return "I don't know" as the answer.
<<SYS>>
User: {question}
Context: {context}
[/INST]
"""
#Add extra - Question: {question}  if needed
# Initialize the prompt template
prompt_template = ChatPromptTemplate.from_template(template)
rag_prompts = []
# Print the results
for query, hits in zip(questions_list, results):
    content_list = [hit['content'] for hit in hits] 
    context = ""
    for i in range(len(content_list)):
        context += f"Document {i}: {content_list[i]} \n"

    formatted_prompt = template.format(context=context, question=query)
    # Add the formatted prompt to the list
    rag_prompts.append(formatted_prompt)
    #print(f"Llama prompt token length: {len(formatted_prompt.split())}")
print(len(rag_prompts))
with open('rag_prompt_file.txt', 'w') as file:
    for text in rag_prompts:
        file.write(text + '\n')  # Add a newline character to separate texts
print(rag_prompts[0])
