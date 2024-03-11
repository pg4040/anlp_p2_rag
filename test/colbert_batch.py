import os
from ragatouille import RAGPretrainedModel
import time
from llm_inf import prepare_data_faculty

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
results = RAG.search(questions_list, k=3) # top_k is the number of documents to retrieve for each query

# Print the results
for query, hits in zip(questions_list, results):
    print(f"Query: {query}")
    for hit in hits:
        print(hit)
et = time.time()
print(f'Retrieval time:, {et-st}')


final_prompt_lists = []
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 100  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="../llama-2-13b-chat.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)

rag_prompt = hub.pull("rlm/rag-prompt")
print(rag_prompt.messages)


