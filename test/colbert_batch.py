import os
from ragatouille import RAGPretrainedModel

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
                count +=1
                if count == 100:
                    return documents, filenames
    return documents

# Usage example:
directory_path = '../Faculty_data/documents'
documents, filenames = prepare_documents(directory_path)
subset_docs = list(documents)[:100]
print(subset_docs)


# Initialize RAGatouille with a pretrained ColBERT model
RAG = RAGPretrainedModel.from_pretrained("./colbertv2.0")
#RAG.index_documents(documents)
index_path = RAG.index(index_name="my_index", collection=documents, document_ids = filenames)



queries = ["What is the first document about?", "Tell me about the second document."]
queries = ["Who wrote the paper 'Zero-Shot and Few-Shot Stance Detection on Varied Topics via Conditional Generation'?", "Which year was the paper 'Appropriateness is all you need!' published?"]

# Perform batch inference
results = RAG.search(queries, k=2) # top_k is the number of documents to retrieve for each query

# Print the results
for query, hits in zip(queries, results):
    print(f"Query: {query}")
    for hit in hits:
        print(hit)

