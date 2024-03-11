import os
from ragatouille import RAGPretrainedModel

def prepare_documents(directory_path):
    documents = []
    for i, filename in enumerate(os.listdir(directory_path), start=1):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                #TODO modify ID from integer to filename
                documents.append({"id": i, "text": text})
    return documents

# Usage example:
directory_path = '../Faculty_data/documents'
documents = prepare_documents(directory_path)
subset_docs = list(documents)[:100]
print(subset_docs)


# Initialize RAGatouille with a pretrained ColBERT model
RAG = RAGPretrainedModel.from_pretrained("./colbertv2.0")
RAG.index_documents(documents)

queries = ["What is the first document about?", "Tell me about the second document."]

# Perform batch inference
results = RAG.search(queries, top_k=2) # top_k is the number of documents to retrieve for each query

# Print the results
for query, hits in zip(queries, results):
    print(f"Query: {query}")
    for hit in hits:
        print(f" - Document ID: {hit['id']}, Score: {hit['score']}")

