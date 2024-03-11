#TODO - Priority
1. Langchain RAG Stack
- LLama 13B (very slow langchain because inference is per question)
- Base retriever + LLama13B is giving I don't know

2. Finetuning Langchain RAG Stack
- Colbert
- Batch colbert

3. Using Larger models with in-context learning (eg. llama-instruction-tuned); <Retriever>

#Then use some transformers good LLM to just see how good the inference is

TODO -All metrics step by step
1. Prepare neat test data for faculty section (Keep existing as is)
- Download research papers
- Remove affiliated, adjunct 
- Update BIO to LeiLi
- Prepare document_list
- Prepare default questions from each document (existing script)
- For research papers, prepare random questions (TODO THIS WELL, Later)
- For each question, write answer as well as the document ID filename from which it was taken

2. First using colbert, index all the documents - To check what happens when we split into paragraphs
3. Retrieve documents, and check retrieval accuracy, F1 score and recall
4. Integrate with LLM (LLama-chat with a RAG prompt)
5. Measure answer comparision metrics on self-data
6. Generate answers for the ANLP Course Test data
