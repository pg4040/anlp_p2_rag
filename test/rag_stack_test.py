from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from llm_inf import prepare_data
from langchain.docstore.document import Document
#from langchaincore.documents import Document
import glob
import time

def get_all_docs():
    all_files = glob.glob('../Faculty_data/documents/*.txt')
    all_docs =  []
    for fi in all_files:
        with open(fi, 'r') as f:
            txt = f.read()
        if txt == '':
            continue
        doc = Document(page_content=txt.strip(), metadata={"id":fi})
        all_docs.append(doc)
    return all_docs

combined_df = prepare_data()
question = combined_df['Question'].iloc[0]
ref_ans = combined_df['Ref_Answer'].iloc[0]
print(f'Question : {question}\n Ref Answer: {ref_ans}\n')
all_docs = get_all_docs()
print(f'st :{time.time()}')
vectorstore = Chroma.from_documents(documents=all_docs, embedding=GPT4AllEmbeddings())
print(f'et :{time.time()}')
print("Performing similarity search.....")
docs = vectorstore.similarity_search(question)#retriever
print('Found most similar documents:', len(docs))



