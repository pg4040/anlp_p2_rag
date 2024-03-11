from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from llm_inf import prepare_data
from langchain.docstore.document import Document
#from langchaincore.documents import Document
import glob
import time
from langchain_community.llms import LlamaCpp
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnablePick

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

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="../llama-2-7b-ft-instruct-es.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)

ans = llm.invoke("Simulate a rap battle between Obama and Trump")
print(ans)

rag_prompt = hub.pull("rlm/rag-prompt")
print(rag_prompt.messages)
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Run
final_ans = chain.invoke({"context": docs, "question": question})
print(final_ans)
