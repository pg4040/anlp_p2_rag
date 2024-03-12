from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from llm_inf import prepare_data_faculty
from langchain.docstore.document import Document
#from langchaincore.documents import Document
import glob
import time
from langchain_community.llms import LlamaCpp
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import csv

model_path = '../../models/llama-2-7b-chat.Q4_0.gguf'
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 10  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)


filename = "questions_rag_prompts.csv"

questions_list = []
rag_prompts_list = []

# Open the file in read mode ('r') and create a csv.reader object
with open(filename, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)

    next(reader, None)

    for row in reader:
        rag_prompt = row[1].replace('\\n', '\n')
        rag_prompts_list.append(rag_prompt)
        questions_list.append(row[0]) #later use this to get the reference answer or store the reference answer in the same csv

st = time.time()
answer_list = []
for i in range(len(rag_prompts_list)):
    answer = llm.invoke(rag_prompts_list[i])
    answer_list.append(answer)
    print(f"generating {i}th answer", end="\r")
et = time.time()
print(f"Got the answers for all in {et-st}")
print(len(answer_list))
for i in range(len(answer_list)):
    print(questions_list[i], answer_list[i])
