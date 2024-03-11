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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

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

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
st = time.time()
answer = llm.invoke(prompt)
print(answer)
en = time.time()
print(f"Time for one prompt : {en-st}")

prompt_list = ["What is the capital of france?", "Who is Ethiopia's first president?", "What is the most popular color of the apple?"]
template = """
Instruction: Answer using 1 sentence only. 
Question: {prompt}
"""

prompt_list = [template.format(prompt=prompt) for prompt in prompt_list]
print(prompt_list)
st = time.time()
answer_list = []
for i in range(len(prompt_list)):
    answer = llm.invoke(prompt_list[i])
    answer_list.append(answer)
print(len(answer_list))
print(answer_list)
en = time.time()
print(f"Time for 3 prompts : {en-st}")


