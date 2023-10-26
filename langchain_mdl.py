# torch langchain transformers faiss-gpu xformers sentence-transformers python_dotenv atlassian-python-api markdownify

import os
from os.path import join, dirname
from dotenv import load_dotenv

import torch
from torch import cuda, bfloat16


from llama2_mdl import llama2_mdl


from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class durham_langchain:
  def __init__(self):
    self.chat_history = []
    self.web_links = [
                      "https://blogs.nvidia.com/blog/2023/10/18/metropolis-jetson-isaac-robotics-edge-ai-developers/",
                      "https://nvidianews.nvidia.com/news/nvidia-partners-with-foxconn-to-build-factories-and-systemsfor-the-ai-industrial-revolution",
                      "https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/",
                      "https://blogs.nvidia.com/blog/2023/10/12/ai-for-creative-industries-uk-startups/",
                      "https://blogs.nvidia.com/blog/2023/10/10/adobe-max-firefly-creative-cloud-substance-3d/",
                      "https://blogs.nvidia.com/blog/2023/10/05/atlas-meditech-brain-surgery-ai-digital-twins/",
                      "https://blogs.nvidia.com/blog/2023/10/04/ai-jim-fan/",
                      "https://blogs.nvidia.com/blog/2023/09/05/covision-adidas-rtx-ai/",
                      "https://blogs.nvidia.com/blog/2023/08/29/hot-chips-dally-research/"
                     ]

  def init_gpt_mdl(self):
    llama = llama2_mdl()
    llama.init_mdl()
    self.llm = HuggingFacePipeline(pipeline=llama.generate_text)
    #print(llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse."))

  def init_vdb(self):
    webloader = WebBaseLoader(self.web_links)
    documents = webloader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": device}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vectorstore = FAISS.from_documents(all_splits, embeddings)
    self.chain = ConversationalRetrievalChain.from_llm(self.llm, vectorstore.as_retriever(), return_source_documents=True)

  def infer(self, query):
    instruction = " You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer, please think rationally and answer from your own knowledge base."
    result = self.chain({"question": query+instruction, "chat_history": self.chat_history})
    self.chat_history.append((query, result["answer"]))
    return result["answer"]


if __name__ == '__main__':
   mdl = durham_langchain()
   mdl.init_gpt_mdl()
   mdl.init_vdb()
   print(mdl.infer("what type of memory that Grace use?"))
   print(mdl.infer("how faster Grace systems than x86 servers?"))