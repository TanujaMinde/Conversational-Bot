import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.llms import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import chainlit as cl
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
from langchain.schema import Document, BaseRetriever
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import fitz

doc = fitz.open('./docs/sample.pdf')
text = ""
for page in doc:
    text+=page.get_text()
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_document_set = text_splitter.split_text(text)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_texts(splitted_document_set, embedding_function, persist_directory='./vectorDB_1')

    
@cl.on_chat_start
def main():
    # Instantiate required classes for the user session
    
    llm = AzureOpenAI(
        openai_api_key=openai.api_key,
        engine="text-davinci-003",
        deployment_name="text-davinci-003",
        temperature=0.5,
        openai_api_version=openai.api_version,
        openai_api_base=openai.api_base

    )

    class DocumentRetrieverExtended(BaseRetriever):
        def get_relevant_documents(self, query: str) -> List[Document]:
            print(db.similarity_search(query))
            return db.similarity_search(query)
        
        def aget_relevant_documents(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query)


    retriever_instance = DocumentRetrieverExtended()

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever_instance,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

    # Store the chain in the user session for reusability
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("chain")
    # Call the chain asynchronously
    res = llm_chain({"question": message.content})
    await cl.Message(content=res["answer"]).send()