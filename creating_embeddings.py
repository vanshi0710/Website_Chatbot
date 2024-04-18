import os 
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm import tqdm
from langchain_community.vectorstores import FAISS


dataset = pd.read_csv('new_data.csv')
raw_text = DataFrameLoader(dataset, page_content_column="body")


text_chunks = raw_text.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, length_function=len
        )
    )


for doc in text_chunks:
        title = doc.metadata["title"]
        # description = doc.metadata["description"]
        content = doc.page_content
        url = doc.metadata["url"]
        final_content = f"TITLE: {title}\BODY: {content}\nURL: {url}"
        # final_content = f"TITLE: {title}\DESCRIPTION: {description}\BODY: {content}\nURL: {url}"
        doc.page_content = final_content
print(len(text_chunks))


os.environ['OPENAI_API_KEY'] = "Your-API-Key"
embeddings = OpenAIEmbeddings()


if not os.path.exists("./data_vectorDB"):
    print("CREATING DB")
    n2 = len(text_chunks)%500
    vectorstore = FAISS.from_documents(
        text_chunks[:n2], embeddings
    )
    
    while(n2<len(text_chunks)):
        n1 = n2
        n2 = n1 + 500
        temp = FAISS.from_documents(
                text_chunks[n1:n2], embeddings
            )
        vectorstore.merge_from(temp)

    vectorstore.save_local("./data_vectorDB")
    
    