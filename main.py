import langchain
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
embeddings = OpenAIEmbeddings(openai_api_key="Your-API-Key")


def get_conversation_chain(vector_store:FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(temperature=0.5,openai_api_key="Your-API-Key")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain


def clear_data():    
    st.session_state.chat_history = []
    st.session_state.my_text = ""
    


system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        you are an ai assistent.
       
        If you don't know the answer, just say that you don't know. Do not use external knowledge.
        Be polite and helpful. 
        Make sure to reference your sources with quotes of the provided context as citations.
        \nContext: {context} \nAnswer:
        
        
        
        \nQuestion: {question} 
        
        
        """
)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

vector_store = FAISS.load_local("./data_vectorDB", embeddings, allow_dangerous_deserialization=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = vector_store
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
        
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
)

st.title("Hi, I am Chatbot")
st.subheader("Chat with Me")
st.markdown(
    """
    This chatbot was created to answer questions about the BotPenguin AI Chatbot Maker.
    This chatbot is an ai assistent at BotPenguin AI Chatbot Maker which provide information related to theBotPenguin AI Chatbot Maker
    """
)
st.session_state.conversation = get_conversation_chain(
    st.session_state.vector_store, system_message_prompt, human_message_prompt
)
human_style = "background-color: #f9f9f9; border-radius: 10px; padding: 10px;"
chatbot_style = "background-color: #e6f7ff; border-radius: 10px; padding: 10px;"




for i, message in enumerate(st.session_state.chat_history):
    if i % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)
    else:
                with st.chat_message("assistant"):                    
                    st.write(message.content)




user_question  = st.chat_input("Ask your question", key = "text")
if user_question:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]
        user_question =""
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):                    
                    st.write(message.content)
            
        st.button("clear" , on_click=clear_data)
