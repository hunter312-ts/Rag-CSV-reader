import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import pandas as pd

from langchain.document_loaders import csv_loader
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate ,PromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(page_title=" Rag Q&A with CSV and chat history",layout="wide",initial_sidebar_state="expanded")
st.title(" rag Q&A with csv and chat history")
st.sidebar.header("Configuration")
st.sidebar.write(
    
    "- Enter your API KEY \n"
    "- Upload Your Csv File \n"
    "- Ask question and See the chat history."
)
load_dotenv()
api_key=st.sidebar.text_input("GROQ API KEY",type="password")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
from langchain.embeddings import HuggingFaceEmbeddings
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not api_key:
    st.warning("please enter your api key to continue---")
    st.stop()

# Initialize the llm 
llm=ChatGroq(api_key=api_key,model="gemma2-9b-it")

#Upload csv
uploaded_csv=st.file_uploader("Upload CSV Files",type=["csv"])
if uploaded_csv:
    with st.spinner("Loading and processing csv files-----"):
        # read the csv file
        df=pd.read_csv(uploaded_csv)
        # convert each row into langchain document
        all_doc=[]
        for i , row in df.iterrows():
            content="\n".join([f"{row}:{row[col]}"  for col in df.columns])
            all_doc.append(Document(page_content=content, metadata={"row":i}))

        #Splitting the documents into chunks
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        split=text_splitter.split_documents(all_doc)
        # Cashe Vector Store
        @st.cache_resource(show_spinner=False)
        def get_vactor(_split):
            return Chroma.from_documents(
                _split,
                embedding,
                persist_directory="./csv_chroma_index"
            )
        vectorstores=get_vactor(split)
        retriever=vectorstores.as_retriever()
        
        #Prompt Template
        contextualize_template=ChatPromptTemplate.from_messages([
            ("system","use the chat history and user question to improve retriever."),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ])

        history_aware_retriever=create_history_aware_retriever(
            llm,retriever,contextualize_template
        )
        qa_prompt=PromptTemplate.from_template(
            "You are helpful assistant. Use the context below to answer the question."
            "If you don’t know the answer, say you don’t know.\n\n"
            "Context:\n{context}\n\nQuestion:\n{input}"
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_cahin=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        # chat history session
        if "csv_chat_history" not in st.session_state:
            st.session_state.csv_chat_history={}
        def get_history(session_id:str):
            if session_id not in st.session_state.csv_chat_history:
                st.session_state.csv_chat_history[session_id]=ChatMessageHistory()
            return st.session_state.csv_chat_history[session_id]
        
        converational_rag=RunnableWithMessageHistory(
            rag_cahin,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # chat interface
        session_id=st.text_input("Session ID",value="default_csv_session")
        user_question=st.chat_input("Ask a question about your csv data...")

        if user_question:
            history=get_history(session_id)
            result=converational_rag.invoke(
                {"input":user_question},
                config={"configurable":{"session_id":session_id}}
            )
            answer=result["answer"]

            st.chat_message("user").write(user_question)
            st.chat_message("assistant").write(answer)

            with st.expander("Full Chat History"):
                for msg  in history.messages:
                    role = getattr(msg, "role", msg.type)
                    content = msg.content
                    st.write(f"**{role.title()}:** {content}")
        else:
            st.info("Please upload your csv file to begin.")


        

