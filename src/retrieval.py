import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import uuid
import sqlite3
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

class retrieval:
    '''
    This class has methods that takes the user input, converts it into vectors, 
    and compare and obtain the similarity score between the stored vectors and user query vectors
    the other method initialize the LLM that will be later used for generation
    '''

    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        db_path=r"D:\mlproject18\Database\vector_db"
        self.model_name=HuggingFaceEmbeddings(model_name=model_name)
        self.db_path=db_path
        self.sqlite_url="sqlite:///D:/mlproject18/Database/chat_history.db"
    
    #Converts the user query into vectors (for testing)
    def manual_conv(self,input:str):
        embedd_texts=self.model_name.embed_query(
            input
        )
        print(embedd_texts[:3])

        return input

    #Searches the vector database for similarity score between user query and existing vectors
    def search(self,input:str,k:int=5,collection_name:str="clinical_database"):
        try:
            vector_db=Chroma(
                collection_name=collection_name,
                embedding_function=self.model_name,
                persist_directory=self.db_path
            )

            print("Similarity scores calculations started....")
            similarity=vector_db.similarity_search(query=input,k=k)

            context_blocks=[]
            for doc in similarity:
                file_name=doc.metadata.get("file_name", "Unknown File")
                page=doc.metadata.get("page", "N/A")
                block=f"[Source: {file_name} | Page: {page}]\n{doc.page_content}"
                context_blocks.append(block)

            context="\n\n---\n\n".join(context_blocks)
            # print(context[:500])
            return context
        
        except Exception as e:
            print(e)
            return 

    def session_history(self,session_id:str):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=self.sqlite_url
        )

    #Initializes the LLM for generation (for testing)
    def model_initialize(self,input_text:str,session_id:str):
        load_dotenv()
        if "GROQ_API_KEY" not in os.environ:
            print("API key not found")
            exit 

        try:
            llm=ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0,
                max_tokens=None,
                reasoning_format='parsed',
                timeout=None,
                max_retries=3
            )

            context_data=self.search(input=input_text)

            system_prompt = """
You are an expert Clinical Drug Information RAG Assistant. Your objective is to answer healthcare and medication queries accurately using **only** the grounded contextual data retrieved from official pharmaceutical documents (such as Prescribing Information and Medication Guides).

### 1. GUIDELINES & PROTOCOLS
- **Strict Grounding & Citations:** Every piece of medical information, dosage guideline, or warning you provide must be explicitly backed by the retrieved context. Always cite your sources using the format provided in the context (e.g., `[Source: filename.pdf | Page: X]`).
- **Strict Hallucination Control:** If the answer cannot be found in the retrieved context, state clearly: "That specific medication detail is not present in the current official prescribing documentation provided." Do not guess or extrapolate medical data.
- **Tone & Persona:** Professional, objective, clinical, and precise. Speak with medical and pharmacological accuracy.

### RETRIEVED CONTEXT:
{context}
"""
            formatted_system_prompt=system_prompt.format(
                context=context_data
                if context_data
                else "No relevant context found."
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", formatted_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}")
            ])

            chain=prompt|llm

            wrapped_chain=RunnableWithMessageHistory(
                chain,
                self.session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            ai_msg=wrapped_chain.stream({
                "input": input_text
            }, 
            config={
                "configurable": {
                    "session_id": session_id
                }
            })

            return wrapped_chain
        
        except Exception as e:
            print(e)


if __name__=="__main__":
    print("Model has started....")
    current_session_id=str(uuid.uuid4())
    print(f"Active Session ID: {current_session_id}")
    while(True):
        user=input("Enter your query: ")
        if(user!='exit'):
            print("Wait a moment....")
            retrieve_obj=retrieval()
            # retrieve_obj.manual_conv(user)
            wrapped_chain=retrieve_obj.model_initialize(input_text=user, session_id=current_session_id)
        else:
            print("Thank you!")
            break