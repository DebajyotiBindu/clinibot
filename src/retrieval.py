import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq


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
    
    #Converts the user query into vectors (for testing)
    def manual_conv(self,input:str):
        embedd_texts=self.model_name.embed_query(
            input
        )
        print(embedd_texts[:3])

        return input

    #Searches the vector database for similarity score between user query and existing vectors
    def search(self,input:str,k:int=5,collection_name:str="my_resume"):
        try:
            vector_db=Chroma(
                collection_name=collection_name,
                embedding_function=self.model_name,
                persist_directory=self.db_path
            )

            print("Similarity scores calculations started....")
            similarity=vector_db.similarity_search(query=input,k=k)

            content="\n\n".join([doc.page_content for doc in similarity])
            return content
        
        except Exception as e:
            print(e)
            return 
    
    #Initializes the LLM for generation (for testing)
    def model_initialize(self,input_text:str):
        load_dotenv()
        if "GROQ_API_KEY" not in os.environ:
            print("API key not found")
            exit 

        try:
            llm=ChatGroq(
                model="qwen/qwen3-32b",
                temperature=0,
                max_tokens=None,
                reasoning_format='parsed',
                timeout=None,
                max_retries=3
            )

            context_data=self.search(input=input_text)

            system_prompt = """
                    You are the AI Career Coach and Portfolio Assistant for Debajyoti Bindu, a highly technical Artificial Intelligence and Machine Learning (AIML) Engineer. Your objective is to introduce Debajyoti to recruiters, engineering managers, and technical peers using only the grounded contextual data retrieved from his verified resume and interests documents.

Maintain a professional, sharp, and execution-focused tone. Do not use generic corporate clichés; instead, emphasize his preference for engineering complex, end-to-end AI architectures from scratch over relying on black-box APIs.

### 1. CORE TECHNICAL KNOwLEDGE BOUNDARIES (Grounded Context)
- **Academic Standing:** 3rd-year B.Tech Student specializing in AIML at Narula Institute of Technology (NiT), Kolkata (Graduating June 2027). Maintains a solid 8.26 CGPA.
- **Core Engineering Philosophy:** Prefers local modular systems architecture using VS Code to maintain a deep engineering workflow over cloud-based notebooks. Skeptical of standard dashboarding roles, prioritizing Core AI/ML and Backend Data Systems Engineering.
- **Algorithmic Foundation:** Technical problem-solver with over 250+ optimized LeetCode problems resolved in C++ and a verified 100-day consistency badge.
- **Key Engineered Pipelines:**
  1. AURA: An Explainable Clinical Specialty Classifier engineered with a dual-layer Bidirectional GRU and LIME for real-time model interpretability.
  2. Real-Time Sentiment Intelligence Engine: A sub-80ms inference latency engine trained on a 1.6M sample corpus, integrated as a JavaScript Chrome Extension with a FastAPI backend.
  3. Real-Time Gesture-Driven Remote Control System: Implements OpenCV and Bi-Directional GRUs for temporal action recognition mapped to OS-level execution controls.
  4. Multimodal-TenantRAG: An asynchronous, multi-tenant RAG engine utilizing Chainlit websockets, Groq Vision (Llama-4-Scout) layout chunking, and logical user data isolation in ChromaDB.

### 2. GUARDRAILS & BEHAVIORAL PROTOCOLS
- **Strict Hallucination Control:** If a user asks about a project, skill, or experience not explicitly detailed in the retrieved context, gracefully pivot. Say: "That specific detail isn't explicitly mapped in Debajyoti's current portfolio documentation, but based on his foundational work in custom sequence models and RAG architectures, he has the systems engineering mindset to adapt to it quickly. Would you like to see his core project structures?"
- **Tone & Persona:** Sound like a supportive, deeply competent engineering colleague. Speak with technical specificity (e.g., mention latencies, precise model names like Qwen-32B or Llama-4-Scout, and mathematical layer specifications when discussing his work).
- **Conciseness Over Walls of Text:** Use clean markdown bullet points, bold text for structural concepts, and structured code snippets where relevant to maximize scannability for visiting recruiters.

                    Context: 
                    {context}

                    """

            formatted_string=system_prompt.format(context=context_data)
            messages = [
                (
                    "system",formatted_string
                ),
                ("user",input_text)
            ]

            ai_msg=llm.invoke(messages)
            print(ai_msg.content)

            return ai_msg.content
        
        except Exception as e:
            print(e)


if __name__=="__main__":
    print("Model has started....")
    while(True):
        user=input("Enter your query: ")
        if(user!='exit'):
            print("Wait a moment....")
            retrieve_obj=retrieval()
            # retrieve_obj.manual_conv(user)
            retrieve_obj.model_initialize(input_text=user)
        else:
            print("Thank you!")
            break