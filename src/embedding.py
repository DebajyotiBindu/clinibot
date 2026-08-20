import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.loading import loader

class embedding:
    '''
    This class helps in converting the chunks of the langchain document object
    into vectors which is returned in the form of numpy ndarray
    '''

    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        self.model=HuggingFaceEmbeddings(model_name=model_name)
    
    #Loads the utility classes from the loader class and provides the chunks
    def reload(self):
        path=r"D:\mlproject18\data"
        load_obj=loader(path=path)

        docs=load_obj.loading()
        chunks=load_obj.chunking(docs=docs)

        return chunks
    
    #Converts the chunks obtained into vectors
    def embedd(self)->np.ndarray:
        chunks=self.reload()

        texts=[chunk.page_content for chunk in chunks]
        vectors=self.model.embed_documents(
            texts
        )

        print(vectors[:5])

        return self.model

class VectorStore:
    '''
    This class is responsible for storing the vectors made by the embedding class
    and store it in a directory and to access the vectors from the directory 
    during the time of retrieval
    '''

    def __init__(self,collection_name:str="clinical_database"):
        self.collection_name=collection_name

        path_name=os.path.join('Database','vector_db')
        os.makedirs(path_name,exist_ok=True)
        print("Directory created successfully")

        self.db_path=path_name

    def store(self,model,docs):
        vector_store=Chroma(
            collection_name=self.collection_name,
            embedding_function=model,
            persist_directory=self.db_path
        )

        #Checks if the data inside the database already exists to prevent duplicate data
        if len(vector_store.get()['ids'])==0:
            print("Database Empty,adding elements....")
            vector_store.add_documents(documents=docs)
            print("Vector Database has been created")

        else:
            print("Data already exists in database")
        
        print(f"{self.collection_name} of length {len(vector_store.get()['ids'])} has been stored")

        return

if __name__=="__main__":
    embedd_obj=embedding()
    vector_obj=VectorStore()

    print("Embedding started....")
    chunks=embedd_obj.reload()

    if chunks:
        print(f"Successfully loaded {len(chunks)} chunks.")
        print(f"Sample Metadata Preview -> File: {chunks[0].metadata.get('file_name')} | Page: {chunks[0].metadata.get('page')}")

    model=embedd_obj.embedd()

    print("Vector Database Initialized")
    vector_obj.store(model=model,docs=chunks)