import os
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class loader:
    def __init__(self,path):
        self.path=path 
    
    #Loads the pdf into a langchain document object
    def loading(self):
        all_docs=[]
        for filename in os.listdir(self.path):
            if filename.endswith(".pdf"):
                loader=PyPDFLoader(os.path.join(self.path, filename))
                docs=loader.load()
                all_docs.extend(docs)
        return all_docs 

    #Makes chunks out of the document object returned by loading function
    def chunking(self,docs,chunk_size=1000,overlap=200):
        text_splitter=RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ],
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            keep_separator=True 
        )

        texts=text_splitter.split_documents(docs)
        return texts 

if __name__=="__main__":
    path=r"D:\mlproject18\data"
    load_obj=loader(path)

    docs=load_obj.loading()
    texts=load_obj.chunking(docs)
    print(texts[0:4])