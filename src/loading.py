import os
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class loader:
    def __init__(self,path):
        self.path=path 
    
    #Loads the pdf into a langchain document object
    def loading(self):
        all_files=[]
        for filename in os.listdir(self.path):
            if filename.endswith((".pdf", ".xml")):
                file_path = os.path.join(self.path, filename)
                all_files.append(file_path)

        loader=DoclingLoader(
            file_path=all_files,
            export_type=ExportType.DOC_CHUNKS
        )
        docs = loader.load()

        return docs 

    #Makes chunks out of the document object returned by loading function
    def chunking(self,docs,chunk_size=1000,overlap=200):
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        texts=text_splitter.split_documents(docs)
        for text in texts:
            source_path=text.metadata.get("source", "")
            if source_path:
                text.metadata["file_name"]=os.path.basename(source_path)
            else:
                text.metadata["file_name"]="N/A"

            dl_meta=text.metadata.get("dl_meta", {})
            doc_items=dl_meta.get("doc_items", [])
            page_no="N/A"
            if doc_items and isinstance(doc_items, list):
                origins=doc_items[0].get("prov", [])
                if origins and isinstance(origins, list):
                    page_no=origins[0].get("page_no", "N/A")
            
            text.metadata["page"]=page_no
            if "dl_meta" in text.metadata:
                del text.metadata["dl_meta"]

        return texts

if __name__=="__main__":
    path=r"D:\mlproject18\data"
    load_obj=loader(path)

    docs=load_obj.loading()
    texts=load_obj.chunking(docs)
    if texts:
        print("Metadata Preview:")
        print(texts[0].metadata)