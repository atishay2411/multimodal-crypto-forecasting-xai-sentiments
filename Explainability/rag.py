# from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# from dotenv import load_dotenv
# from typing import List, Optional
# import tempfile
# from bs4 import BeautifulSoup
# import docx  
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# # Load environment variables
# load_dotenv()

# # Configure Azure OpenAI
# azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
# azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# ada_deployment = os.getenv("ADA_DEPLOYMENT_NAME")
# gpt_deployment = os.getenv("GPT_DEPLOYMENT_NAME")

# # Initialize embeddings and LLM
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment=ada_deployment,
#     api_key=azure_openai_key,
#     azure_endpoint=azure_openai_endpoint,
#     api_version="2023-05-15"
# )

# llm = AzureChatOpenAI(
#     azure_deployment=gpt_deployment,
#     api_key=azure_openai_key,
#     azure_endpoint=azure_openai_endpoint,
#     api_version="2023-05-15"
# )

# # Create FastAPI app
# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variable for vector store (Note: Use proper persistence in production)
# vector_store = None
# UPLOAD_DIR = tempfile.mkdtemp()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# # Pydantic models
# class QuestionRequest(BaseModel):
#     question: str
# def load_documents(upload_dir: str):
#     documents = []
#     for filename in os.listdir(upload_dir):
#         filepath = os.path.join(upload_dir, filename)
#         try:
#             if filename.endswith(".txt"):
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     text = f.read()
#                 documents.append(Document(page_content=text, metadata={"source": filepath}))
            
#             elif filename.endswith(".xml"):
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     content = f.read()
#                 soup = BeautifulSoup(content, "lxml-xml")
#                 text = soup.get_text(separator=" ", strip=True)
#                 documents.append(Document(page_content=text, metadata={"source": filepath}))
            
#             elif filename.endswith(".docx"):
#                 doc = docx.Document(filepath)
#                 text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
#                 documents.append(Document(page_content=text, metadata={"source": filepath}))
        
#         except Exception as e:
#             print(f"Error loading {filename}: {str(e)}")
#     return documents

# @app.post("/upload")
# async def upload_files(files: List[UploadFile] = File(...)):
#     for file in files:
#         # Add docx to allowed extensions
#         if not file.filename.endswith((".txt", ".xml", ".docx", ".csv")):
#             raise HTTPException(status_code=400, 
#                 detail="Invalid file format. Only .txt, .xml, and .docx are allowed.")
        
#         filepath = os.path.join(UPLOAD_DIR, file.filename)
#         contents = await file.read()
#         with open(filepath, "wb") as f:
#             f.write(contents)
    
#     return {"message": f"Successfully uploaded {len(files)} files."}

# @app.post("/process")
# async def process_documents():
#     global vector_store
#     documents = load_documents(UPLOAD_DIR)
    
#     if not documents:
#         raise HTTPException(status_code=400, detail="No documents to process. Upload files first.")
    
#     split_docs = text_splitter.split_documents(documents)
#     vector_store = FAISS.from_documents(split_docs, embeddings)
#     return {"message": f"Processed {len(split_docs)} document chunks."}

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     if not vector_store:
#         raise HTTPException(status_code=400, detail="Process documents first.")
    
#     retriever = vector_store.as_retriever(k=4)
    
#     prompt_template = """Answer the question based only on the following context:
#     {context}
    
#     Question: {question}
#     Answer in a clear and concise manner. If you don't know the answer, say 'I don't know'."""
    
#     prompt = ChatPromptTemplate.from_template(prompt_template)
    
#     rag_chain = (
#         {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     answer = rag_chain.invoke(request.question)
#     return {"answer": answer}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
from dotenv import load_dotenv
import shutil
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables
load_dotenv()

# Azure OpenAI credentials
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
ada_deployment = os.getenv("ADA_DEPLOYMENT_NAME")
gpt_deployment = os.getenv("GPT_DEPLOYMENT_NAME")

# Embeddings and LLM
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=ada_deployment,
    api_key=azure_openai_key,
    azure_endpoint=azure_openai_endpoint,
    api_version="2023-05-15"
)

llm = AzureChatOpenAI(
    azure_deployment=gpt_deployment,
    api_key=azure_openai_key,
    azure_endpoint=azure_openai_endpoint,
    api_version="2023-05-15"
)

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
VECTOR_STORE_PATH = "faiss_index"
CSV_FILE_PATH = "Documents/cleaned_bitcoin_sentiment.csv"
vector_store = None
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
def load_documents_from_csv(file_path: str):
    documents = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file {file_path} not found.")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  

    required_columns = ["timestamp", "sentiment", "text"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV file.")

    for idx, row in df.iterrows():
        combined_text = f"{row['timestamp']} | {row['sentiment']} | {row['text']}"
        documents.append(Document(page_content=combined_text, metadata={"row": idx}))

    return documents


def save_vector_store(store, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    store.save_local(path)

def load_vector_store(path):
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    else:
        return None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Load vector store or create it at startup
@app.on_event("startup")
async def startup_event():
    global vector_store
    vector_store = load_vector_store(VECTOR_STORE_PATH)
    if vector_store:
        print("Vector store loaded from disk.")
    else:
        print("No existing vector store found. Loading CSV and creating new vector store...")
        documents = load_documents_from_csv(CSV_FILE_PATH)
        if not documents:
            raise RuntimeError(f"No documents found in {CSV_FILE_PATH}.")
        split_docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        save_vector_store(vector_store, VECTOR_STORE_PATH)
        print(f"Vector store created with {len(split_docs)} document chunks.")

# API Endpoint
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not vector_store:
        raise HTTPException(status_code=400, detail="Vector store not initialized.")
    
    retriever = vector_store.as_retriever(k=4)
    
    prompt_template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Answer in a clear and concise manner. If you don't know the answer, say 'I don't know'."""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(request.question)
    return {"answer": answer}

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
