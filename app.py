import fitz  # PyMuPDF
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated import for the new package
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# --- CORE RAG FUNCTIONS ---

def get_pdf_text(pdf_files: List[UploadFile]):
    text = ""
    for pdf in pdf_files:
        pdf_stream = pdf.file.read()
        pdf_reader = fitz.open(stream=pdf_stream, filetype="pdf")
        for page in pdf_reader:
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Using the updated class
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def create_rag_chain(vectorstore, groq_api_key):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert financial analyst. Answer the user's question based on the provided document context.
    If the information is not in the context, say that you cannot find the answer in the provided documents.
    Provide a detailed, analytical answer synthesizing information from the context.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- FASTAPI APP ---

app = FastAPI(title="Financial Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None

class QuestionRequest(BaseModel):
    question: str

@app.get("/", summary="Root endpoint to check if the API is running")
async def root():
    return {"message": "Financial Analyst API is running."}

@app.post("/process-pdfs/", summary="Upload and process PDF files")
async def process_pdfs_endpoint(files: List[UploadFile] = File(...)):
    global rag_chain
    
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in the environment.")

    try:
        raw_text = get_pdf_text(files)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        rag_chain = create_rag_chain(vectorstore, groq_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDFs: {e}")
    
    return {"status": "success", "message": f"{len(files)} PDF(s) processed. The API is ready."}

@app.post("/ask/", summary="Ask a question about the processed PDFs")
async def ask_question_endpoint(request: QuestionRequest):
    global rag_chain

    if rag_chain is None:
        raise HTTPException(status_code=400, detail="PDFs have not been processed. Please call the /process-pdfs/ endpoint first.")

    try:
        response = rag_chain.invoke({"input": request.question})
        answer = response.get("answer", "Could not generate an answer from the provided context.")
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
