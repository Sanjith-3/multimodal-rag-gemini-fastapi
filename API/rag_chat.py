from fastapi import FastAPI, Form, UploadFile, File, HTTPException
import base64
from typing import List, Optional
from pydantic import BaseModel
import logging
import os
import shutil
import google.generativeai as genai
from uuid import UUID
from pathlib import Path
import urllib.request
import zipfile
import ssl
import nltk
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import chromadb
from pdf2image import convert_from_path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# Initialize global variables
UPLOAD_DIR = r"C:\College\Internship\Yaane\OCR\Multimodal RAG - Task 2\API\uploads"
FIGURES_DIR = r"C:\College\Internship\Yaane\OCR\Multimodal RAG - Task 2\API\figures"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Initialize NLTK setup code
nltk_data_dir = "/content/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_dir

resources_to_download = [
    'punkt',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'punkt_tab',
    'averaged_perceptron_tagger_eng'
]

for resource in resources_to_download:
    try:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        print(f"Error downloading {resource}: {str(e)}")

nltk.data.path = [nltk_data_dir]

# Initialize models
model_vision = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, max_tokens=1024)

class TextProcessor:
    def __init__(self, max_daily_calls=12):
        self.max_daily_calls = max_daily_calls
        self.api_calls = 0
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            max_tokens=1024
        )

    def generate_summary(self, content):
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well-optimized for retrieval. \
        Content to summarize: {element}"""
        
        prompt = PromptTemplate.from_template(prompt_text)
        
        try:
            chain = {"element": lambda x: x} | prompt | self.model | StrOutputParser()
            return chain.invoke(content)
        except Exception as e:
            print(f"Error generating summary: {e}")
            return None

    def check_api_limit(self):
        current_date = datetime.now().strftime('%Y-%m-%d')
        if current_date != self.current_date:
            self.current_date = current_date
            self.api_calls = 0
        return self.api_calls < self.max_daily_calls

    def process_content(self, docs, tables, max_docs=19):
        if not self.model:
            self.setup_model()

        docs = docs[:max_docs] if docs else []
        text_summaries = []
        table_summaries = []
        
        for idx, doc in enumerate(docs):
            if not self.check_api_limit():
                print("Daily API limit reached")
                break
                
            summary = self.generate_summary(doc.page_content)
            if summary:
                text_summaries.append(summary)
                self.api_calls += 1
            else:
                text_summaries.append(doc.page_content)

        for idx, table in enumerate(tables):
            if not self.check_api_limit():
                print("Daily API limit reached")
                break
                
            summary = self.generate_summary(table.page_content)
            if summary:
                table_summaries.append(summary)
                self.api_calls += 1
            else:
                table_summaries.append(table.page_content)

        return text_summaries, table_summaries

class ImageProcessor:
    def __init__(self, model_vision, figures_dir=FIGURES_DIR, max_daily_calls=12):
        self.model_vision = model_vision
        self.figures_dir = Path(figures_dir)
        self.max_daily_calls = max_daily_calls
        self.api_calls = 0
        self.current_date = datetime.now().strftime('%Y-%m-%d')

    def encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def image_summarize(self, img_base64):
        prompt = """You are an assistant tasked with summarizing images for retrieval.
        These summaries will be embedded and used to retrieve the raw image.
        Give a concise summary of the image that is well optimized for retrieval."""

        try:
            msg = self.model_vision.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                        ]
                    )
                ]
            )
            return msg.content
        except Exception as e:
            print(f"Error summarizing image: {e}")
            return None

    def check_api_limit(self):
        current_date = datetime.now().strftime('%Y-%m-%d')
        if current_date != self.current_date:
            self.current_date = current_date
            self.api_calls = 0
        return self.api_calls < self.max_daily_calls

    def process_images(self):
        base64_images = []
        summaries = []
        paths = []

        all_images = [
            f for f in sorted(self.figures_dir.glob('*'))
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ]

        for img_path in all_images:
            if not self.check_api_limit():
                print("Daily API limit reached")
                break

            try:
                base64_image = self.encode_image(img_path)
                if not base64_image:
                    continue

                summary = self.image_summarize(base64_image)
                if not summary:
                    continue

                base64_images.append(base64_image)
                summaries.append(summary)
                paths.append(str(img_path))

                self.api_calls += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        return base64_images, summaries, paths

class ChromaMultiModalRetriever:
    def __init__(self, persist_directory="./chromadb_storage"):
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.Client(
            settings=chromadb.Settings(
                persist_directory=persist_directory,
                is_persistent=True
            )
        )
        
        try:
            self.client.delete_collection("text_docs")
            self.client.delete_collection("image_docs")
        except:
            pass
        
        self.text_collection = self.client.create_collection(
            "text_docs",
            metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.create_collection(
            "image_docs", 
            metadata={"hnsw:space": "cosine"}
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def add_documents(self, text_docs=None, text_summaries=None, 
                     table_docs=None, table_summaries=None, 
                     image_base64s=None, image_summaries=None):
        
        if text_docs and text_summaries:
            for idx, (doc, summary) in enumerate(zip(text_docs, text_summaries)):
                doc_id = f"text_{idx}"
                embedding = self.embeddings.embed_query(summary)
                self.text_collection.add(
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[{
                        "type": "text",
                        "raw_content": doc.page_content,
                        "doc_id": doc_id,
                        "source_type": "text"
                    }],
                    ids=[doc_id]
                )
        
        if table_docs and table_summaries:
            for idx, (doc, summary) in enumerate(zip(table_docs, table_summaries)):
                doc_id = f"table_{idx}"
                embedding = self.embeddings.embed_query(summary)
                self.text_collection.add(
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[{
                        "type": "table",
                        "raw_content": doc.page_content,
                        "doc_id": doc_id,
                        "source_type": "table"
                    }],
                    ids=[doc_id]
                )
        
        if image_base64s and image_summaries:
            for idx, (img, summary) in enumerate(zip(image_base64s, image_summaries)):
                doc_id = f"image_{idx}"
                embedding = self.embeddings.embed_query(summary)
                self.image_collection.add(
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[{
                        "type": "image",
                        "raw_content": img,
                        "doc_id": doc_id,
                        "source_type": "image"
                    }],
                    ids=[doc_id]
                )

    def retrieve(self, query: str, k: int = 4):
        query_embedding = self.embeddings.embed_query(query)
        
        text_results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=k//2,
            include=["documents", "metadatas", "distances"]
        )
        
        image_results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=k//2,
            include=["documents", "metadatas", "distances"]
        )
        
        combined_results = []
        
        for doc, metadata in zip(text_results['documents'][0], text_results['metadatas'][0]):
            combined_results.append(
                Document(
                    page_content=doc,
                    metadata={
                        "type": metadata["type"],
                        "raw_content": metadata["raw_content"],
                        "doc_id": metadata["doc_id"],
                        "source_type": metadata["source_type"]
                    }
                )
            )
        
        for doc, metadata in zip(image_results['documents'][0], image_results['metadatas'][0]):
            combined_results.append(
                Document(
                    page_content=doc,
                    metadata={
                        "type": metadata["type"],
                        "raw_content": metadata["raw_content"],
                        "doc_id": metadata["doc_id"],
                        "source_type": metadata["source_type"]
                    }
                )
            )
        
        return combined_results

def split_image_text_types(docs: List[Document]) -> Dict:
    result = {
        "texts": [],
        "images": [],
        "sources": []
    }
    
    for doc in docs:
        source_info = {
            "doc_id": doc.metadata["doc_id"],
            "type": doc.metadata["type"],
            "source_type": doc.metadata["source_type"],
            "content": doc.metadata["raw_content"],
            "summary": doc.page_content
        }
        
        if doc.metadata["type"] in ["text", "table"]:
            result["texts"].append(f"""
            Type: {doc.metadata['type']}
            Doc ID: {doc.metadata['doc_id']}
            Summary: {doc.page_content}
            Content: {doc.metadata['raw_content']}
            """)
            result["sources"].append(source_info)
        elif doc.metadata["type"] == "image":
            result["images"].append(doc.metadata["raw_content"])
            result["sources"].append(source_info)
    
    return result

def multimodal_prompt_function(data_dict: Dict) -> List[HumanMessage]:
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            }
            messages.append(image_message)
    
    text_message = {
        "type": "text",
        "text": f"""You are an analyst tasked with understanding detailed information and trends from text documents,
            data tables, and charts/graphs in images. Use the provided context to answer the user's question.
            Only use information from the provided context and do not make up additional details.
            
            User question: {data_dict['question']}
            
            Context documents:
            {formatted_texts}
            
            Answer:"""
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

class MultiModalRAG:
    def __init__(self, retriever: ChromaMultiModalRetriever):
        self.retriever = retriever
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            convert_system_message_to_human=True
        )
        
        self.retrieve_docs = (
            itemgetter("input") |
            RunnableLambda(self.retriever.retrieve) |
            RunnableLambda(split_image_text_types)
        )
        
        self.last_query_context = None
        
        self.rag_chain = (
            {
                "context": itemgetter("context"),
                "question": itemgetter("input")
            } |
            RunnableLambda(multimodal_prompt_function) |
            self.chat_model |
            RunnableLambda(lambda x: x.content)
        )
        
        self.chain_with_sources = (
            RunnablePassthrough.assign(context=self.retrieve_docs)
            .assign(answer=self.rag_chain)
        )

    def query(self, question: str, return_sources: bool = False) -> Dict:
        response = self.chain_with_sources.invoke({"input": question})
        
        self.last_query_context = {
            'query': question,
            'generated_response': response['answer'],
            'retrieved_contexts': response['context']['texts'],
            'retrieved_images': response['context']['images'],
            'sources': response['context']['sources']
        }
        
        if return_sources:
            return {
                "answer": response["answer"],
                "sources": response["context"]["sources"],
                "images": response["context"]["images"]
            }
        return {"answer": response["answer"]}

# Initialize global RAG system
retriever = ChromaMultiModalRetriever()
mm_rag = MultiModalRAG(retriever)

# FastAPI Models
class QueryInput(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class DocumentInfo(BaseModel):
    filename: str

class Source(BaseModel):
    doc_id: str
    type: str  # "text", "table", or "image"
    content: str
    summary: str

class MultiModalRAGResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    images: List[str] = []  # Keep the base64 images

# FastAPI Application
app = FastAPI()

def process_document(file_path: str):
    """Process uploaded document and update the RAG system"""
    loader = UnstructuredPDFLoader(
        file_path=file_path,
        strategy='hi_res',
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=4000,
        combine_text_under_n_chars=2000,
        mode='elements',
        image_output_dir_path=FIGURES_DIR,
        poppler_path=r"C:\Users\ASUS\Downloads\poppler-24.08.0\Library\bin",
        tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
     )

    # Load and process document
    data = loader.load()
    
    # Separate documents and tables
    docs = []
    tables = []
    for doc in data:
        if 'text_as_html' in doc.metadata and 'table' in doc.metadata['text_as_html'].lower():
            tables.append(doc)
        else:
            docs.append(doc)

    # Process text and tables
    text_processor = TextProcessor()
    text_summaries, table_summaries = text_processor.process_content(docs, tables)

    # Process images
    image_processor = ImageProcessor(model_vision)
    base64_images, image_summaries, image_paths = image_processor.process_images()

    # Update retriever with new content
    retriever.add_documents(
        text_docs=docs,
        text_summaries=text_summaries,
        table_docs=tables,
        table_summaries=table_summaries,
        image_base64s=base64_images,
        image_summaries=image_summaries
    )

@app.post("/upload-doc")
async def upload_and_return_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process document and update RAG system
        process_document(file_path)
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "message": "File uploaded and processed successfully!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

def get_rag_chain(query: str) -> str:
    """Execute RAG query and return response"""
    try:
        result = mm_rag.query(query)
        return result["answer"]
    except Exception as e:
        logging.error(f"Error in RAG chain: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")
    
@app.post("/chat", response_model=MultiModalRAGResponse)
async def chat(query_input: QueryInput):
    logging.info(f"User Query: {query_input.question}")
    
    if not query_input.question:
        logging.error("Query should not be None")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    result = mm_rag.query(query_input.question, return_sources=True)
    
    # Convert sources to Source objects
    sources = [
        Source(
            doc_id=source.get("doc_id", "unknown"),
            type=source["type"],
            content=source["content"],
            summary=source["summary"]
        )
        for source in result['sources']
    ]
    
    logging.info(f"Query: {query_input.question}, AI Response: {result['answer']}")
    
    return MultiModalRAGResponse(
        answer=result['answer'],
        sources=sources,
        images=result['images']
    )