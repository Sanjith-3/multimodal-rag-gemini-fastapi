from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict
import base64
from datetime import datetime
import nltk
import urllib.request
import zipfile
import ssl
import nltk
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu
from pdf2image import convert_from_path

# Replace this path with your actual poppler installation path
poppler_path = r"C:\Users\ASUS\Downloads\poppler-24.08.0\Library\bin"  # Adjust version number as needed

# When converting PDF, explicitly specify the poppler path
images = convert_from_path(r'C:\College\Internship\Yaane\OCR\Multimodal RAG - Task 2\Materials\stockreport.pdf', poppler_path=poppler_path)

from langchain_community.document_loaders import UnstructuredPDFLoader
import os

# Add both Poppler and Tesseract to PATH programmatically
os.environ["PATH"] += os.pathsep + r"C:\Users\ASUS\Downloads\poppler-24.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

# Also set TESSERACT_CMD environment variable
os.environ["TESSERACT_CMD"] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# Import RAG-related components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize global variables
UPLOAD_DIR = r"C:\College\Internship\Yaane\OCR\Multimodal RAG - Task 2\API\uploads"
FIGURES_DIR = r"C:\College\Internship\Yaane\OCR\Multimodal RAG - Task 2\API\figures"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)



# Create a specific directory for NLTK data in Colab
nltk_data_dir = "/content/nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set the NLTK_DATA environment variable
os.environ['NLTK_DATA'] = nltk_data_dir

resources_to_download = [
    'punkt',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'punkt_tab',
    'averaged_perceptron_tagger_eng'  # Adding the missing resource
]

for resource in resources_to_download:
    try:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
        print(f"Downloaded {resource} successfully")
    except Exception as e:
        print(f"Error downloading {resource}: {str(e)}")

# Clear any existing paths and add only our custom path
nltk.data.path = [nltk_data_dir]


def download_eng_tagger():
    try:
        # Create directories if they don't exist
        tagger_dir = os.path.join(nltk_data_dir, 'taggers')
        if not os.path.exists(tagger_dir):
            os.makedirs(tagger_dir)

        # Download and extract the tagger
        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip"

        # Handle SSL context
        context = ssl._create_unverified_context()

        print("Downloading English tagger...")
        filename, _ = urllib.request.urlretrieve(url, "tagger.zip", context=context)

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(tagger_dir)

        os.rename(os.path.join(tagger_dir, "averaged_perceptron_tagger"),
                 os.path.join(tagger_dir, "averaged_perceptron_tagger_eng"))

        print("English tagger downloaded and installed successfully")

    except Exception as e:
        print(f"Error downloading English tagger: {str(e)}")
    finally:
        if os.path.exists("tagger.zip"):
            os.remove("tagger.zip")

# Download the English tagger
download_eng_tagger()



# Verify installation
def verify_nltk_resources():
    required_resources = [
        'tokenizers/punkt',
        'tokenizers/punkt_tab/english',
        'taggers/averaged_perceptron_tagger',
        'taggers/averaged_perceptron_tagger_eng',
        'chunkers/maxent_ne_chunker',
        'corpora/words'
    ]

    all_available = True
    for resource in required_resources:
        try:
            nltk.data.find(resource)
            print(f"✓ {resource} is available")
        except LookupError:
            print(f"✗ {resource} is NOT available")
            all_available = False

    return all_available

print("\nVerifying NLTK resources...")
resources_available = verify_nltk_resources()

if resources_available:
    print("\nAll resources are available! You can now try loading your PDF.")
else:
    print("\nSome resources are still missing. Please check the output above.")





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

    def setup_model(self):
        """Initialize Gemini model if not already initialized"""
        if self.model is None:
            self.model = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0,
                max_tokens=1024
            )

    def generate_summary(self, content):
        """Generate summary for a single piece of content"""
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
        """Check if we've hit the API limit for today"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        if current_date != self.current_date:
            self.current_date = current_date
            self.api_calls = 0
        return self.api_calls < self.max_daily_calls

    def process_content(self, docs, tables, max_docs=19):
        self.setup_model()

        # Limit documents if needed
        docs = docs[:max_docs] if docs else []
        text_summaries = []
        table_summaries = []
        for idx, doc in enumerate(docs):
            if not self.check_api_limit():
                print("Daily API limit reached. Stopping processing.")
                break

            print(f"Processing text {idx + 1}/{len(docs)}...")
            summary = self.generate_summary(doc.page_content)

            if summary:
                text_summaries.append(summary)
                self.api_calls += 1
            else:
                text_summaries.append(doc.page_content)

            print(f"Remaining API calls: {self.max_daily_calls - self.api_calls}")

        # Process tables if API calls still available
        for idx, table in enumerate(tables):
            if not self.check_api_limit():
                print("Daily API limit reached. Stopping processing.")
                break

            print(f"Processing table {idx + 1}/{len(tables)}...")
            summary = self.generate_summary(table.page_content)

            if summary:
                table_summaries.append(summary)
                self.api_calls += 1
            else:
                table_summaries.append(table.page_content)

            print(f"Remaining API calls: {self.max_daily_calls - self.api_calls}")

        return text_summaries, table_summaries

      

class ImageProcessor:
    def __init__(self, model_vision, figures_dir=FIGURES_DIR, max_daily_calls=12):
        self.model_vision = model_vision
        self.figures_dir = Path(figures_dir)
        self.max_daily_calls = max_daily_calls
        self.api_calls = 0
        self.current_date = datetime.now().strftime('%Y-%m-%d')

    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def image_summarize(self, img_base64):
        """Generate summary for a single image"""
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
        """Check if we've hit the API limit for today"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        if current_date != self.current_date:
            self.current_date = current_date
            self.api_calls = 0
        return self.api_calls < self.max_daily_calls

    def process_images(self):
        """Process images with rate limiting"""
        base64_images = []
        summaries = []
        paths = []

        # Get list of images
        all_images = [
            f for f in sorted(self.figures_dir.glob('*'))
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ]

        print(f"Found {len(all_images)} images to process")

        for img_path in all_images:
            
            if not self.check_api_limit():
                print("Daily API limit reached. Stopping processing.")
                break

            try:
                print(f"Processing {img_path.name}...")

                # Encode image
                base64_image = self.encode_image(img_path)
                if not base64_image:
                    continue

                # Generate summary
                summary = self.image_summarize(base64_image)
                if not summary:
                    continue

                # Store results
                base64_images.append(base64_image)
                summaries.append(summary)
                paths.append(str(img_path))

                # Update API calls tracking
                self.api_calls += 1

                print(f"Successfully processed {img_path.name}")
                print(f"Remaining API calls for today: {self.max_daily_calls - self.api_calls}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        return base64_images, summaries, paths

class ChromaMultiModalRetriever:
    def __init__(self):
        self.text_db = []
        self.image_db = []
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def add_documents(self, text_docs, text_summaries, table_docs=None, table_summaries=None, 
                     image_base64s=None, image_summaries=None):
        # Add text documents
        for doc, summary in zip(text_docs, text_summaries):
            self.text_db.append(Document(
                page_content=summary,
                metadata={
                    "type": "text",
                    "raw_content": doc.page_content
                }
            ))
        
        # Add table documents
        if table_docs and table_summaries:
            for doc, summary in zip(table_docs, table_summaries):
                self.text_db.append(Document(
                    page_content=summary,
                    metadata={
                        "type": "table",
                        "raw_content": doc.page_content
                    }
                ))
        
        # Add image documents
        if image_base64s and image_summaries:
            for img, summary in zip(image_base64s, image_summaries):
                self.image_db.append(Document(
                    page_content=summary,
                    metadata={
                        "type": "image",
                        "raw_content": img
                    }
                ))

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        # Simple similarity search - in production, use proper vector DB
        results = []
        
        # Add text/table results
        if self.text_db:
            results.extend(self.text_db[:k//2])
            
        # Add image results
        if self.image_db:
            results.extend(self.image_db[:k//2])
            
        return results
    
def split_image_text_types(docs: List[Document]) -> Dict:
    """Split retrieved documents into text and image types"""
    result = {
        "texts": [],
        "images": []
    }
    
    for doc in docs:
        if doc.metadata["type"] in ["text", "table"]:
            result["texts"].append(f"""
            Type: {doc.metadata['type']}
            Summary: {doc.page_content}
            Content: {doc.metadata['raw_content']}
            """)
        elif doc.metadata["type"] == "image":
            result["images"].append(doc.metadata["raw_content"])
    
    return result

def multimodal_prompt_function(data_dict: Dict) -> List[HumanMessage]:
    """Create a multimodal prompt with both text and image context"""
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    
    # Add images to messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            }
            messages.append(image_message)
    
    # Add text context and question
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
        
        # Extract images from the context
        retrieved_images = response["context"].get("images", [])

        self.last_query_context = {
            'query': question,
            'generated_response': response['answer'],
            'retrieved_contexts': response['context']['texts'],
            'retrieved_images': response['context']['images']
        }
        
        if return_sources:
            return {
                "answer": response["answer"],
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

class MultiModalRAGResponse(BaseModel):
    answer: str
    images: List[str] = []  # Base64 encoded images

class ResponseEval(BaseModel):
    answer: str
    reference_context: str
    reference_response: str

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
    logging.info(f"Query: {query_input.question}, AI Response: {result['answer']}")
    
    return MultiModalRAGResponse(
        answer=result['answer'], 
        images=result['images']
    )

@app.post("/evaluate", response_model=ResponseEval)
async def evaluate(response_input: ResponseEval):
    logging.info(
        f"Query response: {response_input.answer}",
        f"Reference context: {response_input.reference_context}",
        f"Reference response: {response_input.reference_response}"
    )
    
    if not response_input.answer:
        logging.error("Response is not generated")
        raise HTTPException(status_code=400, detail="Response cannot be empty")
        
    return ResponseEval(
        answer=response_input.answer,
        reference_context=response_input.reference_context,
        reference_response=response_input.reference_response
    )