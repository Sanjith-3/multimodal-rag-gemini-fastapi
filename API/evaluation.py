from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from typing import List, Optional
from typing import Dict, Any
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
import logging
from groq import Groq
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from langchain_core.messages import HumanMessage

app = FastAPI()

class ImageEvaluator:
    def __init__(self, google_api_key: str):
        """
        Initialize the evaluator with Google API key.
        
        Args:
            google_api_key: API key for Google Gemini
        """
        self.vision_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            max_tokens=1024,
            google_api_key=google_api_key
        )

    def evaluate_image_similarity(self, reference_image: str, retrieved_image: str) -> float:
        """
        Evaluate similarity between two images using Gemini vision model.
        
        Args:
            reference_image: Base64 encoded image string
            retrieved_image: Base64 encoded image string
            
        Returns:
            float: Similarity score between 0 and 1, or None if evaluation fails
        """
        prompt = """Compare these two images and provide a similarity score from 0 to 1.
        0 means completely different, 1 means identical.
        Provide only the numerical score."""
        
        try:
            # Ensure images are properly base64 encoded
            try:
                reference_image_bytes = base64.b64decode(reference_image)
                retrieved_image_bytes = base64.b64decode(retrieved_image)
            except Exception:
                # If decoding fails, assume the images are raw bytes
                reference_image_bytes = reference_image
                retrieved_image_bytes = retrieved_image

            # Re-encode to ensure proper base64 format
            reference_base64 = base64.b64encode(reference_image_bytes).decode('utf-8')
            retrieved_base64 = base64.b64encode(retrieved_image_bytes).decode('utf-8')
            
            response = self.vision_model.invoke([
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{reference_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{retrieved_base64}"}},
                    {"type": "text", "text": prompt}
                ])
            ])
            
            try:
                # Extract numerical score using regex
                import re
                response_text = response.content
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))
                return 0.0
            except Exception as e:
                print(f"Error parsing similarity score: {e}")
                return 0.0

        except Exception as e:
            print(f"Image similarity evaluation error: {e}")
            return 0.0        
import os
from dotenv import load_dotenv

load_dotenv()


@app.post("/evaluate")
async def evaluate(
    reference_response: str = Form(...),
    generated_response: str = Form(...),
    reference_context: str = Form(...),
    retrieved_context: str = Form(...),
    reference_response_images: Optional[List[UploadFile]] = File(None),
    retrieved_response_images: Optional[List[UploadFile]] = File(None),
    reference_context_images: Optional[List[UploadFile]] = File(None),
    retrieved_context_images: Optional[List[UploadFile]] = File(None)
):
    evaluator = RAGEvaluator(groq_api_key=os.getenv("GROQ_API_KEY"))
    
    # Process all images to base64
    async def process_images(images: Optional[List[UploadFile]]) -> List[str]:
        result = []
        if images:
            for image in images:
                try:
                    image_content = await image.read()
                    # Convert bytes directly to base64
                    base64_image = base64.b64encode(image_content).decode('utf-8')
                    result.append(base64_image)
                except Exception as e:
                    logging.error(f"Error processing image: {e}")
                    result.append("")
        return result

    # Process all image sets
    ref_response_images = await process_images(reference_response_images)
    ret_response_images = await process_images(retrieved_response_images)
    ref_context_images = await process_images(reference_context_images)
    ret_context_images = await process_images(retrieved_context_images)
    
    # Get evaluation results
    evaluation_results = evaluator.evaluate_all(
        reference_response=reference_response,
        generated_response=generated_response,
        reference_context=reference_context,
        retrieved_context=retrieved_context,
        reference_response_images=ref_response_images,
        retrieved_response_images=ret_response_images,
        reference_context_images=ref_context_images,
        retrieved_context_images=ret_context_images
    )
    
    # Convert numpy values to Python native types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
    
    evaluation_results = convert_numpy(evaluation_results)
    
    return {
        "generated_response": generated_response,
        "reference_response": reference_response,
        "reference_context": reference_context,
        "retrieved_context": retrieved_context,
        "evaluation_results": evaluation_results
    }
class EnhancedFrameworkEvaluator:
    def __init__(self, groq_api_key: str):
        """
        Initialize the evaluator with necessary models and clients.
        
        Args:
            groq_api_key: API key for Groq
        """
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Initialize Sentence Transformer for semantic similarity
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Evaluation prompts
        self.prompts = {
            "contextual_recall": """
            Task: Evaluate how well the generated response captures the relevant information from the reference context.
            
            Reference Context: {context}
            Generated Response: {response}
            
            Analyze:
            1. Key information coverage
            2. Missing important details
            3. Information completeness
            
            Provide a score between 0 and 1, where:
            0 = No relevant information from context used
            1 = Perfect coverage of all relevant information
            
            Output the score with a brief explanation in this format:
            Score: [number]
            Explanation: [brief explanation]
            """,
            
            "answer_relevancy": """
            Task: Evaluate how relevant the generated response is to the reference response.
            
            Reference Response: {ref_response}
            Generated Response: {gen_response}
            
            Analyze:
            1. Topic alignment
            2. Key points coverage
            3. Answer focus
            
            Provide a score between 0 and 1, where:
            0 = Completely irrelevant
            1 = Perfectly relevant
            
            Output the score with a brief explanation in this format:
            Score: [number]
            Explanation: [brief explanation]
            """,
            
            "faithfulness": """
            Task: Evaluate the faithfulness of the generated response to the reference context.
            
            Reference Context: {context}
            Generated Response: {response}
            
            Analyze:
            1. Factual accuracy
            2. Hallucination detection
            3. Information consistency
            
            Provide a score between 0 and 1, where:
            0 = Complete hallucination/unfaithful
            1 = Perfectly faithful to context
            
            Output the score with a brief explanation in this format:
            Score: [number]
            Explanation: [brief explanation]
            """
        }

    def _get_groq_response(self, prompt: str) -> Dict[str, Any]:
        """
        Get response from Groq API using mixtral model
        """
        completion = self.groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are an expert evaluator for text generation systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        response = completion.choices[0].message.content
        
        # Extract score and explanation
        try:
            score_line = [line for line in response.split('\n') if line.startswith('Score:')][0]
            score = float(score_line.split(':')[1].strip())
            
            explanation_line = [line for line in response.split('\n') if line.startswith('Explanation:')][0]
            explanation = explanation_line.split(':')[1].strip()
            
            return {"score": score, "explanation": explanation}
        except:
            return {"score": 0.0, "explanation": "Error parsing response"}

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using sentence transformers
        """
        embeddings1 = self.semantic_model.encode([text1])
        embeddings2 = self.semantic_model.encode([text2])
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        return float(similarity)

    def evaluate_framework_metrics(
        self,
        generated_response: str,
        reference_response: str,
        reference_context: str
    ) -> Dict[str, Any]:
        """
        Evaluate framework metrics using multiple approaches
        """
        # Calculate metrics using Groq
        contextual_recall = self._get_groq_response(
            self.prompts["contextual_recall"].format(
                context=reference_context,
                response=generated_response
            )
        )
        
        answer_relevancy = self._get_groq_response(
            self.prompts["answer_relevancy"].format(
                ref_response=reference_response,
                gen_response=generated_response
            )
        )
        
        faithfulness = self._get_groq_response(
            self.prompts["faithfulness"].format(
                context=reference_context,
                response=generated_response
            )
        )
        
        # Calculate semantic similarities
        context_similarity = self._calculate_semantic_similarity(
            reference_context,
            generated_response
        )
        
        response_similarity = self._calculate_semantic_similarity(
            reference_response,
            generated_response
        )
        
        # Combine metrics
        contextual_precision = (faithfulness["score"] + context_similarity) / 2
        contextual_relevancy = (answer_relevancy["score"] + response_similarity) / 2
        
        return {
            "contextual_recall": {
                "score": contextual_recall["score"],
                "explanation": contextual_recall["explanation"]
            },
            "answer_relevancy": {
                "score": answer_relevancy["score"],
                "explanation": answer_relevancy["explanation"]
            },
            "faithfulness": {
                "score": faithfulness["score"],
                "explanation": faithfulness["explanation"]
            },
            "contextual_precision": {
                "score": float(contextual_precision),
                "semantic_similarity": float(context_similarity)
            },
            "contextual_relevancy": {
                "score": float(contextual_relevancy),
                "semantic_similarity": float(response_similarity)
            },
            "overall_score": float(np.mean([
                contextual_recall["score"],
                answer_relevancy["score"],
                faithfulness["score"],
                contextual_precision,
                contextual_relevancy
            ]))
        }
    
google_api_key = os.getenv("GOOGLE_API_KEY_2")

class RAGEvaluator:
    def __init__(self, groq_api_key: str):
        self.rouge = Rouge()
        self.text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_evaluator = ImageEvaluator(groq_api_key)
        self.image_evaluator = ImageEvaluator(google_api_key)
        self.framework_evaluator = EnhancedFrameworkEvaluator(groq_api_key)

    def calculate_text_metrics(self, generated_text, reference_text):
    # Check for empty strings
        if not generated_text or not reference_text:
            logging.warning("Generated text or reference text is empty. Returning default metrics.")
            return {
                "bleu_score": 0.0,
                "rouge_l_precision": 0.0,
                "rouge_l_recall": 0.0,
                "rouge_l_f1": 0.0,
                "semantic_similarity": 0.0
            }

    # BLEU Score
        reference = reference_text.split()
        candidate = generated_text.split()
        bleu_score = sentence_bleu(reference, candidate)

    # ROUGE Score
        rouge_scores = self.rouge.get_scores(generated_text, reference_text)[0]

    # Semantic Similarity
        gen_embedding = self.text_embedding_model.encode([generated_text])
        ref_embedding = self.text_embedding_model.encode([reference_text])
        semantic_similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]

        return {
            "bleu_score": bleu_score,
            "rouge_l_precision": rouge_scores['rouge-l']['p'],
            "rouge_l_recall": rouge_scores['rouge-l']['r'],
            "rouge_l_f1": rouge_scores['rouge-l']['f'],
            "semantic_similarity": semantic_similarity
        }

    def evaluate_all(
        self,
        reference_response: str,
        generated_response: str,
        reference_context: str,
        retrieved_context: str,
        reference_response_images: List[str],
        retrieved_response_images: List[str],
        reference_context_images: List[str],
        retrieved_context_images: List[str]
        ):
            # Calculate text metrics
            response_metrics = self.calculate_text_metrics(
                generated_response,
                reference_response
            )
            
            context_metrics = self.calculate_text_metrics(
                retrieved_context,
                reference_context
            )
            
            # Calculate framework metrics
            framework_metrics = self.framework_evaluator.evaluate_framework_metrics(
                generated_response,
                reference_response,
                reference_context
            )
            
            # Image metrics calculation using Groq
            response_image_similarities = []
            for ref_img, ret_img in zip(reference_response_images, retrieved_response_images):
                similarity = self.image_evaluator.evaluate_image_similarity(ref_img, ret_img)
                response_image_similarities.append(float(similarity))
                
            context_image_similarities = []
            for ref_img, ret_img in zip(reference_context_images, retrieved_context_images):
                similarity = self.image_evaluator.evaluate_image_similarity(ref_img, ret_img)
                context_image_similarities.append(float(similarity))
            
            return {
                "response_metrics": {
                    "text": response_metrics,
                    "images": {
                        "individual_scores": response_image_similarities,
                        "average_score": sum(response_image_similarities) / len(response_image_similarities) if response_image_similarities else None
                    }
                },
                "context_metrics": {
                    "text": context_metrics,
                    "images": {
                        "individual_scores": context_image_similarities,
                        "average_score": sum(context_image_similarities) / len(context_image_similarities) if context_image_similarities else None
                    }
                },
                "framework_metrics": framework_metrics
            }

@app.post("/evaluate")
async def evaluate(
    reference_response: str = Form(...),
    generated_response: str = Form(...),
    reference_context: str = Form(...),
    retrieved_context: str = Form(...),
    reference_response_images: Optional[List[UploadFile]] = File(None),
    retrieved_response_images: Optional[List[UploadFile]] = File(None),
    reference_context_images: Optional[List[UploadFile]] = File(None),
    retrieved_context_images: Optional[List[UploadFile]] = File(None)
):
    # Initialize evaluator with your Groq API key
    evaluator = RAGEvaluator(groq_api_key="")
    

    
    # Process all images to base64
    async def process_images(images: Optional[List[UploadFile]]) -> List[str]:
        result = []
        if images:
            for image in images:
                try:
                    image_content = await image.read()
                    base64_image = base64.b64encode(image_content).decode('utf-8')
                    result.append(base64_image)
                except Exception as e:
                    logging.error(f"Error processing image: {e}")
                    result.append("")
        return result

    # Process all image sets
    ref_response_images = await process_images(reference_response_images)
    ret_response_images = await process_images(retrieved_response_images)
    ref_context_images = await process_images(reference_context_images)
    ret_context_images = await process_images(retrieved_context_images)
    
    # Get evaluation results
    evaluation_results = evaluator.evaluate_all(
        reference_response=reference_response,
        generated_response=generated_response,
        reference_context=reference_context,
        retrieved_context=retrieved_context,
        reference_response_images=ref_response_images,
        retrieved_response_images=ret_response_images,
        reference_context_images=ref_context_images,
        retrieved_context_images=ret_context_images
    )
    
    # Convert numpy values to Python native types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
    
    evaluation_results = convert_numpy(evaluation_results)
    
    return {
        "generated_response": generated_response,
        "reference_response": reference_response,
        "reference_context": reference_context,
        "retrieved_context": retrieved_context,
        "evaluation_results": evaluation_results
    }
