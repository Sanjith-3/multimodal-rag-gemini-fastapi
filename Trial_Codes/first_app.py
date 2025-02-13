import streamlit as st
import requests
import base64
from PIL import Image
import io
import json

# Configure the API endpoint
API_BASE_URL = "http://127.0.0.1:8000"  # Adjust if your FastAPI server runs on a different port

def display_base64_image(base64_string, caption=""):
    """Display base64 encoded image in Streamlit"""
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption=caption)

def main():
    st.title("Multimodal RAG System")
    
    # Sidebar for document upload
    st.sidebar.header("Upload Documents")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document", 
        type=['pdf'],
        help="Upload a PDF document to process"
    )

    if uploaded_file:
        with st.sidebar:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Prepare the file for upload
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    
                    try:
                        # Call the upload endpoint
                        response = requests.post(
                            f"{API_BASE_URL}/upload-doc",
                            files=files
                        )
                        
                        if response.status_code == 200:
                            st.success("Document processed successfully!")
                        else:
                            st.error(f"Error: {response.json()['detail']}")
                    except Exception as e:
                        st.error(f"Error connecting to server: {str(e)}")

    # Main chat interface
    st.header("Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display images if present
            if "images" in message and message["images"]:
                for idx, img in enumerate(message["images"]):
                    display_base64_image(img, f"Retrieved Image {idx + 1}")
            
            # Display sources if present
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        **Document ID:** {source['doc_id']}  
                        **Type:** {source['type']}  
                        **Summary:** {source['summary']}  
                        """)

    # Chat input
    if prompt := st.chat_input("Ask a question about the documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={"question": prompt}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display the answer
                        st.write(result["answer"])
                        
                        # Display retrieved images
                        if result.get("images"):
                            for idx, img in enumerate(result["images"]):
                                display_base64_image(img, f"Retrieved Image {idx + 1}")
                        
                        # Display sources in expander
                        if result.get("sources"):
                            with st.expander("View Sources"):
                                for source in result["sources"]:
                                    st.markdown(f"""
                                    **Document ID:** {source['doc_id']}  
                                    **Type:** {source['type']}  
                                    **Summary:** {source['summary']}  
                                    """)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "images": result.get("images", []),
                            "sources": result.get("sources", [])
                        })
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Error connecting to server: {str(e)}")

    # Evaluation section
    st.header("Response Evaluation")
    with st.expander("Evaluate Responses"):
        reference_response = st.text_area("Reference Response")
        reference_context = st.text_area("Reference Context")
        response_images = st.file_uploader("Upload Reference Response Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        context_images = st.file_uploader("Upload Reference Context Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        if st.button("Evaluate"):
            if not reference_response or not reference_context:
                st.warning("Please provide both reference response and context.")
            else:
                with st.spinner("Evaluating..."):
                    try:
                        # Prepare the evaluation request
                        files = []
                        
                        # Add response images
                        for img in response_images:
                            files.append(("response_images", (img.name, img.getvalue(), "image/jpeg")))
                            
                        # Add context images
                        for img in context_images:
                            files.append(("context_images", (img.name, img.getvalue(), "image/jpeg")))
                            
                        # Add text data
                        data = {
                            "reference_response": reference_response,
                            "reference_context": reference_context,
                            "answer": st.session_state.messages[-1]["content"] if st.session_state.messages else ""
                        }
                        
                        response = requests.post(
                            f"{API_BASE_URL}/evaluate",
                            data=data,
                            files=files
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display evaluation results
                            st.subheader("Evaluation Results")
                            
                            # Response Metrics
                            st.markdown("### Response Metrics")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Text Metrics**")
                                metrics = result["evaluation_results"]["response_metrics"]["text"]
                                st.write(f"BLEU Score: {metrics['bleu_score']:.4f}")
                                st.write(f"ROUGE-L F1: {metrics['rouge_l_f1']:.4f}")
                                st.write(f"Semantic Similarity: {metrics['semantic_similarity']:.4f}")
                            
                            with col2:
                                st.markdown("**Image Metrics**")
                                if result["evaluation_results"]["response_metrics"]["images"]["average_score"] is not None:
                                    st.write(f"Average Image Similarity: {result['evaluation_results']['response_metrics']['images']['average_score']:.4f}")
                                else:
                                    st.write("No image metrics available")
                            
                            # Context Metrics
                            st.markdown("### Context Metrics")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Text Metrics**")
                                metrics = result["evaluation_results"]["context_metrics"]["text"]
                                st.write(f"BLEU Score: {metrics['bleu_score']:.4f}")
                                st.write(f"ROUGE-L F1: {metrics['rouge_l_f1']:.4f}")
                                st.write(f"Semantic Similarity: {metrics['semantic_similarity']:.4f}")
                            
                            with col2:
                                st.markdown("**Image Metrics**")
                                if result["evaluation_results"]["context_metrics"]["images"]["average_score"] is not None:
                                    st.write(f"Average Image Similarity: {result['evaluation_results']['context_metrics']['images']['average_score']:.4f}")
                                else:
                                    st.write("No image metrics available")
                        else:
                            st.error(f"Error: {response.json()['detail']}")
                    except Exception as e:
                        st.error(f"Error connecting to server: {str(e)}")

if __name__ == "__main__":
    main()