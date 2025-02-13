import streamlit as st
import requests
import base64
from PIL import Image
import io
import json

# Configure the API endpoint
API_BASE_URL = "http://127.0.0.1:8000"

def display_base64_image(base64_string, caption=""):
    """Display base64 encoded image in Streamlit"""
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption=caption)

def safe_float_format(value, default=0.0):
    """Safely format a value as a float, returning a default if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

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
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    try:
                        response = requests.post(f"{API_BASE_URL}/upload-doc", files=files)
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
        st.session_state.queries = []  # Store queries for evaluation dropdown

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if "images" in message and message["images"]:
                for idx, img in enumerate(message["images"]):
                    display_base64_image(img, f"Retrieved Image {idx + 1}")
            
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
        # Store query for evaluation
        st.session_state.queries.append(prompt)
        
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
                        st.write(result["answer"])
                        
                        if result.get("images"):
                            for idx, img in enumerate(result["images"]):
                                display_base64_image(img, f"Retrieved Image {idx + 1}")
                        
                        if result.get("sources"):
                            with st.expander("View Sources"):
                                for source in result["sources"]:
                                    st.markdown(f"""
                                    **Document ID:** {source['doc_id']}  
                                    **Type:** {source['type']}  
                                    **Summary:** {source['summary']}  
                                    """)
                        
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
        # Dropdown for selecting query to evaluate
        if st.session_state.queries:
            selected_query = st.selectbox(
                "Select Query to Evaluate",
                options=st.session_state.queries,
                format_func=lambda x: f"Query: {x[:50]}..."
            )
            query_index = st.session_state.queries.index(selected_query)
            
            # Auto-populate answer based on selected query
            if query_index < len(st.session_state.messages):
                assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
                if query_index < len(assistant_messages):
                    selected_answer = assistant_messages[query_index]["content"]
                else:
                    selected_answer = ""
        
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
                        files = []
                        
                        for img in response_images:
                            files.append(("response_images", (img.name, img.getvalue(), "image/jpeg")))
                        
                        for img in context_images:
                            files.append(("context_images", (img.name, img.getvalue(), "image/jpeg")))
                        
                        data = {
                            "reference_response": reference_response,
                            "reference_context": reference_context,
                            "answer": selected_answer if "selected_answer" in locals() else st.session_state.messages[-1]["content"]
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
                                st.write(f"BLEU Score: {safe_float_format(metrics['bleu_score']):.4f}")
                                st.write(f"ROUGE-L F1: {safe_float_format(metrics['rouge_l_f1']):.4f}")
                                st.write(f"Semantic Similarity: {safe_float_format(metrics['semantic_similarity']):.4f}")
                            
                            with col2:
                                st.markdown("**Image Metrics**")
                                if result["evaluation_results"]["response_metrics"]["images"]["average_score"] is not None:
                                    st.write(f"Average Image Similarity: {safe_float_format(result['evaluation_results']['response_metrics']['images']['average_score']):.4f}")
                                else:
                                    st.write("No image metrics available")
                            
                            # Context Metrics
                            st.markdown("### Context Metrics")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Text Metrics**")
                                metrics = result["evaluation_results"]["context_metrics"]["text"]
                                st.write(f"BLEU Score: {safe_float_format(metrics['bleu_score']):.4f}")
                                st.write(f"ROUGE-L F1: {safe_float_format(metrics['rouge_l_f1']):.4f}")
                                st.write(f"Semantic Similarity: {safe_float_format(metrics['semantic_similarity']):.4f}")
                            
                            with col2:
                                st.markdown("**Image Metrics**")
                                if result["evaluation_results"]["context_metrics"]["images"]["average_score"] is not None:
                                    st.write(f"Average Image Similarity: {safe_float_format(result['evaluation_results']['context_metrics']['images']['average_score']):.4f}")
                                else:
                                    st.write("No image metrics available")
                                    
                            # Framework Metrics
                            # Framework Metrics
                                st.markdown("### Framework Metrics")
                                metrics = result["evaluation_results"]["framework_metrics"]
                                for metric_name, metric_data in metrics.items():
                                    if isinstance(metric_data, dict):
                                        st.write(f"**{metric_name}:**")
                                        st.write(f"Score: {metric_data['score']:.4f}")
                                    if "explanation" in metric_data:
                                        st.write(f"Explanation: {metric_data['explanation']}")
                                    if "semantic_similarity" in metric_data:
                                        st.write(f"Semantic Similarity: {metric_data['semantic_similarity']:.4f}")
                                    else:
                                        st.write(f"**{metric_name}:** {metric_data:.4f}")
                    except Exception as e:
                        st.error(f"Error connecting to server: {str(e)}")

if __name__ == "__main__":
    main()