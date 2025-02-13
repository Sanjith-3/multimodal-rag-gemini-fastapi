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

if __name__ == "__main__":
    main()