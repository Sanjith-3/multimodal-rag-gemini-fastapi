import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
from PIL import Image
import io
import json
import pandas as pd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_BASE_URL = "http://127.0.0.1:8000"

def display_base64_image(base64_string, caption=""):
    """Display base64 encoded image in Streamlit"""
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption=caption)

def create_metrics_chart(metrics):
    """Create a plotly bar chart for current metrics"""
    df = pd.DataFrame({
        'Metric': ['BLEU', 'ROUGE-L', 'Semantic'],
        'Score': [
            metrics['bleu_score'],
            metrics['rouge_l_f1'],
            metrics['semantic_similarity']
        ]
    })
    
    fig = px.bar(
        df,
        x='Metric',
        y='Score',
        color='Metric',
        title='Response Metrics',
        range_y=[0, 1]
    )
    
    fig.update_layout(
        showlegend=False,
        height=300,
        title_x=0.5,
        title_y=0.95
    )
    
    return fig

def create_metrics_trends(metrics_history):
    """Create a plotly line chart for metrics trends"""
    df = pd.DataFrame(metrics_history)
    df.index = range(1, len(df) + 1)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('BLEU Score Trend', 'ROUGE-L F1 Trend', 'Semantic Similarity Trend')
    )
    
    # Add traces for each metric
    fig.add_trace(
        go.Scatter(x=df.index, y=df['bleu_score'], mode='lines+markers', name='BLEU'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['rouge_l_f1'], mode='lines+markers', name='ROUGE-L'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['semantic_similarity'], mode='lines+markers', name='Semantic'),
        row=1, col=3
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Metrics Trends Across Queries",
        title_x=0.5
    )
    
    # Update y-axis range for all subplots
    fig.update_yaxes(range=[0, 1])
    
    return fig

def main():
    st.title("Multimodal RAG System")
    
    # Initialize metrics history in session state
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    
    # ... (previous document upload sidebar code remains the same)
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history with metrics
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display images if present
            if "images" in message and message["images"]:
                for img_idx, img in enumerate(message["images"]):
                    display_base64_image(img, f"Retrieved Image {img_idx + 1}")
            
            # Display sources if present
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        **Document ID:** {source['doc_id']}  
                        **Type:** {source['type']}  
                        **Summary:** {source['summary']}  
                        """)
            
            # Display metrics for assistant messages
            if message["role"] == "assistant" and idx // 2 < len(st.session_state.metrics_history):
                metrics = st.session_state.metrics_history[idx // 2]
                with st.expander("View Response Metrics"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### Metrics Values")
                        st.write(f"BLEU Score: {metrics['bleu_score']:.4f}")
                        st.write(f"ROUGE-L F1: {metrics['rouge_l_f1']:.4f}")
                        st.write(f"Semantic Similarity: {metrics['semantic_similarity']:.4f}")
                    
                    with col2:
                        st.plotly_chart(create_metrics_chart(metrics), use_container_width=True)

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
                        
                        # Get metrics for the response
                        try:
                            metrics_response = requests.post(
                                f"{API_BASE_URL}/evaluate",
                                json={
                                    "reference_response": result["answer"],
                                    "reference_context": "",  # You might want to pass actual context here
                                    "answer": result["answer"]
                                }
                            )
                            
                            if metrics_response.status_code == 200:
                                metrics_result = metrics_response.json()
                                response_metrics = metrics_result["evaluation_results"]["response_metrics"]["text"]
                                
                                # Store metrics in history
                                st.session_state.metrics_history.append(response_metrics)
                                
                                # Display current metrics
                                with st.expander("View Response Metrics", expanded=True):
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        st.markdown("### Metrics Values")
                                        st.write(f"BLEU Score: {response_metrics['bleu_score']:.4f}")
                                        st.write(f"ROUGE-L F1: {response_metrics['rouge_l_f1']:.4f}")
                                        st.write(f"Semantic Similarity: {response_metrics['semantic_similarity']:.4f}")
                                    
                                    with col2:
                                        st.plotly_chart(create_metrics_chart(response_metrics), use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error getting metrics: {str(e)}")
                        
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

    # Display metrics trends
    if st.session_state.metrics_history:
        st.header("Response Metrics Trends")
        st.plotly_chart(create_metrics_trends(st.session_state.metrics_history), use_container_width=True)

if __name__ == "__main__":
    main()