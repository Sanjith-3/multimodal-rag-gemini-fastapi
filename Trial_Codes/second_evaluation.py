import streamlit as st
import requests
import base64
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from streamlit_custom_tooltip import st_custom_tooltip
import streamlit.components.v1 as components

from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from streamlit_custom_tooltip import st_custom_tooltip
import streamlit.components.v1 as components


# Configure the API endpoint
API_BASE_URL = "http://127.0.0.1:8000"

# Custom CSS to make chat interface full width
st.set_page_config(layout="wide")

def custom_css():
    st.markdown("""
        <style>
        .main > div {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        .stChatMessage {
            width: 100%;
        }
        .css-1y4p8pa {
            width: 100%;
            max-width: 100%;
        }
        .source-tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .source-tooltip .tooltip-content {
            visibility: hidden;
            width: 300px;
            background-color: #f9f9f9;
            color: #333;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            border: 1px solid #ddd;
        }
        .source-tooltip:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

def display_base64_image(base64_string, caption=""):
    """Display base64 encoded image in Streamlit"""
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption=caption)

def create_metrics_chart(metrics_history):
    """Create a plotly chart for metrics history"""
    if not metrics_history:
        return None
    
    df = pd.DataFrame(metrics_history)
    df['query_number'] = range(1, len(df) + 1)
    
    fig = go.Figure()
    
    # Add traces for each metric
    fig.add_trace(go.Scatter(x=df['query_number'], y=df['bleu_score'], 
                            name='BLEU Score', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df['query_number'], y=df['rouge_f1'], 
                            name='ROUGE-L F1', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df['query_number'], y=df['semantic_similarity'], 
                            name='Semantic Similarity', mode='lines+markers'))
    
    fig.update_layout(
        title='Response Metrics History',
        xaxis_title='Query Number',
        yaxis_title='Score',
        hovermode='x unified',
        height=300
    )
    
    return fig

def display_source_with_tooltip(source):
    """Create a hoverable info button with source information"""
    tooltip_content = f"""
    <div class='tooltip-content'>
        <p><strong>Document ID:</strong> {source['doc_id']}</p>
        <p><strong>Type:</strong> {source['type']}</p>
        <p><strong>Summary:</strong> {source['summary']}</p>
    </div>
    """
    
    html = f"""
    <div class='source-tooltip'>
        <span>📄 Source Info</span>
        {tooltip_content}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def main():
    custom_css()
    
    # Initialize session state for metrics history
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    
    # Initialize session state for queries
    if 'queries' not in st.session_state:
        st.session_state.queries = []
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat", "Detailed Evaluation"])
    
    if page == "Chat":
        chat_interface()
    else:
        detailed_evaluation()

def chat_interface():
    # Document upload in sidebar
    st.sidebar.header("Upload Documents")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document", 
        type=['pdf']
    )

    if uploaded_file:
        if st.sidebar.button("Process Document"):
            process_document(uploaded_file)
    
    # Main chat container
    chat_container = st.container()
    
    # Metrics visualization in sidebar
    if st.session_state.metrics_history:
        st.sidebar.header("Response Metrics")
        metrics_chart = create_metrics_chart(st.session_state.metrics_history)
        if metrics_chart:
            st.sidebar.plotly_chart(metrics_chart, use_container_width=True)
    
    with chat_container:
        st.title("Multimodal RAG Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                if "images" in message and message["images"]:
                    cols = st.columns(len(message["images"]))
                    for idx, (img, col) in enumerate(zip(message["images"], cols)):
                        with col:
                            display_base64_image(img, f"Retrieved Image {idx + 1}")
                
                if "sources" in message and message["sources"]:
                    st.markdown("---")
                    for source in message["sources"]:
                        display_source_with_tooltip(source)

        # Chat input
        if prompt := st.chat_input("Ask a question about the documents"):
            handle_chat_input(prompt)

def detailed_evaluation():
    st.title("Detailed Response Evaluation")
    
    # Query selection
    if st.session_state.queries:
        selected_query = st.selectbox(
            "Select Query to Evaluate",
            options=st.session_state.queries,
            format_func=lambda x: f"Query: {x[:100]}..."
        )
        
        # Auto-fill selected query
        st.text_area("Selected Query", value=selected_query, disabled=True)
        
        # Reference inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Response Evaluation")
            reference_response = st.text_area("Reference Response")
            response_images = st.file_uploader(
                "Reference Response Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
            
        with col2:
            st.subheader("Context Evaluation")
            reference_context = st.text_area("Reference Context")
            context_images = st.file_uploader(
                "Reference Context Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
        
        if st.button("Evaluate"):
            evaluate_response(
                selected_query,
                reference_response,
                reference_context,
                response_images,
                context_images
            )
    else:
        st.info("No queries available for evaluation. Please use the chat interface first.")

def process_document(uploaded_file):
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

def handle_chat_input(prompt):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.queries.append(prompt)
    
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
                        cols = st.columns(len(result["images"]))
                        for idx, (img, col) in enumerate(zip(result["images"], cols)):
                            with col:
                                display_base64_image(img, f"Retrieved Image {idx + 1}")
                    
                    if result.get("sources"):
                        st.markdown("---")
                        for source in result["sources"]:
                            display_source_with_tooltip(source)
                    
                    # Store message in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "images": result.get("images", []),
                        "sources": result.get("sources", [])
                    })
                    
                    # Update metrics history
                    if result.get("metrics"):
                        st.session_state.metrics_history.append({
                            'bleu_score': result["metrics"]["bleu_score"],
                            'rouge_f1': result["metrics"]["rouge_l_f1"],
                            'semantic_similarity': result["metrics"]["semantic_similarity"]
                        })
                else:
                    st.error(f"Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"Error connecting to server: {str(e)}")

def evaluate_response(query, reference_response, reference_context, response_images, context_images):
    with st.spinner("Evaluating..."):
        try:
            files = []
            
            # Add images to request
            for img in response_images:
                files.append(("response_images", (img.name, img.getvalue(), "image/jpeg")))
            for img in context_images:
                files.append(("context_images", (img.name, img.getvalue(), "image/jpeg")))
            
            # Add text data
            data = {
                "reference_response": reference_response,
                "reference_context": reference_context,
                "answer": query
            }
            
            response = requests.post(
                f"{API_BASE_URL}/evaluate",
                data=data,
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                display_evaluation_results(result)
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error connecting to server: {str(e)}")

def display_evaluation_results(result):
    st.subheader("Evaluation Results")
    
    # Create tabs for Response and Context metrics
    tab1, tab2 = st.tabs(["Response Metrics", "Context Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Text metrics visualization
            text_metrics = result["evaluation_results"]["response_metrics"]["text"]
            fig = px.bar(
                x=["BLEU", "ROUGE-L F1", "Semantic Similarity"],
                y=[text_metrics["bleu_score"], 
                   text_metrics["rouge_l_f1"],
                   text_metrics["semantic_similarity"]],
                title="Text Metrics"
            )
            st.plotly_chart(fig)
        
        with col2:
            # Image metrics visualization
            if result["evaluation_results"]["response_metrics"]["images"]["average_score"] is not None:
                img_score = result["evaluation_results"]["response_metrics"]["images"]["average_score"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=img_score,
                    title={'text': "Average Image Similarity"},
                    gauge={'axis': {'range': [0, 1]}}
                ))
                st.plotly_chart(fig)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Context text metrics visualization
            context_metrics = result["evaluation_results"]["context_metrics"]["text"]
            fig = px.bar(
                x=["BLEU", "ROUGE-L F1", "Semantic Similarity"],
                y=[context_metrics["bleu_score"], 
                   context_metrics["rouge_l_f1"],
                   context_metrics["semantic_similarity"]],
                title="Context Text Metrics"
            )
            st.plotly_chart(fig)
        
        with col2:
            # Context image metrics visualization
            if result["evaluation_results"]["context_metrics"]["images"]["average_score"] is not None:
                img_score = result["evaluation_results"]["context_metrics"]["images"]["average_score"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=img_score,
                    title={'text': "Context Image Similarity"},
                    gauge={'axis': {'range': [0, 1]}}
                ))
                st.plotly_chart(fig)

def create_response_metrics_chart(metrics_history):
    """Create an interactive plotly chart for response metrics history"""
    if not metrics_history:
        return None
    
    df = pd.DataFrame(metrics_history)
    df['query_number'] = range(1, len(df) + 1)
    
    fig = go.Figure()
    
    # Create traces with improved styling
    fig.add_trace(go.Scatter(
        x=df['query_number'],
        y=df['bleu_score'],
        name='BLEU Score',
        mode='lines+markers',
        line=dict(width=2, color='#1f77b4'),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['query_number'],
        y=df['rouge_f1'],
        name='ROUGE-L F1',
        mode='lines+markers',
        line=dict(width=2, color='#ff7f0e'),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['query_number'],
        y=df['semantic_similarity'],
        name='Semantic Similarity',
        mode='lines+markers',
        line=dict(width=2, color='#2ca02c'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title={
            'text': 'Response Quality Metrics',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Query Number',
        yaxis_title='Score',
        hovermode='x unified',
        height=400,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

def detailed_evaluation():
    st.title("Detailed Response & Context Evaluation")
    
    # Add explanation
    st.info("""
        This page allows you to perform in-depth evaluation of specific queries.
        Select a query from your chat history and provide reference materials to
        evaluate both response accuracy and context relevance.
    """)
    
    # Query selection with improved UI
    if st.session_state.queries:
        # Create a DataFrame for better query display
        query_df = pd.DataFrame({
            'Query Number': range(1, len(st.session_state.queries) + 1),
            'Query Text': st.session_state.queries
        })
        
        # Display queries in a more readable format
        st.subheader("Select Query to Evaluate")
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_query_num = st.selectbox(
                "Query #",
                options=query_df['Query Number'],
                format_func=lambda x: f"Query {x}"
            )
        
        selected_query = query_df[query_df['Query Number'] == selected_query_num]['Query Text'].iloc[0]
        
        # Show selected query in a container
        with st.expander("Selected Query", expanded=True):
            st.text_area("Query Text", value=selected_query, height=100, disabled=True)
        
        # Split evaluation into tabs
        tab1, tab2 = st.tabs(["Response Evaluation", "Context Evaluation"])
        
        with tab1:
            st.subheader("Response Evaluation")
            reference_response = st.text_area(
                "Reference Response",
                help="Enter the expected or ideal response for comparison"
            )
            response_images = st.file_uploader(
                "Reference Response Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload images that should have been included in the response"
            )
        
        with tab2:
            st.subheader("Context Evaluation")
            reference_context = st.text_area(
                "Reference Context",
                help="Enter the relevant context that should have been used"
            )
            context_images = st.file_uploader(
                "Reference Context Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload images that should have been part of the context"
            )
        
        # Evaluation button with loading state
        if st.button("Run Detailed Evaluation", type="primary"):
            with st.spinner("Performing comprehensive evaluation..."):
                evaluate_response(
                    selected_query,
                    reference_response,
                    reference_context,
                    response_images,
                    context_images
                )
    else:
        st.warning("No queries available for evaluation. Please use the chat interface first to ask some questions.")

def display_evaluation_results(result):
    st.header("Evaluation Results")
    
    # Create tabs for different aspects of evaluation
    tabs = st.tabs(["Response Quality", "Context Relevance", "Summary"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Text metrics visualization with improved styling
            text_metrics = result["evaluation_results"]["response_metrics"]["text"]
            fig = px.bar(
                x=["BLEU", "ROUGE-L F1", "Semantic Similarity"],
                y=[text_metrics["bleu_score"], 
                   text_metrics["rouge_l_f1"],
                   text_metrics["semantic_similarity"]],
                title="Response Text Quality Metrics",
                color_discrete_sequence=['#1f77b4'],
                template='plotly_white'
            )
            fig.update_layout(
                yaxis_title="Score",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Image similarity gauge
            if result["evaluation_results"]["response_metrics"]["images"]["average_score"] is not None:
                img_score = result["evaluation_results"]["response_metrics"]["images"]["average_score"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=img_score,
                    title={'text': "Response Image Similarity"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 0.33], 'color': "#ff9999"},
                            {'range': [0.33, 0.66], 'color': "#ffff99"},
                            {'range': [0.66, 1], 'color': "#99ff99"}
                        ]
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Context metrics visualization
            context_metrics = result["evaluation_results"]["context_metrics"]["text"]
            fig = px.bar(
                x=["BLEU", "ROUGE-L F1", "Semantic Similarity"],
                y=[context_metrics["bleu_score"], 
                   context_metrics["rouge_l_f1"],
                   context_metrics["semantic_similarity"]],
                title="Context Relevance Metrics",
                color_discrete_sequence=['#2ca02c'],
                template='plotly_white'
            )
            fig.update_layout(
                yaxis_title="Score",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Context image similarity gauge
            if result["evaluation_results"]["context_metrics"]["images"]["average_score"] is not None:
                img_score = result["evaluation_results"]["context_metrics"]["images"]["average_score"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=img_score,
                    title={'text': "Context Image Relevance"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "#2ca02c"},
                        'steps': [
                            {'range': [0, 0.33], 'color': "#ff9999"},
                            {'range': [0.33, 0.66], 'color': "#ffff99"},
                            {'range': [0.66, 1], 'color': "#99ff99"}
                        ]
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Overall evaluation summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Response Quality Summary")
            response_avg = (text_metrics["bleu_score"] + 
                          text_metrics["rouge_l_f1"] + 
                          text_metrics["semantic_similarity"]) / 3
            st.metric("Average Response Score", f"{response_avg:.2f}")
            
        with col2:
            st.subheader("Context Relevance Summary")
            context_avg = (context_metrics["bleu_score"] + 
                         context_metrics["rouge_l_f1"] + 
                         context_metrics["semantic_similarity"]) / 3
            st.metric("Average Context Score", f"{context_avg:.2f}")

if __name__ == "__main__":
    main()