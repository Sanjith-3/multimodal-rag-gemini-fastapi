import streamlit as st
import random

# Initialize session state
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""
if "reference_context" not in st.session_state:
    st.session_state["reference_context"] = ""
if "reference_solution" not in st.session_state:
    st.session_state["reference_solution"] = ""
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# Function to generate random metrics
def generate_random_metrics():
    return {
        "Context Recall": round(random.uniform(0.5, 1.0), 2),
        "Context Precision": round(random.uniform(0.5, 1.0), 2),
        "Context Relevance": round(random.uniform(0.5, 1.0), 2),
        "Answer Relevance": round(random.uniform(0.5, 1.0), 2),
        "Faithfulness": round(random.uniform(0.5, 1.0), 2),
    }

# Page 1: Initial Input Form
if not st.session_state["submitted"]:
    st.title("Multi Model RAG System")
    st.write("### Please provide the following details:")

    # Upload Document Field
    st.write("#### Upload Document:")
    st.session_state["uploaded_file"] = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "ppt", "png", "jpg"]
    )

    # Input fields
    st.session_state["user_query"] = st.text_input("User Query:")
    st.session_state["reference_context"] = st.text_area("Reference Context:", height=100)  # Updated height
    st.session_state["reference_solution"] = st.text_area("Reference Solution:", height=100)  # Updated height


    # Submit button
    if st.button("Submit"):
        if (
            st.session_state["uploaded_file"]
            and st.session_state["user_query"]
            and st.session_state["reference_context"]
            and st.session_state["reference_solution"]
        ):
            st.session_state["submitted"] = True
            st.experimental_rerun()
        else:
            st.error("Please fill in all the fields before submitting.")

# Page 2: Evaluation Metrics and Results
else:
    st.title("Response and Its Evaluation of Multi Model RAG System")
    st.write("### Here is the outcome of the system:")

    # Display entered data
    st.write("#### Uploaded Document:")
    st.text(st.session_state["uploaded_file"].name if st.session_state["uploaded_file"] else "No file uploaded")

    st.write("#### User Query:")
    st.text(st.session_state["user_query"])

    st.write("#### Retrieved Context:")
    st.text("""## Demo Retrieved Context ##
Renewable energy sources like solar and wind are critical for reducing carbon emissions. Recent AI advancements enable predictive analytics and operational efficiency, which can optimize these systems. The global economy in 2025 is expected to see significant growth in sustainable technologies.""")

    st.write("#### Response:")
    st.text("""## Demo Response ##
Adopting renewable energy and leveraging AI for optimization will drive sustainability and economic growth, aligning with emerging global market trends in 2025.""")

    # Generate random metrics
    metrics = generate_random_metrics()

    # Display evaluation metrics in a single row
    st.write("### Evaluation Metrics:")
    cols = st.columns(5)
    metric_keys = list(metrics.keys())
    for i, col in enumerate(cols):
        col.metric(metric_keys[i], metrics[metric_keys[i]])

    # Back button
    if st.button("Back"):
        st.session_state["submitted"] = False
        st.experimental_rerun()
