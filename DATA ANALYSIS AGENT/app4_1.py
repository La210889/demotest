import os
import pandas as pd
import streamlit as st
from phi.agent import Agent
from phi.tools import tool
from phi.workflow import Workflow
from phi.model.groq import Groq
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ---- 1. Load Environment Variables from .env ----
load_dotenv()

#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# ---- 2. Initialize LLM ----
#model = Gemini()
model = Groq(id="llama-3.3-70b-versatile")

# ---- 3. Initialize Pinecone ----
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "data-insights"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # <-- adjust region to match free plan
    )

index = pc.Index(index_name)

# ---- 4. Load SentenceTransformer model for embeddings ----
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- 5. Tool: Describe CSV data ----
@tool
def describe_data(filepath: str) -> str:
    df = pd.read_csv(filepath)
    return df.describe().to_string()

# ---- 6. Tool: Chunk, Embed, and Store CSV metadata ----
@tool
def embed_and_store(filepath: str) -> str:
    df = pd.read_csv(filepath)
    text_summary = df.describe().to_string()

    # ---- Chunking logic ----
    max_chunk_size = 300  # characters per chunk
    chunks = [text_summary[i:i+max_chunk_size] for i in range(0, len(text_summary), max_chunk_size)]

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vector_id = f"{filepath}_chunk_{i}"
        vectors.append((vector_id, embedding))

    index.upsert(vectors)
    return f"Stored {len(vectors)} chunks for {filepath}"

# ---- 7. Tool: Search similar datasets by metadata ----
@tool
def search_similar(query: str) -> str:
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(query_embedding, top_k=5, include_metadata=False)
    matches = [f"ID: {match['id']}, Score: {match['score']}" for match in results['matches']]
    return "\n".join(matches)

# ---- 8. Create the Agent ----
agent = Agent(
    tools=[describe_data, embed_and_store, search_similar],
    model=model,
    name="Data Analyst Agent",
    description="Analyzes CSVs, chunks and stores metadata, and retrieves similar datasets."
)

workflow = Workflow(
    agents=[agent],
    name="csv_insight_workflow"
)

# Streamlit UI integration
# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("📊 Data Analysis Agent")

# Upload a CSV file
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # --------------------------------------------------
        # Save uploaded file
        # --------------------------------------------------

        file_path = os.path.join(
            os.getcwd(),
            "uploaded_data.csv"
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("File uploaded successfully!")

        # --------------------------------------------------
        # Initial dataset summary
        # --------------------------------------------------

        response = agent.run(
            "Describe the dataset in uploaded_data.csv"
        )

        st.write("### Dataset Summary:")

        if hasattr(response, "content"):
            st.markdown(response.content)
        else:
            st.markdown(str(response))

        # --------------------------------------------------
        # Download summary
        # --------------------------------------------------

        summary_df = pd.read_csv(file_path).describe()

        csv_data = summary_df.to_csv(index=True)

        st.download_button(
            label="Download Dataset Summary as CSV",
            data=csv_data,
            file_name="data_summary.csv",
            mime="text/csv"
        )

        # ==================================================
        # CONVERSATIONAL INTERFACE
        # ==================================================

        st.divider()

        st.subheader("💬 Ask Questions About Your Dataset")

        st.caption(
            "Ask questions about the uploaded CSV in natural language."
        )

        # --------------------------------------------------
        # Initialize chat history
        # --------------------------------------------------

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # --------------------------------------------------
        # Display previous messages
        # --------------------------------------------------

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # --------------------------------------------------
        # Chat input
        # --------------------------------------------------

        user_question = st.chat_input(
            "Ask something about your dataset..."
        )

        # --------------------------------------------------
        # Process question
        # --------------------------------------------------

        if user_question:

            # Display user question
            with st.chat_message("user"):
                st.markdown(user_question)

            # Save question
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )

            # Ask agent
            with st.chat_message("assistant"):

                with st.spinner("Analyzing your dataset..."):

                    try:

                        prompt = f"""
You are a Data Analysis Agent.

The user has uploaded a dataset named:
uploaded_data.csv

Use the available tools to analyze this dataset.

User question:
{user_question}

Answer the question using the dataset.
Do not invent information.
If the available data is insufficient to answer the question,
clearly say so.
"""

                        response = agent.run(prompt)

                        # Get only the actual response
                        if hasattr(response, "content"):
                            answer = response.content
                        else:
                            answer = str(response)

                        st.markdown(answer)

                        # Save answer
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )

                    except Exception as e:

                        st.error(
                            f"Error while answering question: {e}"
                        )

    except Exception as e:

        st.error(
            f"Error during file upload: {e}"
        )

# st.title("Data Analysis Agent")

# # Upload a CSV file
# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# if uploaded_file is not None:
#     try:
#         # Define a valid file path where the app has write access
#         file_path = os.path.join(os.getcwd(), "uploaded_data.csv")

#         # Save the uploaded file locally
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         st.success("File uploaded successfully!")

#         # Run the agent to describe the data
#         response = agent.run("Describe the dataset in uploaded_data.csv")
#         st.write("Dataset Summary:")

#         if hasattr(response, "content"):
#             st.markdown(response.content)
#         else:
#             st.markdown(str(response))
#         # st.write("Dataset Summary:")
#         # st.text(response)  # Display the summary

#         # Allow the user to download the summary as a CSV file
#         # Convert the summary to a pandas DataFrame
#         summary_df = pd.read_csv(file_path).describe()

#         # Create a CSV file for download
#         csv_data = summary_df.to_csv(index=True)

#         st.download_button(
#             label="Download Dataset Summary as CSV",
#             data=csv_data,
#             file_name="data_summary.csv",
#             mime="text/csv"
#         )
        
#     except Exception as e:
#         # Handle errors gracefully and display an error message
#         st.error(f"Error during file upload: {e}")

