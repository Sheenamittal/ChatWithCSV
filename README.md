# Chat with CSV 
This project allows you to upload a CSV file and interact with its data using conversational AI, powered by LangChain and the HuggingFace embeddings. The application uses a pre-trained model (Llama 2) to process user queries and provide meaningful insights based on the uploaded CSV data.

## Features
**Upload CSV Files:** Easily upload your CSV file and process it for interaction.

**Conversational Retrieval:** Ask questions about the content of the CSV file using a conversational AI interface.

**Efficient Embeddings and Vector Search:** Uses FAISS for fast vector-based search over the data.

**Streamlit Interface:** Intuitive and user-friendly interface built with Streamlit.

**Llama 2 Model Integration:** Uses the Llama 2 model for generating responses.

## Requirements
To run this project locally, you'll need the following Python libraries:

pip install streamlit langchain faiss-cpu transformers sentence-transformers


## Application Workflow
Upload CSV: Once you upload a CSV file, it gets loaded using CSVLoader from LangChain.
Embedding and Vector Store: The data is embedded using the HuggingFaceEmbeddings model and saved in a FAISS vector store for efficient retrieval.

Conversational Chain: The pre-trained Llama 2 model processes queries in a conversational format, searching through the CSV data using the FAISS retriever.

Interaction: You can ask queries related to the data, and the model will respond based on the CSV content.

## Files
app.py: The main Streamlit application.

vectorstore/: Stores the FAISS database used for vector retrieval.




