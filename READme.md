# LangChain PDF Chatbot

This project implements a Python-based chatbot application capable of answering user questions based on information extracted from uploaded PDF documents. It leverages advanced Natural Language Processing (NLP) techniques and AI models (specifically Large Language Models and Embeddings via Ollama) to provide contextually relevant responses through a user-friendly web interface. The application is containerized using Docker for easy deployment and reproducibility.

## Table of Contents

1.  [Features and Functionalities](#features-and-functionalities)
2.  [Architecture Overview](#architecture-overview)
3.  [Setup Instructions](#setup-instructions)
    * [Prerequisites](#prerequisites)
    * [Clone the Repository](#clone-the-repository)
    * [Running with Docker Compose](#running-with-docker-compose)
    * [Ollama Model Download](#ollama-model-download)
4.  [API Documentation](#api-documentation)
    * [Endpoint: `/`](#endpoint-/)
    * [Endpoint: `/upload_pdf` (POST)](#endpoint-upload_pdf-post)
5.  [Project Structure](#project-structure)
6.  [Troubleshooting](#troubleshooting)
7.  [License](#license)

## Features and Functionalities

This chatbot provides the following key features:

* **PDF Document Upload:** Users can easily upload PDF files through a dedicated web interface. [cite: 7]
* **Text Extraction & Processing:** The application extracts and processes text content from uploaded PDFs to create a searchable knowledge base. [cite: 4]
* **Intelligent Q&A:** Users can ask questions related to the content of the uploaded PDF documents, and the chatbot will provide accurate and contextually relevant answers. [cite: 5, 6]
* **LangChain Integration:** Utilizes the LangChain framework for orchestrating document loading, text splitting, embedding generation, vector storage, and conversational chains.
* **Ollama Integration:** Leverages Ollama for running local Large Language Models (LLMs) like `mistral` for conversational responses and `nomic-embed-text` for generating document embeddings.
* **Multi-Query Retriever:** Enhances retrieval accuracy by generating multiple perspectives on the user's question to query the vector database more effectively.
* **Dockerized Deployment:** The entire application, including the Flask web server and Ollama, is containerized using Docker Compose for simplified setup and consistent execution across different environments. [cite: 10, 11]
* **Simple User Interface:** A basic, intuitive web interface allows for seamless PDF uploads and chatbot interactions. [cite: 8, 9]

## Architecture Overview

The application follows a typical RAG (Retrieval Augmented Generation) architecture, orchestrated by LangChain:

1.  **Frontend (HTML/CSS):** A simple `index.html` provides the user interface for uploading PDFs and submitting questions.
2.  **Backend (Flask - `app.py`):**
    * Handles HTTP requests (`/` for the UI, `/upload_pdf` for document processing).
    * Receives PDF files, saves them temporarily.
    * Uses `UnstructuredPDFLoader` to extract text from PDFs.
    * Splits text into manageable `chunks` using `RecursiveCharacterTextSplitter`.
    * Generates `OllamaEmbeddings` for these chunks (using `nomic-embed-text` model).
    * Stores the embeddings and document chunks in a `Chroma` vector database.
    * Initializes a `ChatOllama` instance (using `mistral` model) for generating responses.
    * Constructs a LangChain retrieval chain, incorporating a `MultiQueryRetriever` for improved context retrieval.
    * Processes user questions by retrieving relevant document snippets and feeding them, along with the question, to the LLM for generating answers.
3.  **Ollama Service:** Runs as a separate Docker container, providing the local inference server for LLMs and embeddings.
4.  **ChromaDB:** Used as a local vector store for indexing and retrieving document embeddings.

## Setup Instructions

These instructions will guide you through setting up and running the LangChain PDF Chatbot using Docker Compose.

### Prerequisites

* **Docker Desktop:** Ensure Docker Desktop is installed and running on your system (Windows, macOS, or Linux). This includes Docker Engine and Docker Compose.

### Clone the Repository

First, clone this repository to your local machine:

````bash
git clone https://github.com/SulemanTariqMughal/Chatbot_Applab.git
cd ChatbotApplab
Running with Docker Compose

Navigate to the root directory of the cloned project (where docker-compose.yaml and Dockerfile are located).

Build and Start Services:
This command will build the Docker images (if not already built), install dependencies, and start both the Flask application and Ollama services.

Bash
docker compose up --build
 The first time you run this, it will take some time to download base images and install dependencies.

Access the Application:
Once the services are up and running, you can access the Flask web application in your browser:

http://localhost:5001
 Stop the Services:
To stop and remove the running containers:

Bash
docker compose down
Ollama Model Download

The application uses mistral for the LLM and nomic-embed-text for embeddings. The app service will attempt to pull these models automatically from the Ollama service when it starts. However, for a smoother first run, you can pre-pull them using the Ollama service's exposed port.

Option 1 (Recommended - after docker compose up):
Once ollama-service is running (you'll see Listening on [::]:11434 in the logs), you can run the following commands in a new terminal to pull the models:

Bash
docker exec -it chatbotapplab-ollama-service-1 ollama pull mistral
docker exec -it chatbotapplab-ollama-service-1 ollama pull nomic-embed-text
(Replace chatbotapplab-ollama-service-1 with your actual Ollama container name if it differs, you can find it with docker ps).

Option 2 (Directly from host, if Ollama is accessible externally):
If Ollama's port 11434 is exposed on your host, you could also use the Ollama CLI directly from your host if you have it installed:

Bash
ollama pull mistral
ollama pull nomic-embed-text
API Documentation
The Flask application exposes the following endpoints:

Endpoint: /

Method: GET, POST
Description: Serves the main web interface for the chatbot. On POST, it handles PDF uploads and question submissions. 

Request (GET):
No parameters.
Request (POST - PDF Upload & Question):
Form Data (multipart/form-data):
pdf_file: The PDF file to upload.
question: The question string related to the PDF content.
Response:
GET: Renders the index.html template.
POST: Redirects back to the main page (/) after processing, displaying the chatbot's response.
HTTP 200 OK: Successful processing and response.
HTTP 500 Internal Server Error: An error occurred during PDF processing or LLM interaction.
Project Structure
ChatbotApplab/
├── app.py                  # Main Flask application logic
├── index.html              # Frontend HTML for the web interface
├── requirements.txt        # Python dependencies for the Flask app
├── Dockerfile              # Docker build instructions for the Flask app
├── docker-compose.yaml     # Defines and configures multi-container Docker application
└── uploads/                # Directory for temporary PDF storage (created by Dockerfile)
└── README.md               # This documentation file
Troubleshooting
libGL.so.1: cannot open shared object file error: This indicates missing system-level graphics libraries. Ensure libgl1-mesa-glx and poppler-utils (and potentially tesseract-ocr) are correctly installed in your Dockerfile via apt-get install. Remember to docker compose up --build after modifying the Dockerfile.
flask: executable file not found or python -m flask not found: Ensure Flask is in your requirements.txt and you've run docker compose up --build to re-install Python dependencies.
Ollama models not found / download errors: Verify the ollama-service container is running (docker ps) and that you've pulled the mistral and nomic-embed-text models as described in the Ollama Model Download section. Check Ollama container logs for more details.
500 Internal Server Error on /upload_pdf after libGL.so.1 fix: This often points to issues with unstructured not having its full dependencies for PDF parsing. Ensure unstructured[pdf] is in your requirements.txt and rebuild. Also, consider adding tesseract-ocr to your Dockerfile for scanned PDFs.
"Error processing PDF" in UI: Check the terminal logs of the app-1 container for the detailed Python traceback, which will pinpoint the exact line of code causing the issue.
