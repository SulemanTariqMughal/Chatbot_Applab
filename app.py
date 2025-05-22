
# from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename
# import os
# import uuid 
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Dictionary to store vector_db instances, keyed by session_id
# # In a real application, you might persist this or use a proper session management system
# session_vector_dbs = {}

# # Initialize model (these can be global as they don't depend on the PDF)
# local_model = 'llama3.2'
# llm = ChatOllama(model=local_model)
# ollama_embeddings = OllamaEmbeddings(model='nomic-embed-text', show_progress=True)

# # Prompt template for multi-query (global as it's static)
# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template='''You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}'''
# )

# # RAG prompt (global as it's static)
# template = '''Answer the question based ONLY on the following context:
# {context}
# Question: {question}'''
# rag_prompt = ChatPromptTemplate.from_template(template)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload_pdf', methods=['POST'])
# def upload_pdf():
#     if 'pdf_file' not in request.files:
#         return jsonify({"error": "No PDF file provided"}), 400

#     pdf_file = request.files['pdf_file']
#     if pdf_file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     session_id = str(uuid.uuid4()) # Generate a unique session ID

#     filename = secure_filename(pdf_file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     pdf_file.save(file_path)

#     try:
#         # Load and chunk the PDF
#         loader = UnstructuredPDFLoader(file_path=file_path)
#         data = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#         chunks = text_splitter.split_documents(data)

#         # Create and store the Chroma vector_db for this session
#         vector_db = Chroma.from_documents(
#             documents=chunks,
#             embedding=ollama_embeddings,
#             collection_name=f"local_rag_{session_id}" # Use session_id for unique collection
#         )
#         session_vector_dbs[session_id] = vector_db

#         # Clean up the uploaded PDF file after processing
#         os.remove(file_path)

#         return jsonify({"message": "PDF processed successfully", "session_id": session_id}), 200
#     except Exception as e:
#         # Clean up file even if processing fails
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

# @app.route('/ask_question', methods=['POST'])
# def ask_question():
#     data = request.get_json()
#     question = data.get('question')
#     session_id = data.get('session_id')

#     if not question or not session_id:
#         return jsonify({"error": "Question and session ID are required"}), 400

#     # Retrieve the vector_db for the given session_id
#     vector_db = session_vector_dbs.get(session_id)

#     if not vector_db:
#         return jsonify({"error": "No PDF processed for this session ID. Please upload a PDF first."}), 404

#     try:
#         retriever = MultiQueryRetriever.from_llm(
#             vector_db.as_retriever(),
#             llm,
#             prompt=QUERY_PROMPT
#         )

#         chain = (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | rag_prompt # Use the renamed prompt variable
#             | llm
#             | StrOutputParser()
#         )

#         response = chain.invoke(question)
#         return jsonify({"answer": response}), 200

#     except Exception as e:
#         return jsonify({"error": f"Error getting answer: {str(e)}"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import uuid  # Import uuid for generating unique session IDs

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to store vector_db instances, keyed by session_id
# In a real application, you might persist this or use a proper session management system
session_vector_dbs = {}

# Initialize model (these can be global as they don't depend on the PDF)
local_model = 'llama3.2'
# --- IMPORTANT CHANGE HERE ---
# Use the service name 'ollama-service' as the hostname for communication within Docker Compose
llm = ChatOllama(model=local_model, base_url="http://ollama-service:11434")
ollama_embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://ollama-service:11434", show_progress=True)

# Prompt template for multi-query (global as it's static)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template='''You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}'''
)

# RAG prompt (global as it's static)
template = '''Answer the question based ONLY on the following context:
{context}
Question: {question}'''
rag_prompt = ChatPromptTemplate.from_template(template) # Renamed 'prompt' to 'rag_prompt' for clarity

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    session_id = str(uuid.uuid4()) # Generate a unique session ID

    filename = secure_filename(pdf_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(file_path)

    try:
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=ollama_embeddings, # Use the globally initialized embeddings
            collection_name=f"local_rag_{session_id}" # Use session_id for unique collection
        )
        session_vector_dbs[session_id] = vector_db

        # Clean up the uploaded PDF file after processing
        os.remove(file_path)

        return jsonify({"message": "PDF processed successfully", "session_id": session_id}), 200
    except Exception as e:
        # Clean up file even if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id')

    if not question or not session_id:
        return jsonify({"error": "Question and session ID are required"}), 400

    # Retrieve the vector_db for the given session_id
    vector_db = session_vector_dbs.get(session_id)

    if not vector_db:
        return jsonify({"error": "No PDF processed for this session ID. Please upload a PDF first."}), 404

    try:
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt # Use the renamed prompt variable
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
        return jsonify({"answer": response}), 200

    except Exception as e:
        return jsonify({"error": f"Error getting answer: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)