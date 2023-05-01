from flask import Flask, request, jsonify
import langchain
import openai
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os 

API_KEY = os.environ.get('API_KEY')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # Get the PDF file from the POST request
 
    # Process the PDF file
    # ... insert your code here ...flas

    
    
    # Return a response
    return "You are in the API"


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # Get the PDF file from the POST request
    pdf_file = request.files.get('pdf_file')
    prompt = request.form.get("prompt")

    reader = PdfReader(pdf_file)
    raw_text = ''
    for i, page in enumerate(reader.pages):
      text = page.extract_text()
      if text:
        raw_text += text

    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

    text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
    texts = text_splitter.split_text(raw_text)
    #print(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(openai_api_key=API_KEY), chain_type="stuff")

    query = prompt
    docs = docsearch.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    
    # Process the PDF file
    # ... insert your code here ...flas
    
    # Return a response
    return jsonify({'response': response})

@app.route('/process_text', methods=['POST'])
def process_text():
    # Get the text from the POST request
    text = request.form.get('text')
    
    # Process the text
    # ... insert your code here ...
    
    # Return a response
    return jsonify({'message': 'Text processed successfully!'})

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
