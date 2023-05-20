from flask import Flask, g, request, jsonify, session, redirect, url_for
import langchain
import openai
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS, Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import os 
from dotenv import load_dotenv
import json
import pickle

load_dotenv()


API_KEY = os.getenv('API_KEY')
print(API_KEY)
app = Flask(__name__)
pinecone.init(api_key="b243d58a-562a-46f7-86e9-e39326c39c48",environment="us-central1-gcp")

index = pinecone.Index("dcu")
app.secret_key = os.getenv("SECRET_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)


llm = OpenAI(openai_api_key=API_KEY)

chain = load_qa_chain(llm, chain_type="stuff")

def get_similiar_docs(index, query, k=2, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs


def get_answer(query,index, chain):
  similar_docs = get_similiar_docs(index, query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer



@app.route('/', methods=['GET'])
def home():
    ok = "Houston, there's a problem !"
    #session["docsearch"] = None

    if index and embeddings and llm and chain:
       ok = "it's Nominal !"

   # import pinecone

    return "You are in the API and {ok}".format(ok=ok)



@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # Get the PDF file from the POST request
    pdf_file = request.files.get('pdf_file')
    #prompt = request.form.get("prompt")

    reader = PdfReader(pdf_file)
    raw_text = ''
    for i, page in enumerate(reader.pages):
      text = page.extract_text()
      if text:
        raw_text += text

    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
)

    texts = text_splitter.create_documents([raw_text])

    #print(raw_text)

    context = Pinecone.from_documents(texts, embeddings, index_name="dcu")
    session['docsearch'] = pickle.dumps(context)



    #g.my_variable = context 
    
    # Process the PDF file
    # ... insert your code here ...flas
    
    # Return a response
    return "Done"
    #return redirect(url_for('process_text', my_var=context))


@app.route('/process_text', methods=['POST'])
def process_text():
    #print(g.my_variable)
    # Get the text from the POST request
    prompt = request.form.get('text')
    context = pickle.loads(session.get('docsearch', b''))
    #doc_obj = request.get_json("docsearch")
    # Process the text
    # ... insert your code here ...
    query = prompt
    answer = get_answer(query, context, chain)
    
    
    # Return a response
    return jsonify({'message': answer})

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
