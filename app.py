from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
import os
import sys
import json
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)

load_dotenv()

# openai
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
embeddings_model = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-small")

# firebase
firebase_credentials = json.loads(
    os.getenv("FIREBASE_CREDENTIALS").replace(
        "\\n", "\n"))
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://our-project-id.firebaseio.com/'
})

# pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone = Pinecone(api_key=pinecone_api_key)
index_name = "headstarter-prj-3"
if index_name not in pinecone.list_indexes().names():
    print(f"Error: Pinecone index '{index_name}' does not exist.")
    sys.exit(1)
index = pinecone.Index(index_name)


@app.route('/faq', methods=['POST'])
def handle_faq():
    messages = request.json.get('messages', [])
    query = create_pinecone_query(messages)
    response = handle_rag_query(query)
    return jsonify(response), 200


@app.route('/order-info', methods=['POST'])
def order_info():
    order_id = request.json.get('order_id')
    if order_id:
        order_info = get_order_info(order_id)
        return jsonify(order_info), 200
    else:
        return jsonify({'error': 'Order ID is required'}), 400


def get_order_info(order_id):
    # firebase realtime
    ref = db.reference('/orders')
    order_info = ref.child(order_id).get()
    return order_info if order_info else {'error': 'No order found'}


def handle_rag_query(query):
    embeddings = embed_query(query)
    top_k = 5
    query_results = index.query(embeddings, top_k=top_k)
    retrieved_texts = [doc['text'] for doc in query_results['matches']]
    combined_text = "FAQ database entries: \n" + " ".join(retrieved_texts)
    response = generate_response(query, combined_text)
    return {"response": response}


def embed_query(query):
    return embeddings_model.embed_text(query)


def create_pinecone_query(context):
    context = [{"role": "system", "content": "The following is a message exchange between an AI assistant and a user. Summarize it into a brief query that could be vectorized for Pinecone to find the relevant information for the user's last question. Please ONLY provide the query ready for vectorization."}] + context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=context
    )
    return response.choices[0].message.content


def generate_response(query, faq_texts):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "system", "content": "You should generate a very brief response according to the user questions as well as the FAQ database entries."},
            {"role": "system", "content": "If it is apparent that the user's question is unrelated to FAQ, please briefly prompt the user to only ask related questions."},
            {"role": "system", "content": faq_texts},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    app.run(debug=True)
