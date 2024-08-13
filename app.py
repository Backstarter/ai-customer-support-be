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
cred_path = os.getenv("FIREBASE_CRED_PATH")
url = os.getenv("FIREBASE_URL")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': url
})

# pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone = Pinecone(api_key=pinecone_api_key)
index_name = "headstarter-prj-3"
if index_name not in pinecone.list_indexes().names():
    print(f"Error: Pinecone index '{index_name}' does not exist.")
    sys.exit(1)
index = pinecone.Index(index_name)


@app.route('/hello', methods=['POST'])
def hello():
    return jsonify({'hello': 'world'}), 200


@app.route('/faq', methods=['POST'])
def handle_faq():
    messages = request.json.get('messages', [])
    query = create_pinecone_query(messages)
    response = handle_rag_query(query)
    if isinstance(response, dict) and 'error' in response:
        return jsonify(response), 400
    return jsonify(response), 200


@app.route('/order-info', methods=['POST'])
def order_info():
    order_id = request.json.get('order_id')
    if order_id and order_id.isdigit() and len(order_id) == 5:
        order_response = get_order_info(order_id)
        if isinstance(order_response, dict) and 'error' in order_response:
            return jsonify(order_response), 404
        return jsonify({'message': order_response}), 200
    else:
        return jsonify({'error': 'Order ID is required'}), 400


def get_order_info(order_id):
    ref = db.reference('/orders')
    order_info = ref.child(order_id).get()
    if order_info:
        status = order_info.get('delivery status', 'No status available')
        item = order_info.get('item', 'an item')
        return f"Your order for {item} has been {status}."
    else:
        return {'error': 'No order found'}


def handle_rag_query(query):
    try:
        embeddings = embed_query(query)
        top_k = 5
        query_results = index.query(embeddings, top_k=top_k)
        retrieved_texts = [doc['text'] for doc in query_results['matches']]
        combined_text = "FAQ database entries: \n" + " ".join(retrieved_texts)
        response = generate_response(query, combined_text)
        return {"message": response}
    except Exception as e:
        print(e)
        return {'error': 'OpenAI API error'}


def embed_query(query):
    try:
        return embeddings_model.embed_text(query)
    except Exception as e:
        print(e)
        raise Exception("OpenAI API error")


def create_pinecone_query(context):
    try:
        context = [{
            "role": "system",
            "content": "The following is a message exchange between an AI assistant and a user. Summarize it into a brief query that could be vectorized for Pinecone to find the relevant information for the user's last question. Please ONLY provide the query ready for vectorization."
        }] + context
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        raise Exception("OpenAI API error")


def generate_response(query, faq_texts):
    try:
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
    except Exception as e:
        print(e)
        raise Exception("OpenAI API error")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

