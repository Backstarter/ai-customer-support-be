# load
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data.txt")
text_documents = loader.load()

# chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
chunk_data = text_splitter.split_documents(text_documents)
print(len(chunk_data))

#embeddings
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

embeddings = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-small"
    )

#create index
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)
index_name = "headstarter-prj-3"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension=1536, # since using openAI text-embedding-ada-002
        metric="cosine",
        spec=ServerlessSpec(cloud = "aws", region="us-east-1")
    )

# #Embed and store
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore.from_documents(
    chunk_data, 
    embeddings, 
    index_name=index_name)
