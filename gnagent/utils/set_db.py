import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

with open("/home/johannesm/all_tmp/new_docs.txt") as file:
    data = file.read()
    docs = json.loads(data)


db = Chroma(
    embedding_function=HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"trust_remote_code": True, "device": "cuda"}
    ),
    persist_directory="/home/johannesm/all_tmp/new_chroma_db",
)

chunk_size = 1

for i in tqdm(range(0, len(docs), chunk_size)):
    chunk = docs[i : i + chunk_size]
    metadatas = [{"source": f"Document {ind+1}"} for ind in range(i, i + len(chunk))]
    db.add_texts(texts=chunk, metadatas=metadatas)

db.persist()
