import os 
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#chunking function
def chunk_text(text,max_length=300)
    words = text.split()
    chunks = []

    current_chunk = []

    for word in words:
        current_chunk.append(word)

        if len(current_chunk) >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# load PDFs/TXT from data
def load_documents():
    docs = []
    for file in os.listdir(r"C:\LearnPy\personalChatBot\data"):
        path = os.path.join(r"C:\LearnPy\personalChatBot\data",file)

        if file.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""

            for page in reader.pages:
                text += page.extract_text()
               
            docs.extend(chunk_text(text))

        elif file.endswith(".txt"):
            with open(path,"r",encoding="utf-8") as f:
                text = f.read()
                text = text.replace("\n", " ")
                text = " ".join(text.split())
                docs.extend(chunk_text(text))
        
    return docs

documents = load_documents()

documents = documents[2:]


print(f"total chunks created : {len(documents)}")

#generate embeddings for each chunks
embeddings = model.encode(documents, convert_to_numpy = True)
faiss.normalize_L2(embeddings)

#save vector with faiss
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index,"vector_store.index")
np.save("documents.npy",np.array(documents))

print("Embeddings saved successfully")