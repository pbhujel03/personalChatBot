import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

#load the embedding 
model = SentenceTransformer("all-miniLM-L6-v2")

#load the saved documents and Faiss Index
documents = np.load("documents.npy",allow_pickle=True)
index = faiss.read_index("vector_store.index")

# ask questions
while True:
    query = input("\n Ask anything from the notes.(type exit to quit):")
    if query.lower()=="exit":
        break

    #convert the query into embedding 
    query_embedding = model.encode([query],convert_to_numpy=True)

    #search Faiss index
    distances, indices = index.search(query_embedding,3)

    #show the closest answers
    closest_index = indices[0][0]

    print("\nAnswers:")
    print(documents[closest_index][:300],"...")



