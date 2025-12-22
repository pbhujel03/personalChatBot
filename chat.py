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
    raw_query = input("\n Ask anything from the notes.(type exit to quit):")
    query = f"Explain the {raw_query} described in the document"
    if query.lower()=="exit":
        break

    #convert the query into embedding 
    query_embedding = model.encode([query],convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    #search Faiss index
    distances, indices = index.search(query_embedding,3)

    #show the closest answers
    # closest_index = indices[0][0]

    # print("\nAnswers:")
    # print(documents[closest_index][:300],"...")

    print("\nTop 5 matches(DEBUG):\n")
    for rank, idx in enumerate(indices[0]):
        print(f"Rank{rank+1}(chunk {idx}):")
        print(documents[idx][:300])
        print("-"*60)





