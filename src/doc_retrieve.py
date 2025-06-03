from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_vectorstore(index_dir):
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


# list chunks in vectorstore
def list_chunks(index_dir, limit=10):
    vectorstore = load_vectorstore(index_dir)
    all_docs = vectorstore.docstore._dict 
    print(f"Total chunks: {len(all_docs)}\n")
    for i, (doc_id, doc) in enumerate(all_docs.items()):
        print(f"--- Chunk {i+1} [{doc.metadata.get('source')}] ---")
        print(doc.page_content + "\n")
        if i + 1 >= limit:
            break
        
        
# chunks retrieval
def search_documents(index_dir, query: str, k: int) -> list[Document]:
    vectorstore = load_vectorstore(index_dir)
    search_documents = vectorstore.similarity_search_with_score(query, k=k)
    docs = [doc for doc, _ in search_documents]
    scores = [score for _, score in search_documents]
    return docs, scores


# document retrieval
def retrieve_documents(index_dir, query: str, k: int = 15, params: dict = None) -> list[Document]:
    
    threshold = params.get("rag_dist_threshold", 0.8) if params else 0.8
    docs, scores = search_documents(index_dir, query, k)
    rag_content = []
    for i, doc in enumerate(docs):
        if scores[i] > threshold:
            break
        rag_content.append(doc)
        print(f"\n===== Result {i+1} ({len(doc.page_content)})=====")
        print(doc.page_content)
        print("Score (distance):", scores[i])
        
    if not rag_content:
        print("No relevant documents found within the distance threshold.")
        
    return rag_content

        