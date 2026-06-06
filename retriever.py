from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectorstore = Chroma(
    persist_directory="VectorDB",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

def retrieve_docs(query):

    docs = retriever.invoke(query)

    return "\n\n".join([doc.page_content for doc in docs])