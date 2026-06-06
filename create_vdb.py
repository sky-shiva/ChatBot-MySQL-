
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------
# LOAD OCR TEXT
# -----------------------------------

with open(
    r"D:\HACKATHONS and PROJECTS\Project ChatBot(SQL)\sqlclaude.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

print("\nTEXT LOADED SUCCESSFULLY!")

# -----------------------------------
# CHUNKING
# -----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print(f"\nTOTAL CHUNKS: {len(chunks)}")

# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEMBEDDING MODEL LOADED!")

# -----------------------------------
# CREATE VECTOR DB
# -----------------------------------

vectordb = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="VectorDB"
)

print("\nVECTOR DB CREATED SUCCESSFULLY!")

# -----------------------------------
# VERIFY STORAGE
# -----------------------------------

print("\nTOTAL DOCS STORED:")
print(vectordb._collection.count())
