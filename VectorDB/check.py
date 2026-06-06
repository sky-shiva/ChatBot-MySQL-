import chromadb

# ── Change this to your actual vector DB folder path ──
VECTOR_DB_PATH = "D:\HACKATHONS and PROJECTS\Project ChatBot(SQL)\VectorDB"
# ─────────────────────────────────────────────────────

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

collections = client.list_collections()

if not collections:
    print("❌ No collections found in:", VECTOR_DB_PATH)
else:
    print(f"✅ Found {len(collections)} collection(s) in '{VECTOR_DB_PATH}':\n")
    for col in collections:
        c = client.get_collection(col.name)
        print(f"  📦 Name     : {col.name}")
        print(f"     Chunks   : {c.count()}")
        print(f"     Metadata : {col.metadata}")
        print()
    print("👆 Copy the Name above into COLLECTION_NAME in app.py")