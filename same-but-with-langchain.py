from os import environ, listdir
from os.path import join
from langchain_core.documents import Document
from langchain_community.embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI

# --- OpenAI API Configuration ---
LLM_BASE_URL = "http://127.0.0.1:8081/v1"
LLM_API_KEY =  environ.get("OPENAI_API_KEY", "dummy_key_for_local_llm")
MODEL_NAME = "deepseek-llm-7b-chat.Q4_K_M"

# --- Constants ---
DATA_PATH = "./datas"            # Folder containing the documents
SPACY_MODEL = "en_core_web_md"   # SpaCy model used for embeddings
TOP_K = 3                        # Number of top results to include
MAX_TOKENS = 300                 # Max tokens LLM should generate

QUESTION = "Which planet has the most moons?"

# 1: Load documents
filenames = listdir(DATA_PATH)
docs = []

for filename in filenames:
    with open(join(DATA_PATH, filename), encoding="utf-8") as f:
        content = f.read()
        docs.append(Document(page_content=content, metadata={"source": filename}))

# 2. Create embeddings (main.py: Step 1 & 2)
embeddings = SpacyEmbeddings(model_name=SPACY_MODEL)

# A vectorstore is just a collection of document embeddings that allows fast similarity search.
# Compared to main.py, it replaces the manual concept counting, vectorization, 
# and cosine similarity computation with an optimized structure that can quickly retrieve 
# the top-k most relevant documents for a given query.
vectorstore = FAISS.from_documents(docs, embeddings)  

# 4: Retrieve top documents (main.py: Step 3, 4 & 5)
top_docs = vectorstore.similarity_search(QUESTION, k=TOP_K)

# 5: Create context prompt (main.py: Step 6)
context = "Context for answering the question:"
for doc in top_docs:
    context += f"\n- {doc.metadata['source']}:\n{doc.page_content}"

prompt = f"{context}\n\nQuestion: {QUESTION}"
print("==== PROMPT ====")
print(prompt)

# 6. Call LLM (main.py: step 7)
llm = OpenAI(
    model_name=MODEL_NAME,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    max_tokens=MAX_TOKENS
)

answer = llm.invoke(prompt)

print("\n==== ANSWER ====")
print(answer)
