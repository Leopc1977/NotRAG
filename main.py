from os import listdir, environ
from os.path import join
from spacy import load
from math import sqrt
from openai import OpenAI

# --- OpenAI API Configuration ---
API_KEY = environ.get("OPENAI_API_KEY", "")
BASE_URL = "http://127.0.0.1:8081/v1"

# --- Constants ---
THRESHOLD_SIMILARITY = 0.1  # Minimum similarity to consider a match
TOP_K = 3  # Number of top results to return
SPACY_MODEL = "en_core_web_md"  # SpaCy model used for embeddings
POS_FILTER = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "X"]  # Allowed POS tags
DATA_PATH = "./datas"  # Folder containing the documents
MAX_TOKENS = 300 # Maximum number of tokens the LLM should generate for an answer

# Load OPENAI client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Load the SpaCy NLP model
nlp = load(SPACY_MODEL)
# Get the list of files in the data directory
filenames = listdir(DATA_PATH)

# 1. Retrieve concepts
# Count total occurrences of each concept globally and per document
document_concepts = {
    file_name:{} for file_name in filenames
}
global_concept_count = {}

for filename in filenames:
    current_path = join(DATA_PATH, filename)
    with open(current_path, encoding="utf-8") as f:
        content = f.read()
        doc = nlp(content)

        for token in doc:
            if token.pos_ in POS_FILTER:
                # Update global concept count
                token_text = token.text.lower()
                global_concept_count[token_text] = global_concept_count.get(token_text, 0) + 1

                # Update document-specific concept count
                document_concepts[filename][token_text] = document_concepts[filename].get(token_text, 0) + 1

concept_list = list(global_concept_count.keys())

# Ensure all documents have the same concept keys
# Missing concepts are added with a count of 0
for filename, concepts in document_concepts.items():
    for concept in concept_list:
        concepts.setdefault(concept, 0)

# 2. Vectorize documents: normalize each concept by its global count
docs_vector = {
    file_name:[] for file_name in filenames
}

for filename, concepts in document_concepts.items():
    for concept in concept_list:
        docs_vector[filename].append(concepts[concept] / global_concept_count[concept])

# 3. Vectorize a question
question = "Which planet has the most moons?"
question_doc = nlp(question.lower())
concept_index = {concept: idx for idx, concept in enumerate(concept_list)}
question_vector = [0] * len(concept_list)

for token in question_doc:
    if token.pos_ in POS_FILTER and token.text in concept_index:
        question_vector[concept_index[token.text]] = 1

# 4. Compute cosine similarity between question and each document
similarities = {}

for filename, doc_vector in docs_vector.items():
    # Compute dot product
    dot = sum(d * q for d, q in zip(doc_vector, question_vector))

    # Compute norms
    norm_doc = sqrt(sum(x**2 for x in doc_vector))
    norm_question = sqrt(sum(y**2 for y in question_vector))

    # Avoid division by zero
    if norm_doc == 0 or norm_question == 0:
        cosine = 0.0
    else:
        cosine = dot / (norm_doc * norm_question)

    similarities[filename] = cosine

# 5. Filter documents by similarity threshold and select top results
# Keep documents above the threshold
filtered_by_threshold = {doc: score for doc, score in similarities.items() if score > THRESHOLD_SIMILARITY}

# Keep at most TOP_K documents by similarity
if len(filtered_by_threshold) <= TOP_K:
    top_documents = filtered_by_threshold
else:
    top_documents = dict(sorted(filtered_by_threshold.items(), key=lambda item: item[1], reverse=True)[:TOP_K])

# 6. Create prompt
prompt_lines = ["Context for answering the question:"]
for filename in top_documents.keys():
    file_path = join(DATA_PATH, filename)

    with open(file_path, encoding="utf-8") as f:
            content = f.read().strip()
            prompt_lines.append(f"- {filename}:\n{content}")

# Combine context and question into final prompt
prompt = "\n".join(prompt_lines) + f"\n\nQuestion: {question}"

# 7. Call LLM to get the answer
def call_llm(prompt, model="gpt-4", max_tokens=MAX_TOKENS):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content

answer = call_llm(prompt)
print(answer)