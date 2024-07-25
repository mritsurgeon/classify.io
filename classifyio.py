import os
import re
import uuid
from tika import parser
import spacy
from collections import defaultdict
import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
from sentence_transformers import SentenceTransformer, util
import chromadb
import spacy
from gliner_spacy.pipeline import GlinerSpacy
import time


# Load spaCy model
#nlp = spacy.load("en_core_web_sm")
nlp = spacy.blank("en")
nlp.max_length = 2000000

custom_spacy_config = {
    "gliner_model": "urchade/gliner_multi_pii-v1",
    "chunk_size": 250,
    "labels": ["person", "organization", "phone number", "address", "passport number", "email", 
    "credit card number", "social security number", "health insurance id number", 
    "date of birth", "mobile phone number", "bank account number", "medication", "cpf", 
    "driver's license number", "tax identification number", "medical condition", 
    "identity card number", "national id number", "ip address", "email address", "iban", 
    "credit card expiration date", "username", "health insurance number", "registration number", 
    "student id number", "insurance number", "flight number", "landline phone number", 
    "blood type", "cvv", "reservation number", "digital signature", "social media handle", 
    "license plate number", "cnpj", "postal code", "passport_number", "serial number", 
    "vehicle registration number", "credit card brand", "fax number", "visa number", 
    "insurance company", "identity document number", "transaction number", 
    "national health insurance number", "cvc", "birth certificate number", "train ticket number", 
    "passport expiration date", "social_security_number"],
    "style": "ent",
    "threshold": 0.3,
    "map_location": "cpu" 
}

nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

def detect_entities(text):
    if len(text) > nlp.max_length:
        chunks = [text[i:i+nlp.max_length] for i in range(0, len(text), nlp.max_length)]
        entities = []
        for chunk in chunks:
            doc = nlp(chunk)
            entities.extend([(ent.text, ent.label_) for ent in doc.ents])
    else:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Directory containing the files
directory = 'C:\\Data\\demo'

embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(settings=chromadb.Settings(allow_reset=True))

def traverse_directories(directory):
    if not os.listdir(directory):
        return "No files found in the directory. Please mount a restore point."

    total_files = sum([len(files) for _, _, files in os.walk(directory)])
    file_data = defaultdict(list)
    entity_counts = defaultdict(int)
    
    start_time = time.time()
    
    with st.spinner('Processing files...'):
        progress_bar = st.progress(0)
        file_counter = st.empty()
        count = 0
    
        for root, _, files in os.walk(directory):
            for filename in files:
                count += 1
                filepath = os.path.join(root, filename)
                file_counter.text(f"Processing file {count} of {total_files}")
                try:
                    parsed = parser.from_file(filepath)
                    if not parsed['content']:
                        continue
                    
                    # Check if the content is too large
                    if len(parsed['content']) > 10000000:  # 10 million characters
                        st.warning(f"File {filename} is too large to process fully. Processing first 10 million characters.")
                        parsed['content'] = parsed['content'][:10000000]
                    
                    entities = detect_entities(parsed['content'])
                    metadata = parsed['metadata']
                    progress_bar.progress(count / total_files)

                    for _, label in entities:
                        entity_counts[label] += 1

                    entities_dict = defaultdict(list)
                    for entity, label in entities:
                        entities_dict[label].append(entity)

                    file_data[filename].append({
                        'filepath': filepath,
                        'content': parsed['content'],
                        'entities': entities_dict,
                        'metadata': metadata
                    })

                except Exception as e:
                    st.error(f"Error parsing {filepath}: {str(e)}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    st.success(f"Processing complete. Total time: {processing_time:.2f} seconds")

    return file_data, entity_counts, total_files

def prepare_data_for_llm(file_data):
    for filename, entities_list in file_data.items():
        for entity_data in entities_list:
            chunks = split_text(entity_data['content'])
            embeddings = vectorize_chunks(chunks)
            collection = store_embeddings(chroma_client, chunks, embeddings)
    st.session_state.collection = collection

def split_text(text, max_chunk_size=512):
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        chunk.append(word)
        if len(chunk) >= max_chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def vectorize_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

def store_embeddings(chroma_client, chunks, embeddings):
    collection_name = "text_embeddings"
    existing_collections = chroma_client.list_collections()
    if collection_name in existing_collections:
        st.info("Deleting existing collection...")
        chroma_client.delete_collection(name=collection_name)
        st.info("Collection deleted.")

    
    collection = chroma_client.get_or_create_collection(collection_name, metadata={"key": "value"})

    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    metadatas = [{"text": chunk} for chunk in chunks]

    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
    st.session_state.collection = collection
    
    return collection

def escape_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def highlight_text(text, entities):
    text = escape_html(text)
    for label, words in entities.items():
        for word in words:
            description = label  # Default description is the label itself
            if label == "PERSON":
                description = "Person's name"
            elif label == "ORG":
                description = "Organization"
            elif label == "DATE":
                description = "Date"
            elif label == "TIME":
                description = "Time"
            elif label == "CARDINAL":
                description = "Number"
            elif label == "GPE":
                description = "Geopolitical Entity"
            elif label == "MONEY":
                description = "Monetary value"
            elif label == "NORP":
                description = "Nationalities, religious or political groups"
            elif label == "LOC":
                description = "Location"

            pattern = re.compile(r'\b{}\b'.format(re.escape(word)), flags=re.IGNORECASE)
            text = pattern.sub(f"<mark class='entity-{label.lower()}' title='{description}'>{word}</mark>", text)

    return text

def get_bearer_token(veeam_server, veeam_username, veeam_password):
    # Implement actual authentication logic
    return "dummy_bearer_token" 

def get_backups(veeam_server, bearer_token):
    # Implement actual API call to get backups
    return [{"id": 1, "name": "Backup 1"}, {"id": 2, "name": "Backup 2"}]

def get_restore_points(veeam_server, bearer_token, backup_id):
    # Implement actual API call to get restore points
    return [{"id": 1, "name": "Restore Point 1"}, {"id": 2, "name": "Restore Point 2"}]

def mount_restore_point(veeam_server, bearer_token, restore_point_id):
    # Implement actual mount logic
    return {"success": True, "mount_path": f"/path/to/mounted/restore/point/{restore_point_id}"}

st.markdown(
    """
    <style>
        mark.entity-person {
            background-color: yellow;
        }
        mark.entity-org {
            background-color: cyan;
        }
        mark.entity-date {
            background-color: green;
        }
        mark.entity-time {
            background-color: orange;
        }
        mark.entity-cardinal {
            background-color: lightblue;
        }
        mark.entity-gpe {
            background-color: lightgreen;
        }
        mark.entity-money {
            background-color: lightyellow;
        }
        mark.entity-norp {
            background-color: lightpink;
        }
        mark.entity-loc {
            background-color: lightsalmon;
        }
        .app-name {
            font-size: 36px;
            color: #FF5733;
            padding-bottom: 20px;
            border-bottom: 2px solid #FF5733;
            margin-bottom: 20px;
        }
        .file-content {
            white-space: pre-wrap;
            line-height: 1.6;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        .file-info {
            margin-bottom: 20px;
        }
        .file-info h4 {
            margin-bottom: 10px;
        }
        .file-info .entity-list {
            margin-top: 5px;
        }
        .entity-highlight {
            background-color: red;
            color: black;
            padding: 2px 4px;
        }
        .entity-counts {
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<p class='app-name'>Classify.io</p>", unsafe_allow_html=True)
st.subheader('Data Classification Tool')
st.text('Written By Ian Engelbrecht')

with st.expander("Veeam Backup Server Configuration"):
    veeam_server = st.text_input("Veeam Backup Server IP/FQDN")
    veeam_username = st.text_input("Username")
    veeam_password = st.text_input("Password", type="password")

    if st.button("Authenticate"):
        bearer_token = get_bearer_token(veeam_server, veeam_username, veeam_password)
        if bearer_token:
            st.success("Authentication successful")
            st.session_state.bearer_token = bearer_token
        else:
            st.error("Failed to authenticate. Please check your credentials and try again.")

if 'bearer_token' in st.session_state:
    selected_backup = st.selectbox("Select a backup", get_backups(veeam_server, st.session_state.bearer_token))
    selected_restore_point = st.selectbox("Select a restore point", get_restore_points(veeam_server, st.session_state.bearer_token, selected_backup['id']))

    if st.button("Mount Restore Point"):
        mount_info = mount_restore_point(veeam_server, st.session_state.bearer_token, selected_restore_point['id'])
        if mount_info['success']:
            st.success(f"Restore point mounted at {mount_info['mount_path']}")
            st.session_state.mount_info = mount_info
        else:
            st.error("Failed to mount restore point")

if st.button("Start Analysis"):
    directory_to_traverse = st.session_state.mount_info['mount_path'] if 'mount_info' in st.session_state and st.session_state.mount_info['success'] else directory
    file_data, entity_counts, total_files = traverse_directories(directory_to_traverse)
    if isinstance(file_data, str):
        st.warning(file_data)
    else:
        st.session_state.file_data = file_data
        st.session_state.entity_counts = entity_counts
        st.session_state.total_files = total_files

        st.subheader('Counts of each PII Entity Type Detected')
        entity_counts = dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True))
        fig = px.bar(x=list(entity_counts.values()), y=list(entity_counts.keys()), orientation='h', title='Counts of each PII Entity Type Detected')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Files Matched to Entities')
        for filename, entities_list in file_data.items():
            with st.expander(f"File: {filename}"):
                for entity_data in entities_list:
                    st.markdown("<div class='file-info'>", unsafe_allow_html=True)
                    st.write("### Content:")
                    st.markdown(f"<div class='file-content'>{highlight_text(entity_data['content'], entity_data['entities'])}</div>", unsafe_allow_html=True)
                    st.write("### Entities:")
                    for label, entities in entity_data['entities'].items():
                        st.write(f"###### {label}:")
                        st.markdown("<ul class='entity-list'>", unsafe_allow_html=True)
                        for entity in entities:
                            st.markdown(f"<li><span class='entity-highlight'>{entity}</span></li>", unsafe_allow_html=True)
                        st.markdown("</ul>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        # Prepare data for LLM
        prepare_data_for_llm(file_data)

def retrieve_relevant_chunks(user_query, k=10):
    if 'collection' not in st.session_state:
        st.error("Collection not found. Please run the analysis first.")
        return [], [], []

    collection = st.session_state.collection
    query_embedding = embedding_model.encode([user_query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    relevant_chunks = results['documents'][0] if 'documents' in results and results['documents'] else []
    relevant_metadata = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else []
    distances = results['distances'][0] if 'distances' in results and results['distances'] else []
    
    return relevant_chunks, relevant_metadata, distances
    
def generate_response(user_query, relevant_chunks, relevant_metadata, distances):
    input_text = f"Query: {user_query}\n\n"
    if relevant_chunks:
        input_text += "Relevant Information:\n"
        for i, (chunk, metadata, distance) in enumerate(zip(relevant_chunks, relevant_metadata, distances)):
            input_text += f"Document {i+1} (Relevance: {1 - distance:.4f}):\n{chunk}\n"
            if metadata:
                input_text += f"Metadata: {metadata}\n"
            input_text += "\n"
    else:
        input_text += "No relevant information found.\n\n"

    input_text += "Based on the above information, please answer the following question:\n"
    input_text += user_query

    response = ollama.chat(model='tinyllama', messages=[
        {'role': 'system', 'content': 'You are an AI assistant specializing in data regulation and classification. Your role is to analyze the given information and respond to queries with a focus on data types, sensitivity, and applicable regulations. Provide concise, factual responses based solely on the given context. Identify specific data types if present, mention relevant regulations only if applicable, and do not speculate beyond the provided information. If the query cannot be answered based on the given context, state this clearly.'},
        {'role': 'user', 'content': input_text}
    ])
    final_response = response['message']['content']
    
    return final_response
    
    return final_response

if 'file_data' in st.session_state:
    with st.sidebar:
        st.subheader('RAG Chatbot')
        user_query = st.text_input("Ask a question about the data:")

        if st.button("Get Answer"):
            if user_query:
                start_time = time.time()
                
                with st.spinner('Retrieving relevant chunks...'):
                    retrieval_start = time.time()
                    relevant_chunks, relevant_metadata, distances = retrieve_relevant_chunks(user_query)
                    retrieval_end = time.time()
                    retrieval_time = retrieval_end - retrieval_start
                
                with st.spinner('Generating response...'):
                    generation_start = time.time()
                    response = generate_response(user_query, relevant_chunks, relevant_metadata, distances)
                    generation_end = time.time()
                    generation_time = generation_end - generation_start
                
                end_time = time.time()
                total_time = end_time - start_time
                
                st.success(f"Response: {response}")
                st.info(f"Total processing time: {total_time:.2f} seconds")
                st.info(f"Chunk retrieval time: {retrieval_time:.2f} seconds")
                st.info(f"Response generation time: {generation_time:.2f} seconds")
            else:
                st.warning("Please enter a question.")
