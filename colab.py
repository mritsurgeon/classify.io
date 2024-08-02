##Colab Basic Classify.io 
import os
import uuid
from tika import parser
import spacy
from collections import defaultdict
import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import chromadb
from gliner_spacy.pipeline import GlinerSpacy
import time
import torch

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

# Initialize spaCy and add GlinerSpacy
nlp = spacy.blank("en")
nlp.max_length = 2000000
custom_spacy_config = {
    "gliner_model": "urchade/gliner_multi_pii-v1",
    "chunk_size": 250,
    "labels": ["person", "organization", "phone number", "address", "passport number", "email",
               "credit card number", "social security number", "health insurance id number",
               "date of birth", "mobile phone number", "bank account number"],
    "style": "ent",
    "threshold": 0.3,
    "map_location": device
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

def process_file(uploaded_file):
    file_data = defaultdict(list)
    entity_counts = defaultdict(int)
    start_time = time.time()
    with st.spinner('Processing file...'):
        try:
            parsed = parser.from_buffer(uploaded_file.read())
            if not parsed['content']:
                return "No content found in the file."
            if len(parsed['content']) > 10000000:
                st.warning(f"File is too large to process fully. Processing first 10 million characters.")
                parsed['content'] = parsed['content'][:10000000]
            entities = detect_entities(parsed['content'])
            metadata = parsed['metadata']
            for _, label in entities:
                entity_counts[label] += 1
            entities_dict = defaultdict(list)
            for entity, label in entities:
                entities_dict[label].append(entity)
            file_data[uploaded_file.name].append({
                'content': parsed['content'],
                'entities': entities_dict,
                'metadata': metadata
            })
        except Exception as e:
            return f"Error parsing file: {str(e)}"
    end_time = time.time()
    processing_time = end_time - start_time
    st.success(f"Processing complete. Total time: {processing_time:.2f} seconds")
    return file_data, entity_counts

def create_colored_label(label):
    label_colors = {
        "PERSON": ("#FF4136", "#FFEEEE"),
        "LOCATION": ("#2ECC40", "#EEFFEE"),
        "ORGANIZATION": ("#0074D9", "#EEF6FF"),
        "PHONE NUMBER": ("#FF851B", "#FFF6EE"),
        "ADDRESS": ("#B10DC9", "#F9EEFF"),
        "EMAIL": ("#39CCCC", "#EEFFFF"),
        "PASSPORT NUMBER": ("#FFDC00", "#FFFFEE"),
        "CREDIT CARD NUMBER": ("#F012BE", "#FFEEFF"),
        "SOCIAL SECURITY NUMBER": ("#3D9970", "#EEFFF5"),
        "HEALTH INSURANCE ID NUMBER": ("#85144b", "#FFEEF5"),
        "DATE OF BIRTH": ("#7FDBFF", "#EEF9FF"),
        "MOBILE PHONE NUMBER": ("#01FF70", "#EEFFEE"),
        "BANK ACCOUNT NUMBER": ("#001f3f", "#EEF0F5")
    }
    label_color, bg_color = label_colors.get(label.upper(), ("#AAAAAA", "#F5F5F5"))
    label_style = f'background-color: {label_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;'
    entity_style = f'background-color: {bg_color};'
    return label_style, entity_style, label.upper()

# Streamlit UI
st.title('Data Classification Tool')
st.info(f"Using device: {device}")

uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'doc', 'docx'])

if uploaded_file is not None:
    result = process_file(uploaded_file)
    if isinstance(result, str):
        st.warning(result)
    else:
        file_data, entity_counts = result
        st.session_state.file_data = file_data
        st.session_state.entity_counts = entity_counts

        st.subheader('Counts of each PII Entity Type Detected')
        entity_counts = dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True))
        fig = px.bar(x=list(entity_counts.values()), y=list(entity_counts.keys()), orientation='h', title='Counts of each PII Entity Type Detected')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Detected Entities')
        col1, col2 = st.columns(2)
        for filename, entities_list in file_data.items():
            for entity_data in entities_list:
                for i, (label, entities) in enumerate(entity_data['entities'].items()):
                    column = col1 if i % 2 == 0 else col2
                    with column:
                        for entity in entities:
                            label_style, entity_style, label_text = create_colored_label(label)
                            entity_lines = entity.split('\n')
                            entity_html = '<br>'.join([f'<span style="{entity_style}">{line}</span>' for line in entity_lines])
                            st.markdown(f'{entity_html} <span style="{label_style}">{label_text}</span>', unsafe_allow_html=True)
                        st.write("")  # Add a small gap between entity types
