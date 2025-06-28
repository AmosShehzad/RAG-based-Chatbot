import gradio as gr
import fitz  # PyMuPDF
import os
import re
from io import BytesIO
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Load models
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="deepset/tinyroberta-squad2")

# Globals
all_chunks = []
chunk_sources = []
chunk_embeddings = None
combined_text = ""

def extract_text_from_pdfs(pdf_files):
    global combined_text, chunk_sources
    combined_text = ""
    texts = []
    chunk_sources = []

    for file in pdf_files:
        doc = fitz.open(file.name)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                texts.append((text, f"{os.path.basename(file.name)} - Page {i+1}"))
                combined_text += " " + text
    return texts

def split_and_embed(texts_with_sources):
    global all_chunks, chunk_sources, chunk_embeddings
    all_chunks = []
    chunk_sources = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for text, source in texts_with_sources:
        docs = splitter.create_documents([text])
        for doc in docs:
            all_chunks.append(doc.page_content)
            chunk_sources.append(source)

    if all_chunks:
        chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    else:
        chunk_embeddings = None

def generate_wordcloud():
    global combined_text
    if not combined_text.strip():
        return None

    cleaned = re.sub(r"[^a-zA-Z\s]", "", combined_text.lower())
    word_freq = Counter(cleaned.split())
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def answer_question(question):
    global all_chunks, chunk_sources, chunk_embeddings
    if not all_chunks or chunk_embeddings is None:
        return "Please upload and index PDFs first."

    q_emb = embed_model.encode([question], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, chunk_embeddings)[0]

    threshold = 0.5  # similarity threshold to filter relevant chunks
    above_thresh_idx = [i for i, sim in enumerate(sims) if sim > threshold]

    if not above_thresh_idx:
        return "No relevant content found in the PDFs for your question."

    # Sort by similarity descending
    above_thresh_idx.sort(key=lambda i: sims[i], reverse=True)

    max_context_chars = 2000
    context_chunks = []
    total_chars = 0
    for i in above_thresh_idx:
        chunk_len = len(all_chunks[i])
        if total_chars + chunk_len > max_context_chars:
            break
        context_chunks.append(all_chunks[i])
        total_chars += chunk_len

    context = "\n\n".join(context_chunks)
    if not context.strip():
        return "No sufficient content to answer the question."

    try:
        result = qa_pipeline(question=question, context=context)
        answer = result.get("answer", "No answer found.")
    except Exception:
        return "Error generating answer from the model."

    sources = "\n".join(set(chunk_sources[i] for i in above_thresh_idx[:len(context_chunks)]))
    confidence = np.mean([sims[i] for i in above_thresh_idx[:len(context_chunks)]]) * 100
    return f"**Answer:** {answer}\n\n**Sources:**\n{sources}\n\n**Confidence:** {confidence:.2f}%"

with gr.Blocks() as demo:
    gr.Markdown("# PDF Chatbot")
    gr.Markdown("Upload PDFs, extract text, then ask questions.")

    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], file_count="multiple")
        extract_btn = gr.Button("Extract & Index")
        wc_img = gr.Image(label="Word Cloud")

    with gr.Row():
        question_input = gr.Textbox(lines=2, placeholder="Ask your question here...")
        ask_btn = gr.Button("Get Answer")
        answer_out = gr.Markdown()

    def extract_and_show_wordcloud(files):
        texts = extract_text_from_pdfs(files)
        split_and_embed(texts)
        return generate_wordcloud()

    extract_btn.click(extract_and_show_wordcloud, inputs=[pdf_input], outputs=[wc_img])
    ask_btn.click(answer_question, inputs=[question_input], outputs=[answer_out])

demo.launch()
