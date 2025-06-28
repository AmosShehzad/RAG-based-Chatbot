Title: RAG-Based Chatbot for Multi-PDF Question Answering with Word Cloud Visualization and Confidence Scoring

Overview:
This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDF documents, processes their content, and answers natural language questions based on the combined knowledge of the documents. The application uses a Gradio interface for interaction, PyMuPDF for extracting text from PDFs, SentenceTransformers for creating semantic embeddings, scikit-learn for similarity-based retrieval, and Hugging Face Transformers for answering the questions.

Unique Enhancements:

1. Word Cloud Preview (Visual Summary):
   Before asking questions, users are shown a word cloud generated from the content of all uploaded PDFs. This allows them to quickly understand the key topics present in the documents and helps guide meaningful queries.

2. Confidence Score for Answers:
   Each generated answer includes a confidence score based on the cosine similarity between the question and the retrieved chunks. This provides transparency about how well the chatbot understands and matches the question with relevant content from the PDFs.

Challenges Faced:
- FAISS library compatibility issues on Windows were resolved by switching to scikit-learn for vector similarity search.
- Ensuring that the word cloud was both readable and helpful required tuning of text preprocessing and visualization.
- Accurately interpreting similarity scores to compute confidence required normalization and tuning based on real test cases.


