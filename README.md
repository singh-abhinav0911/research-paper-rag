# 📄 Research Paper Summarizer

A RAG-based research paper summarizer built with Streamlit, ChromaDB, and LLaMA 3.3 70B via Groq.

## 🚀 Features
- Upload and process multiple research PDFs
- AI-generated summaries with customizable sections
- Multi-turn chat with your papers
- Side-by-side paper comparison
- Export summaries to Word and PDF

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **LLM**: LLaMA 3.3 70B via Groq
- **PDF parsing**: PyMuPDF

## ⚙️ Setup

1. Clone the repo
   git clone https://github.com/YOUR_USERNAME/research-paper-rag.git
   cd research-paper-rag

2. Install dependencies
   pip install -r requirements.txt

3. Add your Groq API key — create a .env file
   GROQ_API_KEY=your_key_here

4. Run the app
   streamlit run app.py

## 📸 Usage
1. Upload one or more research paper PDFs from the sidebar
2. Click Process All Papers
3. Generate summaries, chat with papers, or compare them side by side
