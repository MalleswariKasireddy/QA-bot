# 📄 PDF Q&A Bot — Agentic AI with RAG + LangChain

> Chat with any PDF using AI. Upload a document, ask questions in natural language, and get accurate answers grounded in the document content.

Built by **Malliswari Kasireddy** · [LinkedIn](https://www.linkedin.com/in/malliswari-kasireddy-90b10521b/)

---

## 🚀 Demo

![Demo GIF](docs_sample/demo.gif)
*(Add a screen recording GIF here after running the app)*

---

## 🧠 How It Works (RAG Architecture)

```
PDF Upload
    ↓
PyPDFLoader  →  Text Chunks (RecursiveCharacterTextSplitter)
    ↓
OpenAI Embeddings  →  FAISS Vector Store
    ↓
User Question  →  Similarity Search (Top 4 chunks)
    ↓
GPT-3.5-Turbo  →  Answer grounded in retrieved context
    ↓
Conversational Memory  →  Follow-up questions supported
```

**RAG = Retrieval Augmented Generation** — the LLM only answers using the retrieved document chunks, which reduces hallucination and keeps answers accurate.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | OpenAI GPT-3.5-Turbo |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | FAISS (local, no extra cost) |
| PDF Parsing | PyPDF |
| UI | Streamlit |
| Memory | ConversationBufferMemory |

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/pdf-qa-bot.git
cd pdf-qa-bot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

### 5. Run the app
```bash
streamlit run src/app.py
```

Then open http://localhost:8501 in your browser. 🎉

---

## 💡 Features

- ✅ Upload any PDF (financial reports, research papers, contracts, manuals)
- ✅ Ask questions in plain English
- ✅ Conversational memory — ask follow-up questions naturally
- ✅ Source page references shown for every answer
- ✅ Fast local vector search with FAISS
- ✅ Clean Streamlit UI

---

## 📁 Project Structure

```
pdf-qa-bot/
├── src/
│   └── app.py              # Main Streamlit application
├── docs_sample/            # Sample PDFs for testing
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Support multiple PDFs at once
- [ ] Swap OpenAI for Anthropic Claude API
- [ ] Deploy to GCP Cloud Run
- [ ] Add Azure Blob Storage for PDF persistence
- [ ] Export chat history as PDF

---

## 📜 License

MIT License — free to use, modify, and share.

---

## 🤝 Connect

If you found this useful, let's connect on [LinkedIn](https://www.linkedin.com/in/malliswari-kasireddy-90b10521b/)!

#AgenticAI #LangChain #RAG #LLMs #Python #OpenAI
