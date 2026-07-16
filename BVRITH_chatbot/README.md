# 🎓 BVRIT College FAQ Chatbot

A production-quality Retrieval-Augmented Generation (RAG) chatbot for answering questions about BVRIT College using a knowledge base document.

## 🏗️ Architecture

```
User Question
    │
    ▼
┌─────────────────┐
│  Streamlit UI    │
│  (app.py)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Chatbot Logic   │────▶│  LLM (OpenRouter)│
│  (chatbot.py)    │     │  GPT-4o Mini    │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Vector Store    │◀───▶│  ChromaDB        │
│  (vector_store.py)│    │  (chroma_db/)    │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Document        │
│  Loader          │
│  (utils.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  college_kb.docx │
│  (data/)        │
└─────────────────┘
```

## ✨ Features

- **RAG-based Q&A**: Answers only from the college knowledge base document
- **Citation Support**: Every answer includes section citations
- **Conversation History**: Maintains context across follow-up questions
- **Suggested Questions**: Quick-start buttons for common queries
- **Typing Animation**: Visual feedback during response generation
- **Copy Response**: One-click copy of bot responses
- **Source Viewer**: Toggle to view retrieved document chunks
- **Confidence Score**: Shows retrieval confidence for each answer
- **Response Time**: Displays time taken to generate response
- **Dark Mode Friendly**: Modern UI that works in both light and dark mode
- **Section Filter**: Filter by college sections (Admissions, Placements, etc.)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | Python 3.10+ |
| **UI** | Streamlit |
| **RAG Framework** | LangChain |
| **Vector Database** | ChromaDB |
| **Embeddings** | OpenAI/OpenRouter Embeddings |
| **LLM** | GPT-4o Mini (via OpenRouter) |
| **Document Loader** | Docx2txt |
| **Environment** | python-dotenv |

## 📋 Prerequisites

- Python 3.10 or higher
- OpenRouter API key (or OpenAI API key)

## 🚀 Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd college-chatbot
```

2. **Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Copy `.env.example` to `.env` and add your API key:

```bash
copy .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

5. **Add your knowledge base document**

Place your college knowledge base `.docx` file in the `data/` folder:

```
data/college_kb.docx
```

## 🎯 Usage

Run the application:

```bash
streamlit run app.py
```

The app will:
1. Load and process the knowledge base document
2. Create embeddings and store them in ChromaDB
3. Launch the Streamlit web interface
4. Be ready to answer your questions

## 📁 Project Structure

```
college-chatbot/
│
├── app.py              # Streamlit UI (main entry point)
├── chatbot.py          # Core chatbot logic (CollegeChatbot class)
├── vector_store.py     # ChromaDB vector store management
├── prompts.py          # System prompts and prompt templates
├── utils.py            # Utility functions (document loading, config)
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (API keys)
├── .env.example       # Example environment file
│
├── data/              # Knowledge base documents
│   └── college_kb.docx
│
├── chroma_db/          # Persisted vector database (auto-created)
│
└── README.md           # This file
```

## 🧪 How It Works

1. **Document Loading**: The `.docx` file is loaded using LangChain's `Docx2txtLoader`
2. **Text Splitting**: Content is split into chunks (800 chars, 150 overlap) using `RecursiveCharacterTextSplitter`
3. **Embedding**: Each chunk is embedded using OpenAI/OpenRouter embeddings
4. **Storage**: Embeddings are stored in ChromaDB with persistence to disk
5. **Retrieval**: User queries retrieve top-5 most relevant chunks via similarity search
6. **Generation**: The LLM generates answers grounded only in the retrieved context
7. **Citations**: Every answer includes section citations from the source document

## 🔒 Security

- API keys are stored in `.env` (never committed to version control)
- The chatbot ignores prompt injection attempts
- System prompt is never revealed to users

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | — |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | LLM model to use | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |

### Chunking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunk_size` | 800 | Maximum characters per chunk |
| `chunk_overlap` | 150 | Overlap between consecutive chunks |
| `top_k` | 5 | Number of retrieved documents |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.