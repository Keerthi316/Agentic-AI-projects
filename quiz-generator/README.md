# 🎯 QuizForge — AI-Powered Quiz Generator

Turn your PowerPoint presentations into interactive multiple-choice quizzes using AI.

## Overview

QuizForge is a Streamlit web app that accepts `.pptx` files and uses OpenAI to automatically generate multiple-choice questions (MCQs) from the slide content. Users can configure the number of questions and difficulty level, take the quiz interactively, and review detailed explanations for every answer.

A Flask REST API backend is also included for headless or programmatic usage.

---

## Features

- **PPT Upload** — Upload any `.pptx` file; the app extracts and previews slide content
- **AI Question Generation** — Powered by OpenAI (`gpt-4o-mini` by default); generates 5–30 MCQs per session
- **Three Difficulty Levels**
  - 🌱 Simple — recall and recognition
  - 📚 Medium — comprehension and application
  - 🧠 Complex — analysis, evaluation, and synthesis
- **Interactive Quiz UI** — One question at a time with immediate feedback and explanations
- **Distractor Explanations** — Each wrong option comes with a reason why it's incorrect
- **Results Dashboard** — Score summary with per-question breakdown and correct answers
- **Retry Logic** — Automatically retries AI generation up to 3 times on failure
- **Flask API** — REST endpoints for upload, quiz generation, and file cleanup

---

## Project Structure

```
quiz-generator/
├── streamlit_app.py          # Streamlit frontend (main entry point)
├── requirements.txt          # Frontend dependencies
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
└── backend/
    ├── app.py                # Flask REST API server
    ├── quiz_generator.py     # OpenAI quiz generation logic
    ├── ppt_parser.py         # PPTX text extraction
    ├── requirements.txt      # Backend dependencies
    └── .env                  # API keys (not committed)
```

---

## Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd quiz-generator
```

### 2. Install dependencies

**For the Streamlit app:**

```bash
pip install -r requirements.txt
```

**For the Flask backend (optional):**

```bash
pip install -r backend/requirements.txt
```

### 3. Configure environment variables

Create a `.env` file inside the `backend/` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini          # optional, defaults to gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1  # optional, for OpenRouter or custom endpoints
```

---

## Running the App

### Streamlit (recommended)

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

### Flask API (optional)

```bash
cd backend
python app.py
```

The API server starts at `http://localhost:5000`.

---

## Flask API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a `.pptx` file |
| `POST` | `/api/generate-quiz` | Generate MCQs from an uploaded file |
| `POST` | `/api/cleanup` | Delete an uploaded file |

### Upload a file

```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@presentation.pptx"
```

### Generate a quiz

```bash
curl -X POST http://localhost:5000/api/generate-quiz \
  -H "Content-Type: application/json" \
  -d '{"file_id": "<returned_file_id>", "num_questions": 10, "difficulty": "medium"}'
```

**Difficulty options:** `simple`, `medium`, `complex`
**Question range:** 5–30

---

## How It Works

1. The PPTX file is parsed using `python-pptx` — text is extracted from shapes and tables on every slide.
2. The combined slide text is sent to the OpenAI Chat Completions API with a structured prompt.
3. The model returns a JSON array of MCQ objects, each with 4 options, a correct answer, an explanation, and distractor reasoning.
4. If the model returns fewer questions than requested, the app retries automatically (up to 5 batch iterations, 3 top-level retries).
5. The Streamlit UI renders the quiz one question at a time and tracks answers in session state.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `python-pptx` | Parse `.pptx` files |
| `openai` | Call OpenAI Chat Completions API |
| `python-dotenv` | Load environment variables from `.env` |
| `flask` + `flask-cors` | REST API backend |
| `gunicorn` | Production WSGI server for Flask |

---

## Configuration

Streamlit theme is set in `.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#6366f1"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#0f172a"
font = "sans serif"
```

---

## Notes

- Maximum file size for the Flask API is **50MB**.
- The app supports OpenRouter or any OpenAI-compatible endpoint via `OPENAI_BASE_URL`.
- Uploaded files are stored temporarily and can be deleted via the cleanup endpoint or the "New Upload" button in the UI.
