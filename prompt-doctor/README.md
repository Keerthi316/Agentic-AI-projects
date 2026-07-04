# 🏥 Prompt Doctor

> **Learn prompt engineering by doing — with AI-graded, interactive exercises.**

Prompt Doctor is a Streamlit-powered learning platform that teaches prompt engineering through progressive, hands-on levels. You write prompts, run them against sample inputs, and get structured AI-powered feedback (grading) from an examiner — no manual review required.

---

## ✨ Features

- **🎮 Gamified Levels** — 5 progressively harder levels covering core prompt engineering concepts.
- **🤖 AI-Powered Grading** — An AI examiner evaluates your prompt against level-specific principles and gives actionable feedback.
- **📊 Two-Step Workflow** — First "Run Prompt" to see the AI's raw output, then "Evaluate Prompt" to get structured grading.
- **✅ Visual Pass/Fail Badges** — Each principle shows a clear pass/fail badge with the exact weakness and a guiding question.
- **🔓 Level Unlock System** — Pass all principles in a level to unlock the next one.
- **📝 Editable Sample Input** — Modify the sample input to experiment with different scenarios.
- **🗂️ Domain Selection** — Choose from General, Writing, Programming, Data Analysis, and Customer Support domains.

---

## 🧠 What You'll Learn

| Level | Principle | Focus |
|-------|-----------|-------|
| 1 | Role & Clear Instruction | Basic prompt structure — assign a persona and give unambiguous instructions |
| 2 | Structured Output | Request JSON and other machine-parseable formats |
| 3 | Few-Shot Examples | Use input-output examples to guide the AI's behaviour |
| 4 | Reasoning & Multi-Step Tasks | Chain-of-thought prompting for complex problems |
| 5 | Defensive Constraints | Guardrails against messy or adversarial input |

---

## 🏗️ Architecture

```
User writes prompt in Streamlit
        │
        ▼
runner.py ──► OpenRouter (user prompt + sample input) ──► raw output
        │
        ▼
examiner.py ──► OpenRouter (examiner system prompt + user prompt + level) ──► JSON
        │
        ▼
Streamlit displays grading
        │
        ▼
All principles pass? ──► Unlock next level
        │
        No ──► Show weaknesses + guiding questions
```

The **runner** executes your prompt against a sample input so you can see what the AI produces. The **examiner** evaluates your prompt text against level-specific principles and returns structured JSON with pass/fail results, weaknesses, and guiding questions.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend / App | [Streamlit](https://streamlit.io) |
| AI API | [OpenRouter](https://openrouter.ai) (`openai/gpt-4o-mini`) |
| Backend Logic | Python 3.10+ |
| HTTP | `requests` |
| Environment | `python-dotenv` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- An [OpenRouter](https://openrouter.ai/keys) API key (free credits available)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/prompt-doctor.git
   cd prompt-doctor
   ```

2. **Create a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**

   ```bash
   cp .env.example prompt_doctor/.env
   ```

   Open `prompt_doctor/.env` and replace `your_openrouter_api_key_here` with your actual OpenRouter API key.

5. **Run the app**

   ```bash
   streamlit run prompt_doctor/app.py
   ```

   The app will open in your browser at `http://localhost:8501`.

---

## 📖 How to Use

1. **Select a level** from the sidebar (Level 1 is always unlocked).
2. **Read the task** and the principles you need to satisfy.
3. **Write your prompt** in the text area.
4. Click **🚀 Run Prompt** to see what your prompt produces with the sample input.
5. Click **📊 Evaluate Prompt** to get AI feedback on your prompt.
6. Review the pass/fail results for each principle:
   - ✅ **Pass** — you've satisfied this principle.
   - ❌ **Needs Work** — read the weakness and guiding question, then revise your prompt.
7. When **all principles pass**, the next level unlocks automatically. 🎉

---

## 📁 Project Structure

```
prompt-doctor/
├── README.md                      # You are here
├── requirements.txt               # Python dependencies
├── .env.example                   # API key template
└── prompt_doctor/
    ├── app.py                     # Streamlit entry point — UI, session state, navigation
    ├── levels.py                  # Level definitions — tasks, sample inputs, principles
    ├── runner.py                  # Executes user prompt against sample input via OpenRouter
    ├── examiner.py                # Evaluates user prompt via OpenRouter, returns structured JSON
    ├── ARCHITECTURE.md            # Deep-dive architecture documentation
    └── .env                       # API key (gitignored — create from .env.example)
```

---

## 🔧 Configuration

All configuration is done through environment variables in `prompt_doctor/.env`:

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key. Get one at [openrouter.ai/keys](https://openrouter.ai/keys). |

You can also modify the model, timeout, and temperature in `examiner.py` and `runner.py` if needed.

---

## 🧪 Development

### Running tests

Tests are located alongside the source modules. Run them with:

```bash
pytest prompt_doctor/
```

### Adding a new level

1. Add a new `Level` dataclass to `levels.py`.
2. Add corresponding principles in `LEVEL_PRINCIPLES` in `examiner.py`.
3. The UI automatically picks up new levels — no changes needed in `app.py`.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- Built with [Streamlit](https://streamlit.io) — turns Python scripts into web apps.
- Powered by [OpenRouter](https://openrouter.ai) — unified API for LLMs.
- Icons by [Icons8](https://icons8.com).