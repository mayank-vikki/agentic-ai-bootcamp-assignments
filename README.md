# Agentic AI Enterprise Mastery Bootcamp — Assignments

**Bootcamp**: [Manifold AI — Agentic AI Enterprise Mastery (April 2025)](https://github.com/manifoldailearning/agentic-bootcamp-april-12)

## Branch Strategy

| Branch | Content |
|--------|---------|
| `main` | Shared utils, README |
| `week-01/assignment` | Week 1 assignment |
| `week-02/assignment` | Week 2 assignment |
| `week-NN/assignment` | ... |
| `capstone` | Final capstone project |

## Setup

```bash
cp .env.example .env   # add your API keys
pip install -r requirements.txt
```

> **Important:** `.env` must NOT be committed. It contains API keys and is excluded via `.gitignore`.

## How to Run

```bash
python app.py
```

## Usage

```python
from utils import get_model, logged_invoke

llm = get_model("deepseek")  # or "openai", "openai-mini", etc.
response = logged_invoke(llm, "Hello, world!", model_name="deepseek")
print(response.content)
```

Logs are written to `logs/{session_id}.jsonl`.
