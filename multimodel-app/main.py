import os
import time
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIStatusError

# Load OPENROUTER_API_KEY (and any other vars) from .env into os.environ
load_dotenv()

# Point the OpenAI client at OpenRouter's base URL.
# OpenRouter speaks the OpenAI API, so the same client works unchanged.
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

QUESTION = "What is the capital of France? Answer in one sentence."

MODELS = [
    "openai/gpt-oss-20b:free",
    "openrouter/owl-alpha",
    "google/gemma-4-31b-it:free",
]

# Price per million tokens (input, output) in USD
PRICES = {
    "openai/gpt-oss-20b:free":                (0.0, 0.0),
    "openrouter/owl-alpha":              (0.0, 0.0),
    "google/gemma-4-31b-it:free":              (0.0, 0.0),
}


TIMEOUT = 30  # seconds before giving up on a single model


def ask(question, model):
    start = time.perf_counter()
    try:
        # Send the question and block until the full response arrives
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            timeout=TIMEOUT,
        )
    except APITimeoutError:
        return None, time.perf_counter() - start, 0, 0, 0.0, f"Timed out after {TIMEOUT}s"
    except APIStatusError as e:
        return None, time.perf_counter() - start, 0, 0, 0.0, f"HTTP {e.status_code}: {e.message}"
    except Exception as e:
        return None, time.perf_counter() - start, 0, 0, 0.0, str(e)

    latency = time.perf_counter() - start

    # The answer lives at choices[0].message.content
    answer = response.choices[0].message.content

    in_tokens  = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens

    in_price, out_price = PRICES.get(model, (0.0, 0.0))
    cost = (in_tokens * in_price + out_tokens * out_price) / 1_000_000

    return answer, latency, in_tokens, out_tokens, cost, None


C_MODEL   = 42
C_PREVIEW = 46
C_LATENCY =  8
C_COST    = 10


def clip(text, width):
    """Truncate text to width, adding an ellipsis if cut."""
    return text if len(text) <= width else text[: width - 1] + "…"


def print_table(rows):
    """Print results as an aligned terminal table.

    rows: list of (model, answer, latency, cost, error)
    """
    header = (
        f"{'MODEL':<{C_MODEL}}  "
        f"{'PREVIEW':<{C_PREVIEW}}  "
        f"{'LATENCY':>{C_LATENCY}}  "
        f"{'COST':>{C_COST}}"
    )
    print()
    print(header)
    print("─" * len(header))

    for model, answer, latency, cost, error in rows:
        preview = clip(
            f"[ERROR] {error}" if error else answer.replace("\n", " "),
            C_PREVIEW,
        )
        print(
            f"{clip(model, C_MODEL):<{C_MODEL}}  "
            f"{preview:<{C_PREVIEW}}  "
            f"{latency:>{C_LATENCY - 1}.2f}s  "
            f"${cost:>{C_COST - 1}.6f}"
        )


if __name__ == "__main__":
    rows = []
    for model in MODELS:
        print(f"Asking {model} …", flush=True)
        answer, latency, in_tok, out_tok, cost, error = ask(QUESTION, model)
        rows.append((model, answer, latency, cost, error))

    print_table(rows)
