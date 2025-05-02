from openai import OpenAI
import fitz
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError, APIError

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def ask_openai(prompt):
    """Send a prompt to OpenAI with retry logic."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

def extract_text(pdf_path):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def summarize_audit(text, filename):
    """Summarize the audit in structured and compact format."""
    prompt = f"""
You are a senior auditor. Analyze the following audit report and provide a clear, compact summary (no more than 1 page) with the following structure:

### Audit of {filename}

1. Audit Topic
- [Short name or objective of the audit]

2. Main Findings
- Bullet-point summary of 2–4 key findings

3. Measures Proposed
- Actionable responses or controls (2–4 bullets)

4. Future Steps
- Follow-up recommendations or next steps (2–3 bullets)

TEXT:
{text[:8000]}
"""
    return ask_openai(prompt)

def compare_audits(summaries):
    """Compare audits to highlight similarities and differences."""
    joined = "\n\n".join(summaries)
    prompt = (
        "Compare the following structured audit summaries. Identify:\n"
        "- Similar audit topics or risks\n"
        "- Differences in findings or approaches\n"
        "- Shared improvement areas\n\n"
        f"{joined}"
    )
    return ask_openai(prompt)

def extract_learnings(summaries):
    """Extract 3–5 actionable learnings from audit summaries."""
    joined = "\n\n".join(summaries)
    prompt = (
        "Based on these structured audit summaries, provide 3–5 concise audit lessons or future best practices.\n\n"
        f"{joined}"
    )
    return ask_openai(prompt)

def analyze_audits(paths, filenames):
    """Process all uploaded audits and return summaries, comparison, and learnings."""
    summaries = []
    for path, name in zip(paths, filenames):
        text = extract_text(path)
        summary = summarize_audit(text, name)
        summaries.append(summary)

    comparison = compare_audits(summaries)
    learnings = extract_learnings(summaries)

    full_text = "=== AUDIT SUMMARIES ===\n\n" + "\n\n".join(summaries)
    full_text += "\n\n=== COMPARISON ===\n\n" + comparison
    full_text += "\n\n=== LEARNINGS ===\n\n" + learnings

    return summaries, comparison, learnings, full_text
