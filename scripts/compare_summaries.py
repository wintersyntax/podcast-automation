import os
from pathlib import Path
import requests
from dotenv import load_dotenv


load_dotenv()


MODEL = "google/gemini-2.5-flash"

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)


INPUTS = {
    "apple": "tests/fixtures/transcript_compiler/apple-transcript.txt",
    "whisper": "tests/fixtures/transcript_compiler/whisper-transcript.txt",
    "compiled": "var/compiler-reports/compiled-transcript.txt",
}


OUTPUT_DIR = Path(
    "var/test_output/summaries"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def summarize_file(
    name,
    file_path
):

    print(
        f"Summarizing {name}..."
    )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        transcript = f.read()


    prompt = f"""
You are a podcast analyst.

Create a structured markdown summary.

Include:

# Summary

# Key Ideas

# Important Details

# Questions To Explore

Transcript:

{transcript}
"""


    api_key = os.getenv(
        "PODCAST_KNOWLEDGE_API_KEY"
    )

    if not api_key:
        raise Exception(
            "Missing PODCAST_KNOWLEDGE_API_KEY"
        )


    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },
        timeout=120,
    )


    response.raise_for_status()


    data = response.json()


    summary = (
        data["choices"][0]
        ["message"]
        ["content"]
    )


    output = OUTPUT_DIR / f"{name}-summary.md"


    output.write_text(
        summary,
        encoding="utf-8"
    )


    print(
        "Saved:",
        output
    )


def main():

    for name, file_path in INPUTS.items():

        if not Path(file_path).exists():

            print(
                "Missing:",
                file_path
            )

            continue


        summarize_file(
            name,
            file_path
        )


if __name__ == "__main__":
    main()
