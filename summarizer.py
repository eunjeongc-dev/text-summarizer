"""Text summarizer using the Anthropic API.

Supports three summary styles: bullet, short, and detail.
"""

import argparse
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-7"

STYLE_INSTRUCTIONS = {
    "bullet": (
        "Summarize the following text as a concise bulleted list of the key "
        "points. Use 5-8 bullets. Each bullet should be a single, "
        "self-contained sentence. Output only the bullet list, no preamble."
    ),
    "short": (
        "Summarize the following text in 2-3 sentences capturing only the "
        "most essential information. Output only the summary, no preamble."
    ),
    "detail": (
        "Provide a thorough, well-structured summary of the following text. "
        "Cover the main arguments, supporting details, and conclusions in "
        "several paragraphs. Preserve nuance and context. Output only the "
        "summary, no preamble."
    ),
}


def summarize(text: str, style: str, language: str = "Korean") -> str:
    """Summarize text in the requested style using the Anthropic API."""
    if style not in STYLE_INSTRUCTIONS:
        raise ValueError(
            f"Unknown style '{style}'. Choose from: {list(STYLE_INSTRUCTIONS)}"
        )

    client = anthropic.Anthropic()
    instruction = STYLE_INSTRUCTIONS[style]

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=f"You are an expert summarizer. Always respond in {language}.",
        messages=[
            {
                "role": "user",
                "content": f"{instruction}\n\n---\n\n{text}",
            }
        ],
    )

    return "".join(block.text for block in response.content if block.type == "text")


def read_input(path: str | None) -> str:
    """Read text from a file path or stdin."""
    if path:
        return Path(path).read_text(encoding="utf-8")
    if sys.stdin.isatty():
        raise SystemExit(
            "No input provided. Pass --file PATH or pipe text via stdin."
        )
    return sys.stdin.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize text using the Anthropic API."
    )
    parser.add_argument(
        "--style",
        choices=list(STYLE_INSTRUCTIONS),
        default="short",
        help="Summary style (default: short).",
    )
    parser.add_argument(
        "--file",
        help="Path to a text file to summarize. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--language",
        default="Korean",
        help="Language for the summary output (default: Korean).",
    )
    args = parser.parse_args()

    text = read_input(args.file).strip()
    if not text:
        raise SystemExit("Input text is empty.")

    print(summarize(text, args.style, args.language))


if __name__ == "__main__":
    main()
