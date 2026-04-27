"""Text summarizer using the sumy library (no API required).

Supports three summary styles (bullet, short, detail) for both English
and Korean text. Language is auto-detected from the input by default.
"""

import argparse
import re
import sys
from pathlib import Path

import nltk
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words

STYLE_SENTENCE_COUNTS = {
    "bullet": 6,
    "short": 3,
    "detail": 12,
}

HANGUL_RE = re.compile(r"[가-힯]")
KOREAN_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|(?<=[다요죠])\s+")


def ensure_nltk_data() -> None:
    """Download NLTK punkt tokenizers if missing."""
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def detect_language(text: str) -> str:
    """Return 'korean' if Hangul is present, else 'english'."""
    return "korean" if HANGUL_RE.search(text) else "english"


def split_korean_sentences(text: str) -> list[str]:
    """Naive Korean sentence splitter used as a tokenizer fallback."""
    parts = KOREAN_SENTENCE_END_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_parser(text: str, language: str) -> PlaintextParser:
    """Build a sumy parser, falling back to manual splitting for Korean."""
    try:
        return PlaintextParser.from_string(text, Tokenizer(language))
    except LookupError:
        if language == "korean":
            joined = "\n".join(split_korean_sentences(text))
            return PlaintextParser.from_string(joined, Tokenizer("english"))
        raise


def build_summarizer(language: str) -> LsaSummarizer:
    """Configure an LSA summarizer with stemmer/stop words when available."""
    try:
        summarizer = LsaSummarizer(Stemmer(language))
    except LookupError:
        summarizer = LsaSummarizer()
    try:
        summarizer.stop_words = get_stop_words(language)
    except LookupError:
        pass
    return summarizer


def summarize(text: str, style: str, language: str | None = None) -> str:
    """Summarize text in the requested style using sumy."""
    if style not in STYLE_SENTENCE_COUNTS:
        raise ValueError(
            f"Unknown style '{style}'. Choose from: {list(STYLE_SENTENCE_COUNTS)}"
        )

    if language is None:
        language = detect_language(text)

    parser = build_parser(text, language)
    summarizer = build_summarizer(language)
    sentences = [str(s) for s in summarizer(parser.document, STYLE_SENTENCE_COUNTS[style])]

    if not sentences:
        return text.strip()

    if style == "bullet":
        return "\n".join(f"- {s}" for s in sentences)
    return " ".join(sentences)


def read_input(path: str | None) -> str:
    """Read text from a file path or stdin."""
    if path:
        return Path(path).read_text(encoding="utf-8")
    if sys.stdin.isatty():
        raise SystemExit("No input provided. Pass --file PATH or pipe text via stdin.")
    return sys.stdin.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize text locally using the sumy library."
    )
    parser.add_argument(
        "--style",
        choices=list(STYLE_SENTENCE_COUNTS),
        default="short",
        help="Summary style (default: short).",
    )
    parser.add_argument(
        "--file",
        help="Path to a text file to summarize. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--language",
        choices=["korean", "english"],
        help="Force language. Auto-detected from the text when omitted.",
    )
    args = parser.parse_args()

    ensure_nltk_data()

    text = read_input(args.file).strip()
    if not text:
        raise SystemExit("Input text is empty.")

    print(summarize(text, args.style, args.language))


if __name__ == "__main__":
    main()
