"""Inspect how a tokenizer splits text into tokens.

meta-llama/Llama-3.1-8B is a gated repo: you need a Hugging Face account that has
been granted access, plus a token (`huggingface-cli login` or HF_TOKEN env var).
If that model isn't reachable, we fall back to an open tokenizer so the demo still
runs -- token counts will differ, but the mechanics are identical.
"""

from transformers import AutoTokenizer

PRIMARY = "meta-llama/Llama-3.1-8B"
FALLBACK = "NousResearch/Meta-Llama-3.1-8B"  # ungated mirror, same vocab

SAMPLES = ["Hello world", "नमस्ते दुनिया", "print('hi')", "  spaces   here"]


def load_tokenizer():
    for name in (PRIMARY, FALLBACK):
        try:
            return AutoTokenizer.from_pretrained(name), name
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
    raise SystemExit("No tokenizer could be loaded -- check network / HF access.")


def main():
    tok, name = load_tokenizer()
    print(f"\nTokenizer: {name}  (vocab size {tok.vocab_size})\n")

    for text in SAMPLES:
        ids = tok.encode(text)
        print(f"{len(ids):3d} tokens | {text!r}")
        print(f"    → {[tok.decode([i]) for i in ids]}")


if __name__ == "__main__":
    main()
