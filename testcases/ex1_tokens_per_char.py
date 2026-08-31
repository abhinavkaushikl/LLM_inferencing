"""Exercise 1.1 -- tokens-per-character ratio: Hindi vs English.

5 sentence pairs (same meaning in both languages) run through the Llama-3.1
tokenizer. BOS is stripped so we measure only the content tokens.
"""

from transformers import AutoTokenizer

MODEL = "NousResearch/Meta-Llama-3.1-8B"  # ungated mirror of meta-llama/Llama-3.1-8B

PAIRS = [
    ("How are you today?",
     "आप आज कैसे हैं?"),
    ("The model generates one token at a time during decoding.",
     "डिकोडिंग के दौरान मॉडल एक बार में एक टोकन बनाता है।"),
    ("I went to the market yesterday and bought vegetables.",
     "मैं कल बाज़ार गया और सब्ज़ियाँ खरीदीं।"),
    ("Inference latency depends on the number of tokens, not characters.",
     "इन्फरेंस लेटेंसी टोकन की संख्या पर निर्भर करती है, अक्षरों पर नहीं।"),
    ("This is a simple sentence written for a tokenizer experiment.",
     "यह टोकनाइज़र प्रयोग के लिए लिखा गया एक सरल वाक्य है।"),
]


def stats(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    chars = len(text)
    return len(ids), chars, len(ids) / chars


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"Tokenizer: {MODEL}\n")

    totals = {"en": [0, 0], "hi": [0, 0]}
    print(f"{'#':>2}  {'lang':4} {'tok':>4} {'chars':>6} {'tok/char':>9}  text")
    print("-" * 88)
    for i, (en, hi) in enumerate(PAIRS, 1):
        for lang, text in (("en", en), ("hi", hi)):
            n_tok, n_chr, ratio = stats(tok, text)
            totals[lang][0] += n_tok
            totals[lang][1] += n_chr
            print(f"{i:>2}  {lang:4} {n_tok:>4} {n_chr:>6} {ratio:>9.3f}  {text}")
        print()

    print("-" * 88)
    agg = {}
    for lang in ("en", "hi"):
        t, c = totals[lang]
        agg[lang] = t / c
        print(f"{lang}: {t:>4} tokens / {c:>4} chars  ->  {t / c:.3f} tokens per char"
              f"   ({c / t:.2f} chars per token)")
    print(f"\nGap (tokens/char): Hindi is {agg['hi'] / agg['en']:.2f}x English")
    print(f"Gap (total tokens for the same 5 meanings): "
          f"{totals['hi'][0]}/{totals['en'][0]} = "
          f"{totals['hi'][0] / totals['en'][0]:.2f}x")

    print("\nPer-pair token counts (same meaning, en -> hi):")
    for i, (en, hi) in enumerate(PAIRS, 1):
        e = len(tok.encode(en, add_special_tokens=False))
        h = len(tok.encode(hi, add_special_tokens=False))
        print(f"  {i}: {e:>3} -> {h:>3}  ({h / e:.2f}x)")


if __name__ == "__main__":
    main()
