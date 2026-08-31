"""Exercise 1.2 -- measure TTFT and TPOT separately with HuggingFace generate().

Why not a real 8B here: fp16 8B = ~16 GB of weights and this box has 16 GB of
*total* unified memory, so it would swap and the timings would be meaningless.
The harness is model-agnostic -- point MODEL at an 8B on a 24 GB+ GPU and the
same numbers come out.

TTFT  = time from generate() call to the 1st output token.
TPOT  = mean gap between consecutive output tokens (decode steps 2..N).
"""

import argparse
import time
from statistics import median

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class Stopwatch(StoppingCriteria):
    """Called once per generated token -- gives us a per-token timestamp."""

    def __init__(self, device):
        self.device = device
        self.stamps = []

    def __call__(self, input_ids, scores, **kwargs):
        sync(self.device)
        self.stamps.append(time.perf_counter())
        return False


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_prompt(tok, n_tokens):
    """A prompt that tokenizes to ~n_tokens."""
    base = ("The transformer architecture processes tokens in parallel during the "
            "prefill phase and one at a time during decode. ")
    ids = tok.encode(base * (n_tokens // 8 + 4), add_special_tokens=False)[:n_tokens - 1]
    return tok.decode(ids)


def run(model, tok, device, prompt, max_new_tokens):
    inputs = tok(prompt, return_tensors="pt").to(device)
    n_in = inputs.input_ids.shape[1]
    watch = Stopwatch(device)

    sync(device)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,   # no early EOS, keeps the sample size fixed
        do_sample=False,
        stopping_criteria=StoppingCriteriaList([watch]),
        pad_token_id=tok.eos_token_id,
    )
    sync(device)
    t_end = time.perf_counter()

    n_out = out.shape[1] - n_in
    ttft = watch.stamps[0] - t0
    gaps = [b - a for a, b in zip(watch.stamps, watch.stamps[1:])]
    tpot = sum(gaps) / len(gaps)
    return {
        "n_in": n_in, "n_out": n_out,
        "ttft": ttft, "tpot": tpot,
        "e2e": t_end - t0,
        "e2e_model": ttft + tpot * (n_out - 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"device={device} dtype={dtype} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params / 1e9:.2f}B  weights={n_params * dtype.itemsize / 1e9:.1f} GB\n")

    print(f"{'prompt_tok':>10} {'out_tok':>8} {'TTFT ms':>9} {'TPOT ms':>9} "
          f"{'TTFT/TPOT':>10} {'E2E s':>7} {'decode tok/s':>13}")
    print("-" * 76)
    rows = []
    for n in args.prompt_lens:
        prompt = make_prompt(tok, n)
        # MPS recompiles per input shape, so warm up at THIS length before timing
        run(model, tok, device, prompt, 8)
        trials = [run(model, tok, device, prompt, args.new_tokens)
                  for _ in range(args.repeats)]
        r = min(trials, key=lambda t: t["ttft"])       # least-noise run
        r["tpot"] = median([t["tpot"] for t in trials])
        rows.append(r)
        spread = max(t["ttft"] for t in trials) / min(t["ttft"] for t in trials)
        print(f"{r['n_in']:>10} {r['n_out']:>8} {r['ttft'] * 1e3:>9.1f} {r['tpot'] * 1e3:>9.2f} "
              f"{r['ttft'] / r['tpot']:>10.1f} {r['e2e']:>7.2f} {1 / r['tpot']:>13.1f}"
              f"   (TTFT spread {spread:.2f}x over {args.repeats} runs)")

    print("\nPrefill scaling (TTFT vs prompt length):")
    base = rows[0]
    for r in rows:
        print(f"  {r['n_in']:>5} tok: {r['ttft'] * 1e3:>8.1f} ms   "
              f"({r['n_in'] / base['n_in']:.1f}x tokens -> {r['ttft'] / base['ttft']:.2f}x TTFT)"
              f"   {r['ttft'] * 1e6 / r['n_in']:.1f} us/prompt-token")


if __name__ == "__main__":
    main()
