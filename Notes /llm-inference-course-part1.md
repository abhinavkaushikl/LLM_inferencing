# LLM Inference: Zero se PhD tak
## Part 1 — Foundations (Module 0 to 4)
---

# Poora Roadmap (pehle ye dekh lo)

Ye course 14 modules ka hai. Har module pichle par khada hai. Aaj hum Module 0–4 karenge, jo baaki sab ka foundation hai.

| # | Module | Kya seekhoge |
|---|--------|--------------|
| 0 | Prerequisites | Matrix math, GPU memory hierarchy, FLOPs vs bytes |
| 1 | Inference kya hai | Autoregressive loop, prefill vs decode, tokenization |
| 2 | Forward pass anatomy | RMSNorm, RoPE, Attention, SwiGLU, LM head, sampling |
| 3 | **KV Cache** | Kyun chahiye, memory math, MHA/MQA/GQA/MLA |
| 4 | **Roofline model** | Arithmetic intensity, memory-bound vs compute-bound |
| 5 | Batching | Static → dynamic → continuous batching, chunked prefill |
| 6 | Memory management | PagedAttention, fragmentation, prefix caching |
| 7 | Attention kernels | Online softmax, FlashAttention 1/2/3, FlashDecoding |
| 8 | Quantization | INT8/FP8/INT4, GPTQ, AWQ, SmoothQuant, KV-cache quant |
| 9 | Speculative decoding | Draft models, rejection sampling proof, Medusa, EAGLE |
| 10 | Parallelism | Tensor/Pipeline/Expert parallel, P/D disaggregation |
| 11 | MoE inference | Routing, expert placement, load imbalance |                                                                   
| 12 | Long context | Sliding window, StreamingLLM, H2O, RingAttention |
| 13 | Serving systems | TTFT/TPOT/goodput, schedulers, vLLM vs SGLang vs TRT-LLM |
| 14 | Frontier | Diffusion LLMs, KV-free architectures, hardware co-design |

---

# Kaise padhna hai (ye seriously follow karo)

PhD student ki tarah padhne ka matlab hai — **paper padhna, math derive karna, aur khud code likhna**. Sirf blog padhne se depth nahi aayegi.

Har module ke liye ye cycle chalao:

1. **Concept samjho** (ye document)
2. **Numbers nikalo** — har concept ka calculation khud kaagaz pe karo
3. **Code likho** — chhota sa NumPy/PyTorch mein implement karo
4. **Paper padho** — jo references diye hain
5. **Measure karo** — real GPU par profile karke dekho theory match hoti hai ya nahi
                
Agar aap step 3 aur 5 skip karoge, to aap "LLM inference ke baare mein jaante ho" — lekin "LLM inference jaante nahi ho". Farq bada hai.

**Setup jo chahiye:** Python, PyTorch, ek GPU (Colab T4 bhi chalega shuru mein), aur `transformers` library. Baad mein `vllm` bhi.

---
---

# Module 0 — Prerequisites

Ye module boring lagega lekin isko skip mat karna. LLM inference ka 80% intuition yahin se aata hai.

## 0.1 Matrix multiplication ka cost

Ek matrix multiply: `A (m×k) @ B (k×n) = C (m×n)`

- **FLOPs** = `2 × m × k × n`
  (2 kyun? Har output element ke liye k multiplications + k additions = 2k operations. Aur output elements m×n hain.)
- **Memory movement** = `(m×k + k×n + m×n) × bytes_per_element`

Ye do numbers — FLOPs aur bytes — poore course ki jaan hain. Yaad rakho.

**Example:** `A (1×4096) @ B (4096×4096)` — yani ek single token ka ek linear layer.
- FLOPs = 2 × 1 × 4096 × 4096 = **33.5 MFLOP**
- Memory (fp16) = (4096 + 16.7M + 4096) × 2 bytes ≈ **33.5 MB**

Dhyaan do: ~33 MFLOP compute, ~33 MB memory. Ratio = **1 FLOP per byte**. Ye number bahut important hai, Module 4 mein iska matlab samjhenge.

Ab wahi matrix, lekin 1024 tokens ke saath: `A (1024×4096) @ B (4096×4096)`
- FLOPs = 2 × 1024 × 4096 × 4096 = **34.4 GFLOP**
- Memory = (4.2M + 16.7M + 4.2M) × 2 ≈ **50 MB**
- Ratio = **~687 FLOP per byte**

Same weights, lekin ratio 687× badh gaya. Ye ek line poori LLM serving industry ka business model hai. Isko internalize karo.

## 0.2 GPU memory hierarchy

GPU mein memory ki layers hain, tez se dheemi:

```
Registers      →  ~256 KB per SM      →  ~20 TB/s     (instant)
Shared Mem/L1  →  ~192 KB per SM      →  ~15 TB/s     (~30 cycles)
L2 Cache       →  ~40-50 MB total     →  ~5 TB/s      (~200 cycles)
HBM (VRAM)     →  40-192 GB           →  1-8 TB/s     (~400-600 cycles)
CPU RAM        →  100s of GB          →  ~50 GB/s     (PCIe, bahut slow)
```

**Key insight:** Jo bhi data HBM se padhna padta hai, woh mehnga hai. Kernel optimization ka poora khel yahi hai — HBM traffic kam karo, jitna ho sake data ko shared memory/registers mein reuse karo.

FlashAttention (Module 7) exactly yahi karta hai. Woh attention ko "faster math" se fast nahi karta — woh HBM reads/writes kam karke fast karta hai.

## 0.3 Reference hardware numbers

Ye table paas rakho, baar-baar kaam aayegi:

| GPU | HBM Size | HBM BW | FP16 Dense TFLOPS | Ridge point (FLOP/byte) |
|-----|----------|--------|-------------------|--------------------------|
| T4 | 16 GB | 320 GB/s | 65 | ~203 |
| A100-40 | 40 GB | 1555 GB/s | 312 | ~200 |
| A100-80 | 80 GB | 2039 GB/s | 312 | ~153 |
| H100 SXM | 80 GB | 3350 GB/s | ~990 | ~295 |
| H200 | 141 GB | 4800 GB/s | ~990 | ~206 |
| B200 | 192 GB | 8000 GB/s | ~2250 | ~281 |

**Ridge point** = TFLOPS ÷ Bandwidth. Iska matlab: "GPU ko busy rakhne ke liye har byte par kitne FLOPs karne padenge."

A100-80 par ridge point ~153 hai. Aur humne upar dekha ki single-token matmul ka intensity **1** hai. Yani hum GPU ki compute capacity ka **0.65%** use kar rahe hain. Baaki 99.35% time GPU sirf memory ka intezaar kar raha hai.

> **Ye course ka central drama hai.** LLM decoding memory-bound hai, aur poori field isi problem ko alag-alag angle se attack kar rahi hai.

## 0.4 Numeric formats

| Format | Bits | Range | Kahan use hota hai |
|--------|------|-------|--------------------|
| FP32 | 32 | huge | Reference/accumulation |
| FP16 | 16 | ±65504 | Common inference (overflow risk) |
| BF16 | 16 | FP32 jaisa range, kam precision | Default aajkal |
| FP8 (E4M3) | 8 | ±448 | H100+ weights/activations |
| INT8 | 8 | -128..127 | Quantized weights |
| INT4 | 4 | -8..7 | Aggressive weight quant |

BF16 vs FP16: dono 16-bit hain, lekin BF16 mein exponent bits zyada (8) aur mantissa kam (7). Isliye BF16 overflow nahi karta, precision thodi kam hoti hai. Training aur inference dono mein aajkal BF16 default hai.

**Rule of thumb:** Model weights ka size = `params × bytes_per_param`.
- 8B model in BF16 → 16 GB
- 8B model in INT8 → 8 GB
- 8B model in INT4 → 4 GB
- 70B model in BF16 → 140 GB (ek A100-80 mein fit nahi hoga!)

## Module 0 Exercises

1. Ek 70B model (BF16) ko A100-80 par chalane ke liye minimum kitne GPUs chahiye? Sirf weights consider karo. Ab KV cache aur activations ke liye 20% headroom bhi jodo.
2. `A (512×2048) @ B (2048×8192)` ka FLOPs, bytes, aur arithmetic intensity nikalo (BF16). Kya ye H100 par compute-bound hoga?
3. NumPy mein matmul likho aur timing measure karo. Theoretical FLOPS se compare karo. Gap kyun hai?

---
---

# Module 1 — Inference kya hai actually

## 1.1 Training vs Inference — fundamental farq

| | Training | Inference |
|---|---|---|
| Direction | Forward + Backward | Sirf Forward |
| Data | Poora sequence ek saath | Ek token at a time (generation mein) |
| Parallelism | Sequence-level parallel | Sequential dependency |
| Memory | Gradients + optimizer states | Weights + KV cache |
| Bottleneck | Usually compute | Usually **memory bandwidth** |
| Optimization goal | Throughput | Latency **aur** throughput dono |

Sabse bada farq: **training mein poora sequence parallel process hota hai. Inference generation mein nahi ho sakta**, kyunki token N+1 banane ke liye token N chahiye — aur token N abhi bana hi nahi.

Ye sequential dependency LLM inference ki asli dushman hai.

## 1.2 Autoregressive generation loop

LLM ek function hai: `f(tokens) → next_token_probabilities`

Text generate karne ke liye hum isko loop mein chalate hain:

```python
# Conceptual pseudocode — abhi KV cache nahi hai
tokens = tokenizer.encode("Delhi ki capital")   # [1, 2847, 892, 4471]

for step in range(max_new_tokens):
    logits = model(tokens)          # shape: [len(tokens), vocab_size]
    next_logits = logits[-1]        # sirf LAST position chahiye
    next_token = sample(next_logits)
    tokens.append(next_token)
    
    if next_token == EOS:
        break
```

Ek cheez notice karo jo bahut important hai: **hum poora sequence forward karte hain, lekin sirf last position ka output use karte hain.** Baaki sab compute waste hai.

Agar prompt 100 tokens ka hai aur hum 100 tokens generate karte hain:
- Step 1: 101 tokens process, 1 use
- Step 2: 102 tokens process, 1 use
- ...
- Step 100: 200 tokens process, 1 use

Total ≈ 15,000 token-forward-passes, jabki actual naya kaam sirf 100 tokens ka tha. **150× waste.**

Ye waste KV cache khatam karta hai (Module 3). Lekin pehle samjho ki waste hai kyun.

## 1.3 Prefill vs Decode — do bilkul alag phases

Ye distinction poore course mein baar baar aayegi. Isko ab clear kar lo.

### Phase 1: PREFILL (a.k.a. prompt processing)

Input prompt ke saare tokens **ek saath, parallel mein** process hote hain.

- Input shape: `[batch, prompt_len, d_model]`
- Sab positions ek hi matmul mein
- Compute-heavy — GPU ke tensor cores properly busy
- **Compute-bound**
- Output: last position ka logit + saare positions ka KV cache
- Metric: **TTFT** (Time To First Token)

### Phase 2: DECODE (a.k.a. generation)

Ek-ek token, ek-ek step.

- Input shape: `[batch, 1, d_model]` ← seq length **1**
- Har step mein poore model ke weights HBM se padhne padte hain
- Sirf 1 token ka kaam, lekin 16 GB weights read
- **Memory-bandwidth-bound**
- Metric: **TPOT** (Time Per Output Token) / ITL (Inter-Token Latency)

### Visual mental model

```
PREFILL: "Delhi ki capital kya hai"
         ┌───┬───┬───┬───┬───┐
         │ t1│ t2│ t3│ t4│ t5│   ← saath mein, ek forward pass
         └───┴───┴───┴───┴───┘
              (parallel, fat matmuls, GPU khush)

DECODE:  ┌───┐
         │ t6│  → "Delhi"        step 1
         └───┘
              ┌───┐
              │ t7│  → " hi"     step 2
              └───┘
                   ┌───┐
                   │ t8│  → " hai"  step 3
                   └───┘
              (sequential, skinny matmuls, GPU bhukha)
```

**Isko yaad rakho:** Prefill aur decode itne alag hain ki modern serving systems inhe **alag GPUs par** bhi chalate hain (P/D disaggregation — Module 10). Ek hi request ke do phases, do alag machines.

## 1.4 Tokenization — jo zyadatar log ignore kar dete hain

Model text nahi dekhta, **token IDs** dekhta hai. Tokenizer text ko integers mein badalta hai.

Aajkal ke LLMs mostly **BPE (Byte Pair Encoding)** ya uske variants use karte hain.

### BPE kaise kaam karta hai

1. Text ko bytes mein todo
2. Sabse frequent adjacent pair merge karo
3. Repeat karo, jab tak vocabulary size target tak na pahunche

```
"lower lowest"
bytes:     l o w e r _ l o w e s t
merge 1:   lo w e r _ lo w e s t       (l+o most frequent)
merge 2:   low e r _ low e s t         (lo+w)
merge 3:   lowe r _ lowe s t
...
```

### Inference ke liye kyun matter karta hai

**1. Latency directly tokens par depend karti hai, characters par nahi.**

Hindi/Devanagari text English se zyada tokens leta hai. Example (rough, GPT-4 tokenizer):
- `"How are you"` → ~3 tokens
- `"आप कैसे हैं"` → ~10-12 tokens

Yani same meaning, **3-4× zyada tokens** → 3-4× zyada latency aur cost. Ye "tokenizer fairness" ka ek real research problem hai.

**2. Vocabulary size embedding aur LM head ka size decide karta hai.**

 Llama 3: vocab = 128,256, d_model = 4096
- Embedding matrix: 128256 × 4096 × 2 bytes = **1.05 GB**
- LM head (agar untied): another **1.05 GB**

8B model mein se 2 GB sirf vocab ke liye! Aur LM head ka matmul har decode step mein hota hai: `2 × 1 × 4096 × 128256 = 1.05 GFLOP`.

**3. Tokenization non-injective edge cases** — same text, alag tokenization ho sakti hai. Ye "glitch tokens" aur prompt-injection issues create karta hai.

### Hands-on

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

for text in ["Hello world", "नमस्ते दुनिया", "print('hi')", "  spaces   here"]:
    ids = tok.encode(text)
    print(f"{len(ids):3d} tokens | {text!r}")
    print(f"    → {[tok.decode([i]) for i in ids]}")
```

Isko chalao. Dekho ki whitespace kaise handle hota hai, code kaise tokenize hota hai, Hindi kaise. Ye 10 minute aapko tokenizer ke baare mein 10 blogs se zyada sikhayega.

## 1.5 Inference ke performance metrics

In terms ko precisely samjho — interviews aur papers mein constantly aayenge.

| Metric | Definition | Kis phase se |
|--------|-----------|--------------|
| **TTFT** | Request aane se pehla token milne tak ka time | Prefill |
| **TPOT / ITL** | Do consecutive output tokens ke beech ka time | Decode |
| **E2E Latency** | TTFT + (TPOT × output_tokens) | Dono |
| **Throughput** | Total output tokens/sec across all requests | System-level |
| **Goodput** | Throughput, lekin sirf woh requests jo SLO meet karti hain | System-level |

**Goodput** sabse important metric hai production mein, aur sabse kam samjha jaata hai. Agar aap batch size badha kar throughput 2× kar dete ho lekin 40% requests apna latency SLO miss kar deti hain, to aapka goodput actually gir gaya.

**Classic tradeoff:**
```
Batch size ↑  →  Throughput ↑  →  Per-request latency ↑
Batch size ↓  →  Throughput ↓  →  Per-request latency ↓
```

Chatbot chahiye? Latency optimize karo. Offline batch summarization? Throughput optimize karo. Dono ek saath nahi mil sakte — ye ek fundamental tension hai.

## Module 1 Exercises

1. Upar wala tokenizer script chalao. Hindi aur English mein 5-5 sentences ka tokens-per-character ratio nikalo. Kitna gap hai?
2. HuggingFace `generate()` se ek 8B model chalao, aur TTFT aur TPOT alag-alag measure karo. Ratio kya hai?
3. Ek 500-token prompt aur 200-token output ke liye, agar TTFT = 300ms aur TPOT = 25ms hai, to E2E latency kya hai? Agar prompt 2000 tokens ho jaye to TTFT roughly kitna hoga? (Hint: prefill compute-bound hai, yani ~linear in prompt length)

---
---

# Module 2 — Forward Pass ki Anatomy

Ab hum model ke andar ghusenge. Main Llama-3-8B ko running example ki tarah use karunga.

## 2.1 Llama-3-8B ka specification

```
n_layers        = 32
d_model         = 4096
n_heads         = 32
head_dim        = 128        (4096 / 32)
n_kv_heads      = 8          ← GQA! (Module 3 mein detail)
d_ffn           = 14336
vocab_size      = 128256
max_seq_len     = 8192 (base) / 128K (extended)
norm            = RMSNorm
pos_encoding    = RoPE (theta = 500000)
activation      = SwiGLU
attention       = GQA, causal
```

Inhe yaad kar lo. Har calculation mein use honge.

## 2.2 Poora flow

```
input_ids  [B, S]
    ↓
Embedding lookup                      → [B, S, 4096]
    ↓
┌─────────── × 32 layers ────────────┐
│  x_norm = RMSNorm(x)                │
│  attn_out = GQA(x_norm)  + RoPE     │
│  x = x + attn_out            ← residual
│                                     │
│  x_norm = RMSNorm(x)                │
│  ffn_out = SwiGLU_FFN(x_norm)       │
│  x = x + ffn_out             ← residual
└─────────────────────────────────────┘
    ↓
Final RMSNorm                         → [B, S, 4096]
    ↓
LM Head (4096 → 128256)               → [B, S, 128256]
    ↓
Sampling                              → next token
```

Ab har component ko todte hain.

## 2.3 RMSNorm

LayerNorm ka simplified version. Mean subtract nahi karta, sirf scale karta hai.

```
RMSNorm(x) = (x / sqrt(mean(x²) + ε)) * g
```

Yahan `g` ek learnable vector hai, size `d_model`.

```python
def rmsnorm(x, weight, eps=1e-6):
    # x: [..., d_model]
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight
```

**Kyun RMSNorm aur LayerNorm nahi?** LayerNorm mein mean-subtraction ek extra pass hai data par. RMSNorm ~10-15% faster hai aur quality mein koi meaningful farq nahi. Inference mein har microsecond matter karta hai.

**Cost:** FLOPs negligible (~4 × d_model per token), lekin ye ek **memory-bound elementwise op** hai. Isliye production kernels RMSNorm ko aage/peeche ke ops ke saath **fuse** karte hain (fused RMSNorm + QKV projection). Ye kernel fusion ka ek classic example hai.

## 2.4 Attention — QKV projections

```python
Q = x @ W_q     # [B,S,4096] @ [4096, 32*128=4096]  → [B,S,4096]
K = x @ W_k     # [B,S,4096] @ [4096,  8*128=1024]  → [B,S,1024]  ← chhota!
V = x @ W_v     # [B,S,4096] @ [4096,  8*128=1024]  → [B,S,1024]  ← chhota!
```

Dhyaan do: **K aur V ke projections chhote hain** kyunki `n_kv_heads = 8`, `n_heads = 32` nahi. Ye GQA hai. Isse KV cache 4× chhota ho jaata hai. Module 3 mein poora detail.

Reshape:
```
Q → [B, S, 32, 128]   (32 query heads)
K → [B, S,  8, 128]   (8 KV heads)
V → [B, S,  8, 128]
```

Attention compute karte waqt, har KV head ko **4 query heads** ke saath share kiya jaata hai (32/8 = 4).

## 2.5 RoPE — Rotary Position Embedding

Ye elegant idea hai, aur long-context research ka centre hai. Dhyaan se samjho.

### Problem

Transformer attention **permutation-invariant** hai. Bina position info ke, "kutta aadmi ko kaata" aur "aadmi kutte ko kaata" model ke liye same hain. Position information daalni padegi.

### RoPE ka idea

Position ko **add** mat karo — query aur key vectors ko position-dependent angle se **rotate** karo.

Head dimension (128) ko 64 pairs mein todo: `(x₀,x₁), (x₂,x₃), ..., (x₁₂₆,x₁₂₇)`

Har pair `i` ke liye frequency:
```
θᵢ = 1 / (base^(2i/d))          base = 500000 for Llama 3
```

Position `m` par, pair `i` ko `m·θᵢ` radians rotate karo:

```
[x'₂ᵢ  ]   [cos(mθᵢ)  -sin(mθᵢ)] [x₂ᵢ  ]
[x'₂ᵢ₊₁] = [sin(mθᵢ)   cos(mθᵢ)] [x₂ᵢ₊₁]
```

### Magic property (ye derive karo khud)

RoPE ke baad, do positions ka dot product **sirf unke relative distance par depend karta hai**:

```
⟨R_m·q, R_n·k⟩ = f(q, k, m−n)
```

Yani absolute position ka koi zikr nahi — sirf `m − n`. Ye exactly woh hai jo hum chahte hain: model ko "kitna door" pata hona chahiye, "kahan exactly" nahi.

**Proof sketch (khud complete karo):** 2D rotation matrices ka property hai `R_m^T R_n = R_{n−m}`. Isko 64 independent 2D subspaces par apply karo. Ye ek achha exercise hai — kaagaz pe likh kar karo.

```python
def apply_rope(x, cos, sin):
    # x: [B, S, H, D], cos/sin: [S, D/2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)
```

### Inference ke liye critical points

1. **RoPE Q aur K par lagta hai, V par NAHI.** V mein position info nahi jaati.
2. **KV cache mein post-RoPE keys store hote hain.** Yani cache position-baked hai. Isliye cached KV ko dusri position par reuse karna galat hoga — ye prefix caching (Module 6) mein ek subtle constraint hai.
3. **`base` value long context ka key hai.** Llama 2 mein base = 10000, Llama 3 mein 500000. Zyada base = slower rotation = longer effective context. YaRN, NTK-scaling, Position Interpolation — sab isi knob se khelte hain.
4. **cos/sin tables precompute hote hain** — har step recompute karna waste hai.

## 2.6 Scaled Dot-Product Attention

```
Attention(Q,K,V) = softmax(QKᵀ / √d_head + M) V
```

`M` causal mask hai: `M[i,j] = 0 if j ≤ i else −∞`. Yani token sirf apne aur peeche wale tokens ko dekh sakta hai.

### Prefill mein (S tokens)

```
QKᵀ:      [S, 128] @ [128, S]  → [S, S]     FLOPs = 2·S²·128
softmax:  [S, S]                            memory-bound
scores@V: [S, S] @ [S, 128]    → [S, 128]   FLOPs = 2·S²·128
```

Per head: `4·S²·d_head` FLOPs. **S² — quadratic!** Yahi long context ka fundamental problem hai.

### Decode mein (1 new token, S cached)

```
QKᵀ:      [1, 128] @ [128, S]  → [1, S]     FLOPs = 2·S·128
scores@V: [1, S] @ [S, 128]    → [1, 128]   FLOPs = 2·S·128
```

Linear in S. Compute chhota hai — lekin poora KV cache HBM se padhna padta hai. Phir se memory-bound.

### √d_head kyun?

Agar `q` aur `k` ke components independent, mean 0, variance 1 hain, to `q·k` ka variance `d_head` hoga. Bina scaling ke, `d_head=128` par dot products ka magnitude ~11 tak jaayega, softmax saturate ho jaayega, gradients vanish. `√d_head` se divide karne se variance wapas 1 ho jaata hai.

Ye derive karo khud — 2 line ka proof hai, lekin isse samajh aayega ki design choices random nahi hote.

## 2.7 SwiGLU FFN

Llama ka FFN standard 2-layer MLP nahi hai. **Teen** matrices hain:

```python
def swiglu_ffn(x, W_gate, W_up, W_down):
    gate = x @ W_gate          # [B,S,4096] → [B,S,14336]
    up   = x @ W_up            # [B,S,4096] → [B,S,14336]
    hidden = silu(gate) * up   # elementwise gating
    return hidden @ W_down     # [B,S,14336] → [B,S,4096]

def silu(x):
    return x * torch.sigmoid(x)    # a.k.a. Swish
```

**Gating intuition:** `gate` branch decide karti hai ki `up` branch ka kaunsa information aage jaayega. Ye ek learned, input-dependent filter hai.

**Parameter count:** 3 matrices × (4096 × 14336) = **176M params per layer**

Attention: (4096×4096) + 2×(4096×1024) + (4096×4096) = **41.9M params per layer**

> **FFN attention se 4.2× bada hai!** Ye counter-intuitive hai — sab "attention is all you need" bolte hain, lekin parameters aur FLOPs ka majority FFN mein hai. Isliye quantization aur MoE dono primarily FFN ko target karte hain.

`14336 ≈ 3.5 × 4096` kyun? Original transformer mein `4 × d_model` tha. SwiGLU mein 3 matrices hain, to param count same rakhne ke liye `(2/3) × 4 = 8/3 ≈ 2.67 × d_model` hona chahiye. Llama ne isko 14336 par round kiya (hardware-friendly multiple of 256).

## 2.8 LM Head aur Sampling

```
logits = x_final @ W_lm_head    # [B, S, 4096] @ [4096, 128256]
```

**Decode mein:** sirf last position chahiye → `[B, 1, 128256]`. FLOPs = `2 × B × 4096 × 128256 = 1.05 GFLOP` per sequence. Chhota nahi hai!

**Prefill mein:** Agar aap saare positions ke logits compute karte ho to `S × 1.05 GFLOP` — bilkul waste, kyunki chahiye sirf last. Achhe implementations prefill mein LM head sirf last position par lagate hain. Ye ek real optimization hai jo naive code miss kar deta hai.

### Sampling strategies

| Method | Kaise | Kab use karo |
|--------|-------|--------------|
| **Greedy** | `argmax(logits)` | Deterministic tasks, benchmarks |
| **Temperature** | `softmax(logits / T)` | T<1 sharper, T>1 flatter |
| **Top-k** | Top k tokens rakho, renormalize | k=40-50 typical |
| **Top-p (nucleus)** | Cumulative prob p tak tokens rakho | p=0.9-0.95 typical |
| **Min-p** | `p_max × min_p` se kam wale hatao | Adaptive, aajkal popular |

```python
def sample(logits, temperature=0.8, top_p=0.95):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # p se aage wale hata do
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx.gather(-1, idx)
```

**Inference engineer ke liye note:** Sampling ka cost small hai lekin zero nahi. Top-p mein 128K vocab par sort karna padta hai — batch 256 par ye measurable ho jaata hai. vLLM jaise systems ke paas fused sampling kernels hain jo sort avoid karte hain.

## 2.9 Total parameter accounting

Chalo verify karte hain ki 8B actually 8B hai:

```
Embedding:      128256 × 4096                    =   525 M

Per layer:
  W_q:          4096 × 4096                      =  16.8 M
  W_k:          4096 × 1024                      =   4.2 M
  W_v:          4096 × 1024                      =   4.2 M
  W_o:          4096 × 4096                      =  16.8 M
  W_gate:       4096 × 14336                     =  58.7 M
  W_up:         4096 × 14336                     =  58.7 M
  W_down:       14336 × 4096                     =  58.7 M
  norms:        2 × 4096                         =  ~0.008 M
  ─────────────────────────────────────────────────────────
  Total/layer                                    = 218.1 M

× 32 layers                                      =  6979 M
LM head:        4096 × 128256                    =   525 M
final norm                                       =   ~0 M
═══════════════════════════════════════════════════════════
TOTAL                                            ≈  8029 M  ✓
```

8.03 B. Match ho gaya.

**Isko khud kaagaz pe dobara karo.** Agar aap ye calculation blind kar sakte ho, to aapko architecture samajh aa gaya hai.

Aur ek breakdown notice karo:
- FFN: 58.7×3×32 = **5.6 B (70%)**
- Attention: 41.9×32 = **1.34 B (17%)**
- Embeddings: 1.05 B **(13%)**

## Module 2 Exercises

1. Ek single transformer layer NumPy mein from scratch likho (RMSNorm + GQA + RoPE + SwiGLU). HuggingFace ke output se compare karo — max absolute difference < 1e-3 hona chahiye.
2. RoPE ka relative-position property prove karo: dikhao ki `⟨R_m q, R_n k⟩` sirf `m−n` par depend karta hai.
3. Llama-3-70B ke liye (n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8, d_ffn=28672) poora parameter breakdown nikalo.
4. `√d_head` scaling ka variance argument formally likho.

---
---

# Module 3 — KV Cache (ye course ka dil hai)

## 3.1 Motivation

Module 1 mein humne dekha: naive generation mein har step par poora sequence recompute hota hai. Ye waste hai. Kyun?

Decode step `t` par, positions `1..t−1` ke K aur V vectors **bilkul wahi hain** jo pichle step mein the. Kyunki:
- `K = x @ W_k` — `x` position `i` par change nahi hua
- Causal attention — future tokens past ko affect nahi karte

To unhe **cache** kar lo.

```python
# Bina cache: O(S²) work per step
for step in range(N):
    K = all_tokens @ W_k    # sab dobara compute

# Cache ke saath: O(S) work per step
for step in range(N):
    k_new = new_token @ W_k       # sirf 1 token
    K_cache = cat([K_cache, k_new])
    K = K_cache                   # reuse
```

**Compute saving:** Generation ke liye O(S²) → O(S) per step.
**Cost:** Memory. Aur ye bahut zyada memory hai.

## 3.2 KV Cache size — the formula

```
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes_per_elem
           ↑
        K aur V
```

### Llama-3-8B ke liye (BF16, GQA with 8 KV heads)

**Per token:**
```
2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes = 128 KB per token
```

Ye number yaad rakho: **128 KB per token**.

**Ab scale karo:**

| Context | Batch=1 | Batch=32 | Batch=128 |
|---------|---------|----------|-----------|
| 2K | 256 MB | 8 GB | 32 GB |
| 8K | 1 GB | 32 GB | 128 GB ✗ |
| 32K | 4 GB | 128 GB ✗ | — |
| 128K | 16 GB | 512 GB ✗ | — |

A100-80GB par: weights 16 GB lete hain, to ~60 GB KV cache ke liye bacha. Yani 8K context par max batch ~60. **Ye aapki throughput ceiling hai** — GPU compute ki wajah se nahi, KV cache memory ki wajah se.

### Agar GQA na hota (MHA, 32 KV heads)?

```
2 × 32 × 32 × 128 × 2 = 524,288 bytes = 512 KB per token
```

**4× zyada.** 8K context par batch=32 ke liye 128 GB chahiye hota — A100 par possible hi nahi. GQA ke bina modern long-context serving economically viable hi nahi hoti.

## 3.3 Attention variants — KV cache ka evolution

Ye ek proper research arc hai. Har step KV cache chhota karne ke liye hai.

### MHA (Multi-Head Attention) — 2017

Har query head ka apna K aur V head.
```
n_kv_heads = n_heads = 32
```
- KV per token (8B config): 512 KB
- Quality: baseline (best)
- Problem: cache bahut bada

### MQA (Multi-Query Attention) — Shazeer, 2019

**Saare** query heads ek hi K,V head share karte hain.
```
n_kv_heads = 1
```
- KV per token: 16 KB — **32× reduction!**
- Quality: noticeable degradation, training instability
- Use: PaLM, Falcon

### GQA (Grouped-Query Attention) — 2023

Beech ka raasta. Query heads ko groups mein baanto, har group ka apna KV head.
```
n_kv_heads = 8, n_heads = 32  → 4 query heads per KV head
```

```
MHA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8       (8 query heads)
      │  │  │  │  │  │  │  │
      K1 K2 K3 K4 K5 K6 K7 K8       (8 KV heads)

GQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
      └──┬──┘  └──┬──┘  ...
        K1       K2                  (2 KV heads)

MQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
      └───────┬───────┘
             K1                      (1 KV head)
```

- KV per token: 128 KB — **4× reduction vs MHA**
- Quality: MHA ke almost barabar
- Use: Llama 2 70B, Llama 3 (all sizes), Mistral, Gemma, basically sab

**GQA aaj ka default hai.** Sweet spot mil gaya.

### MLA (Multi-head Latent Attention) — DeepSeek-V2/V3, 2024

Sabse clever idea. K,V heads kam karne ke bajaye, K aur V ko ek **low-rank latent vector** mein compress karo.

```
c_kv = x @ W_down_kv       # [d_model] → [d_latent], d_latent << d_model
# Cache mein sirf c_kv store karo!

# Attention ke time par decompress:
K = c_kv @ W_up_k
V = c_kv @ W_up_v
```

DeepSeek-V3 mein `d_latent` (compressed KV dim) 512 hai, jabki equivalent MHA mein `n_heads × head_dim` bahut zyada hota.

- KV per token: MHA ka ~**1/14**
- Quality: papers claim **MHA se behtar** (low-rank ek regularizer ki tarah kaam karta hai)
- Extra cost: decompression matmuls (compute ↑, memory ↓ — aur decode memory-bound hai, to ye achha trade hai)
- Complication: RoPE ko alag handle karna padta hai (decoupled RoPE), kyunki rotation aur low-rank projection commute nahi karte

MLA abhi frontier hai. DeepSeek papers padhna — ye 2024-25 ka sabse important architecture innovation hai.

### Comparison table

| Variant | n_kv_heads | KV/token (8B cfg) | Quality | Kaun use karta hai |
|---------|-----------|-------------------|---------|---------------------|
| MHA | 32 | 512 KB | Baseline | GPT-3, older models |
| MQA | 1 | 16 KB | ↓ | PaLM, Falcon |
| GQA | 8 | 128 KB | ≈ MHA | Llama 3, Mistral, Gemma |
| MLA | latent | ~36 KB | ≥ MHA | DeepSeek V2/V3 |

## 3.4 KV Cache ka layout — aur woh cheez jisne vLLM banaya

Naive implementation:
```python
kv_cache = torch.zeros(batch, max_seq_len, n_kv_heads, head_dim)
```

Har sequence ke liye `max_seq_len` (say 8192) space **pehle se** allocate. Problem:

Maan lo ek request ka actual output 300 tokens hai, lekin humne 8192 ke liye reserve kiya.
**Waste = (8192 − 300) / 8192 = 96%**

Ek study mein paya gaya ki naive systems mein KV memory ka **60-80% waste** hota hai:
- **Internal fragmentation:** reserved but unused (upar wala case)
- **External fragmentation:** allocation blocks ke beech ke gaps
- **Reservation waste:** future tokens ke liye held space

### PagedAttention ka insight

OS ke virtual memory se idea uthao. KV cache ko **fixed-size blocks** (typically 16 tokens) mein baanto, aur ek **block table** rakho jo logical positions ko physical blocks par map kare.

```
Logical view (sequence ko lagta hai contiguous hai):
  [tok 0..15][tok 16..31][tok 32..47]

Physical view (GPU memory mein bikhre hue):
  Block 42: tok 0..15
  Block  7: tok 16..31
  Block 91: tok 32..47

Block table: seq_A → [42, 7, 91]
```

Faayde:
- Waste **< 4%** (sirf last block partially filled)
- **Copy-on-write sharing** — beam search / parallel sampling mein prefix blocks share ho sakte hain
- **Prefix caching** — same system prompt wali requests blocks share karti hain
- Dynamic growth bina reallocation ke

Ye vLLM ka core contribution hai (Kwon et al., SOSP 2023). Module 6 mein poora detail karenge, including block table lookups attention kernel ke andar kaise hote hain.

## 3.5 KV Cache decode mein bandwidth bhi khaata hai

Sirf storage nahi — har decode step par **poora KV cache padhna padta hai**.

Batch=32, context=4096, Llama-3-8B:
```
KV cache size = 128 KB × 4096 × 32 = 16.8 GB
Weights       = 16 GB
─────────────────────────────────────
Har decode step par HBM read = 32.8 GB
```

A100-80 par (2 TB/s):
```
32.8 GB / 2000 GB/s = 16.4 ms per step
→ max ~61 steps/sec
→ 61 × 32 = ~1950 tokens/sec throughput
```

Notice: **KV cache reads ab weights reads jitne mehnge ho gaye hain.** Long context par KV reads dominate kar lete hain. Isliye:
- KV cache quantization (FP8/INT4) direct speedup deta hai, sirf memory saving nahi
- FlashDecoding jaise kernels KV reads ko optimize karte hain
- Sparse attention (sirf important KV padho) ek active research area hai

## Module 3 Exercises

1. Ek chhote model (GPT-2 small) ke liye KV cache **manually implement** karo — bina `use_cache=True` ke. Verify karo ki output naive recompute se bit-exact match karta hai.
2. Llama-3-70B (n_layers=80, n_kv_heads=8, head_dim=128) ke liye 128K context par KV cache size nikalo, batch=1. Kitne H100 chahiye sirf KV ke liye?
3. Derive karo: kis context length par KV cache reads model weight reads se zyada ho jaate hain, batch=B ke function mein? (Llama-3-8B use karo)
4. **Padho:** *"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"* (Ainslie et al., 2023). Uptraining procedure kya hai?

---
---

# Module 4 — Roofline Model (jo sab kuch explain karta hai)

Ye module Module 0 aur 3 ko jodta hai. Iske baad aap kisi bhi inference optimization ko dekhkar bata paoge ki woh kaam karegi ya nahi.

## 4.1 Arithmetic Intensity

```
I = FLOPs performed / Bytes moved from HBM        [FLOP/byte]
```

Ek operation ki fundamental property hai ye. Compare karo GPU ke **ridge point** se:

```
Ridge point = Peak FLOPS / Peak Bandwidth
```

**Regla:**
```
I < ridge point   →  MEMORY-BOUND    (GPU memory ka intezaar kar raha)
I > ridge point   →  COMPUTE-BOUND   (GPU actually kaam kar raha)
```

Attainable performance:
```
Perf = min(Peak_FLOPS,  I × Bandwidth)
```

```
Performance
  ↑
  │            ┌────────────── Peak FLOPS (compute ceiling)
  │           ╱
  │          ╱  ← slope = memory bandwidth
  │         ╱
  │        ╱
  └───────┴────────────────────→ Arithmetic Intensity
       ridge point
   ↑                    ↑
 DECODE               PREFILL
 yahan hai            yahan hai
```

## 4.2 Decode ka analysis — the brutal truth

**Setup:** Llama-3-8B, BF16, batch=1, A100-80GB.

**Ek decode step par bytes read:**
```
Weights: 8.03 B params × 2 bytes = 16.06 GB
KV cache (2K context): 128 KB × 2048 = 0.26 GB
─────────────────────────────────────────────
Total ≈ 16.3 GB
```

**Ek decode step par FLOPs:**
```
≈ 2 × params = 2 × 8.03 B = 16.06 GFLOP
```
(Ye standard approximation hai: har parameter ek multiply + ek add mein participate karta hai.)

**Arithmetic intensity:**
```
I = 16.06 GFLOP / 16.3 GB ≈ 0.99 FLOP/byte
```

**A100-80 ka ridge point = 153 FLOP/byte.**

```
0.99 << 153   →  BURI TARAH memory-bound
```

**Utilization = 0.99/153 = 0.65%**

Yani GPU ki compute capacity ka **99.35% idle** hai.

**Time per token:**
```
Memory time:  16.3 GB / 2039 GB/s  = 8.0 ms   ← ye bottleneck hai
Compute time: 16.06 GFLOP / 312 TFLOPS = 0.05 ms
```

**160× gap.** Max ~125 tokens/sec at batch=1. Aur ye theoretical upper bound hai — real systems 60-70% achieve karte hain.

> **Ye single calculation LLM inference ki poori field ko explain karti hai.** Har optimization — quantization, batching, speculative decoding, MLA — ya to bytes kam karti hai, ya un bytes se zyada kaam nikalti hai.

## 4.3 Prefill ka analysis

**Setup:** Same model, prompt = 2048 tokens.

```
FLOPs = 2 × params × tokens = 2 × 8.03B × 2048 = 32.9 TFLOP
       (+ attention ka S² term, jo yahan ~1-2% hai)

Bytes = 16.06 GB (weights, sirf EK BAAR — sab tokens share karte hain)
        + activations (relatively chhota)

I = 32.9 TFLOP / ~17 GB ≈ 1935 FLOP/byte
```

```
1935 >> 153   →  COMPUTE-BOUND ✓
```

**Time:**
```
Compute: 32.9 TFLOP / 312 TFLOPS = 105 ms   ← bottleneck
Memory:  17 GB / 2039 GB/s = 8.3 ms
```

Ab GPU actually kaam kar raha hai. Utilization achhi hai.

## 4.4 Batching kyun magic hai

Ab dekho batch size badhane se kya hota hai. Weights **share** hote hain — batch=1 ho ya batch=64, weights ek hi baar padhne hain.

Llama-3-8B, 2K context, A100-80:

| Batch | Bytes read | FLOPs | Intensity | Bound | tok/s (total) |
|-------|-----------|-------|-----------|-------|---------------|
| 1 | 16.3 GB | 16 GF | 0.99 | Memory | ~125 |
| 8 | 18.1 GB | 128 GF | 7.1 | Memory | ~900 |
| 32 | 24.4 GB | 514 GF | 21 | Memory | ~2,600 |
| 64 | 32.8 GB | 1028 GF | 31 | Memory | ~3,900 |
| 128 | 49.6 GB | 2056 GF | 41 | Memory | ~5,100 |

**Observations:**

1. Throughput batch=1 se batch=128 tak **~40× badhta hai** — same hardware par!
2. Lekin intensity abhi bhi 41 hai, ridge point 153 se kam. **Batch=128 par bhi hum memory-bound hain.**
3. Per-token latency badhti hai (batch=128 par ~25 ms/step vs 8 ms), lekin throughput bahut zyada.
4. Batch aur badhaya nahi ja sakta — KV cache memory khatam ho jaayegi.

Point 2 dilchasp hai: **decode ko compute-bound banane ke liye batch ~150-200+ chahiye**, lekin KV cache us se pehle memory bhar deta hai. Ye tension "KV cache chhota karo" wali saari research ko drive karti hai — chhota KV = bada batch = behtar utilization.

## 4.5 Har optimization ko roofline lens se dekho

Ab aap kisi bhi technique ko classify kar sakte ho:

| Technique | Kya karta hai | Roofline effect |
|-----------|--------------|-----------------|
| **Weight quantization (INT4)** | Weight bytes ÷4 | I ×4 → direct decode speedup |
| **KV cache quantization** | KV bytes ÷2 or ÷4 | I ↑, long context par bada faayda |
| **Continuous batching** | Batch effectively ↑ | I ↑, GPU idle kam |
| **GQA / MLA** | KV bytes ↓ | I ↑, + bada batch possible |
| **FlashAttention** | HBM traffic ↓ (tiling) | I ↑ attention ke andar |
| **Speculative decoding** | Ek weight-read mein k tokens | I ×k (effective) |
| **MoE** | FLOPs ↓, weight-read same-ish | I ↓ — **memory-bound hota hai zyada!** |
| **Tensor parallelism** | Bandwidth aggregate ↑ | Absolute BW ↑, comm overhead cost |

Dhyaan do MoE wali line: MoE FLOPs kam karta hai lekin decode already compute-bound nahi hai. Isliye MoE ka decode speedup utna nahi milta jitna FLOP-count se lagta hai. Ye ek common misconception hai.

**Speculative decoding wali line poori tarah samjho:** Agar hum ek forward pass mein 4 tokens verify kar sakein, to same 16 GB weight-read se 4× kaam nikala. Intensity 4× ho gayi. Yahi speculative decoding ka asli mechanism hai — "guessing" nahi, **memory-bandwidth amortization**. Module 9 mein detail.

## 4.6 Ek practical roofline calculator

```python
def analyze(params_B, layers, kv_heads, head_dim, 
            batch, ctx, bytes_w=2, bytes_kv=2,
            peak_tflops=312, bw_gbs=2039):
    
    w_bytes  = params_B * 1e9 * bytes_w
    kv_bytes = 2 * layers * kv_heads * head_dim * bytes_kv * ctx * batch
    total_b  = w_bytes + kv_bytes
    
    flops = 2 * params_B * 1e9 * batch
    
    I = flops / total_b
    ridge = peak_tflops * 1e12 / (bw_gbs * 1e9)
    
    t_mem = total_b / (bw_gbs * 1e9)
    t_cmp = flops / (peak_tflops * 1e12)
    t     = max(t_mem, t_cmp)
    
    print(f"Batch {batch:4d} | ctx {ctx:6d}")
    print(f"  Bytes/step  : {total_b/1e9:7.2f} GB "
          f"(weights {w_bytes/1e9:.1f}, kv {kv_bytes/1e9:.1f})")
    print(f"  Intensity   : {I:7.2f} FLOP/byte  (ridge {ridge:.0f})")
    print(f"  Bound       : {'MEMORY' if I < ridge else 'COMPUTE'}")
    print(f"  Latency     : {t*1000:7.2f} ms/step")
    print(f"  Throughput  : {batch/t:7.0f} tok/s")
    print(f"  MFU         : {t_cmp/t*100:7.2f}%\n")

# Llama-3-8B
for b in [1, 8, 32, 64, 128]:
    analyze(8.03, 32, 8, 128, batch=b, ctx=2048)
```

Isko chalao. Phir knobs ghumao:
- `bytes_w=1` (INT8 quantization) — kitna faayda?
- `bytes_kv=1` (FP8 KV cache) — long context par kya hota hai?
- `ctx=32768` — kahan cheezein toot jaati hain?
- H100 numbers daalo (`peak_tflops=990, bw_gbs=3350`)

Ye calculator aapka "napkin math" tool ban jaayega. Har naye paper ko padhte waqt, pehle isme numbers daalkar dekho ki claim believable hai ya nahi.

## Module 4 Exercises

1. Calculator chalao aur pata karo: Llama-3-8B ko A100-80 par **compute-bound** banane ke liye kitna batch chahiye (2K context)? Kya utna KV cache memory mein fit hoga?
2. INT4 weights + FP8 KV cache ke saath, batch=32, ctx=8192 par theoretical throughput kya hai? BF16 se kitna better?
3. **Real measurement:** ek GPU par `nvidia-smi dmon` ya Nsight se actual bandwidth utilization measure karo decode ke dauraan. Theory se kitna gap hai? Gap ka kya reason hai?
4. Derive karo: kis batch size par KV cache bytes = weight bytes ho jaate hain, context length ke function mein?

---
---

# Papers — Part 1 ke liye reading list

Priority order mein. Har paper padhkar 5-line summary likho apne notes mein.

**Zaroori (abhi padho):**
1. **Attention Is All You Need** (Vaswani et al., 2017) — origin
2. **Efficiently Scaling Transformer Inference** (Pope et al., 2022) — ye Module 4 ka canonical paper hai, roofline analysis ka gold standard
3. **GQA** (Ainslie et al., 2023)
4. **RoFormer / RoPE** (Su et al., 2021)
5. **Efficient Memory Management for LLM Serving with PagedAttention** (Kwon et al., 2023) — vLLM

**Aage ke liye (Module 5-9 mein use honge):**
6. Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019) — MQA
7. Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022) — continuous batching
8. FlashAttention (Dao et al., 2022) + FlashAttention-2 (2023)
9. DeepSeek-V2 / V3 technical reports — MLA
10. Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)

**Survey (overview ke liye achha):**
- *Towards Efficient Generative LLM Serving: A Survey* — landscape samajhne ke liye
- *LLM Inference Unveiled: Survey and Roofline Model Insights*

---

# Part 1 ka Capstone Project

Ye seriously karo. Ek weekend lagega, lekin isse depth aa jaayegi jo padhne se kabhi nahi aati.

**Task:** Llama-3-8B (ya TinyLlama agar GPU chhoti hai) ke liye **pure PyTorch mein inference engine likho** — koi HuggingFace `generate()` nahi.

Requirements:
1. Model weights load karo (safetensors se)
2. Forward pass khud implement karo: RMSNorm, RoPE, GQA, SwiGLU
3. KV cache manually manage karo
4. Prefill aur decode alag-alag code paths
5. Top-p sampling
6. Instrument karo: TTFT, TPOT, tokens/sec log karo

Phir:
7. Apne measured numbers ko Module 4 ke roofline predictions se compare karo
8. Gap analyze karo — kahan time ja raha hai? (`torch.profiler` use karo)
9. Ek optimization apply karo (jaise `torch.compile`) aur dobara measure karo

**Deliverable:** Ek chhoti report jisme aapke measured numbers, theoretical predictions, aur gap ka explanation ho.

Ye kar liya to aap top 5% mein ho jaoge un logon mein jo "LLM inference jaante hain".

---

# Aage kya (Part 2 mein)

- **Module 5:** Continuous batching — Orca ka iteration-level scheduling, chunked prefill
- **Module 6:** PagedAttention deep dive — block tables, copy-on-write, prefix caching, radix trees (SGLang)
- **Module 7:** FlashAttention — online softmax derivation, tiling, FA-2/FA-3, FlashDecoding

Batao jab ready ho, ya koi specific module pehle chahiye to woh bhi bata do.
