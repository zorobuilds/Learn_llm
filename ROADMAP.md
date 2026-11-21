# 🗺️ TorchPy: The 4-Season Master Roadmap
> **From First Principles to a 42M Chat-Aligned LLM**

This document outlines the entire curriculum of TorchPy across 4 distinct seasons. Every single concept is introduced with visual intuition first, mathematical rigor second, and runnable Python/PyTorch code third.

---

## 🟢 Season 1: Neural Foundations (Episodes 1 – 5)
*Goal: Demystify how numbers become decisions.*

* **Ep 1: The Philosophy, Intuition & The 4-Season Roadmap**
  * Why learning AI through high-level APIs fails.
  * The anatomy of a neuron: Inputs ($x$), Weights ($w$), Bias ($b$), and Activation ($\sigma$).
* **Ep 2: The Single Neuron in Pure Python**
  * Building a neuron without PyTorch or NumPy.
  * The weighted sum $z = \sum w_i x_i + b$ and the Sigmoid squashing function.
* **Ep 3: Forward Propagation & Tensor Shapes**
  * Transitioning from scalar loops to 2D matrix multiplication ($X \cdot W^T + b$).
  * Understanding batching, dimensional alignment, and tensor broadcasting.
* **Ep 4: Backpropagation & Gradient Descent Intuition**
  * Loss functions (Mean Squared Error & Binary Cross-Entropy).
  * The Chain Rule visually unfolded: How error gradients flow backward to update weights.
* **Ep 5: Our First Multi-Layer Perceptron (MLP) in PyTorch**
  * Connecting multiple neurons into hidden layers.
  * Solving non-linear decision boundaries and transitioning into PyTorch `nn.Module`.

---

## 🟡 Season 2: From Text to Attention (Episodes 6 – 10)
*Goal: Transforming unstructured raw text into geometric vector spaces.*

* **Ep 6: Tokenization & Word Embeddings**
  * Why models can't read strings: Byte-Pair Encoding (BPE) & token IDs.
  * Embedding lookup tables: Mapping discrete token IDs to dense vectors ($d_{model}$).
* **Ep 7: The Self-Attention Mechanism (Q, K, V)**
  * The intuition: Words negotiating meaning based on context.
  * Queries (what I'm looking for), Keys (what I offer), and Values (what information I carry).
* **Ep 8: Scaled Dot-Product Math & Causal Masking**
  * Calculating attention scores: $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$.
  * Why we divide by $\sqrt{d_k}$ (preventing vanishing softmax gradients).
  * The lower-triangular causal mask: Preventing the model from cheating on future tokens.
* **Ep 9: Multi-Head Attention & Rotary Position Embeddings (RoPE)**
  * Splitting embedding channels into parallel attention heads.
  * Why absolute positional embeddings fail on long sequences and how RoPE rotates vectors in 2D pairs.
* **Ep 10: The Complete Transformer Decoder Block**
  * Combining Multi-Head Attention, Feed-Forward Networks (MLP), Residual Connections, and Layer Normalization.

---

## 🟣 Season 3: Baseline GPT & The Failure Post-Mortem (Episodes 11 – 15)
*Goal: Building our first generative models (V1 & V2) and learning from architectural failure.*

* **Ep 11: Stacking Decoder Blocks into a GPT Model**
  * Assembling an autoregressive language model from scratch.
  * The generation loop: Logits, Temperature, Top-$k$ sampling, and Repetition Penalties.
* **Ep 12: The End-to-End Training Loop**
  * Data loader pipeline on the TinyStories dataset.
  * Mixed precision (FP16/BF16), AdamW optimizer, and Cosine Annealing learning rate schedules.
* **Ep 13: The V1/V2 Post-Mortem: Why 40 Million Parameters Failed**
  * **The Single-Block Depth Wall:** Why putting 40M parameters into a single massive decoder layer collapsed reasoning depth.
  * **The Untied Weight Trap:** Why un-tying token embeddings and the LM head caused 26M redundant parameters and severe overfitting.
* **Ep 14: Analyzing Attention Collapse & Loss Spikes**
  * Visualizing attention heatmaps when models fail to generalize.
  * Debugging exploding gradients and loss plateaus.
* **Ep 15: The Architectural Pivot: Designing the Modern V3 & V4 LLM**
  * Transitioning from legacy GPT-2 designs to modern open-source foundations (LLaMA style).

---

## 🔵 Season 4: Modern LLM & Conversational Alignment (Episodes 16 – 20)
*Goal: Building, pretraining, and aligning a production-grade 42M parameter chat assistant.*

* **Ep 16: Modern Architecture: RMSNorm, SwiGLU & GQA**
  * Replacing LayerNorm with computationally efficient Root Mean Square Normalization (RMSNorm).
  * Replacing standard ReLU/GELU with Swish-Gated Linear Units (SwiGLU).
  * Grouped-Query Attention (GQA) for memory-efficient key-value caching.
* **Ep 17: Vocabulary Surgery: Pruning GPT-2 to 16,384 Tokens**
  * Why standard 50,257 vocabularies waste tens of millions of parameters on small models.
  * Pruning the embedding matrix down to 16k tokens to reallocate parameter budget into depth (12 stacked layers).
* **Ep 18: 400M-Token Pretraining Run on Real Hardware**
  * Multi-day training run on custom datasets.
  * Monitoring training loss, validation perplexity, gradient norms, and throughput (tokens/sec).
* **Ep 19: Conversational SFT (Supervised Fine-Tuning)**
  * Formatting multi-turn chat dialogues with role markers (`<|user|>`, `<|assistant|>`).
  * **Assistant-Masked Loss:** Masking user prompt tokens so gradients only calculate on the assistant's responses.
* **Ep 20: Quantization, Evaluation & Deployment**
  * Evaluating model outputs on reasoning, story-telling, and instruction-following benchmarks.
  * Exporting to GGUF / ONNX for local real-time CPU/GPU inference.

---

### 📊 Model Architecture Evolution Summary

| Specification | V1 (Baseline) | V2 (Single-Block) | V3 (Modern Pretrained) | V4 (Chat-Aligned SFT) |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters** | 40.2M | 40.2M | 42.1M | 42.1M |
| **Stacked Layers** | 1 (Wide) | 1 (Wide) | **12 (Deep)** | **12 (Deep)** |
| **Hidden Dimension ($d_{model}$)** | 1024 | 1024 | 512 | 512 |
| **Vocabulary Size** | 50,257 (GPT-2) | 50,257 (GPT-2) | **16,384 (Pruned)** | **16,384 (Pruned)** |
| **Norm Layer** | LayerNorm | LayerNorm | **RMSNorm** | **RMSNorm** |
| **MLP Activation** | GELU | GELU | **SwiGLU** | **SwiGLU** |
| **Positional Encoding** | Learned Absolute | Learned Absolute | **RoPE** | **RoPE** |
| **Alignment Stage** | Raw Pretrain | Raw Pretrain | Base Pretrained | **Stage 2 Masked SFT** |
