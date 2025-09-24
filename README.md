# LLM Learning Curriculum

This repository contains a structured, hands-on curriculum for learning Large Language Models (LLMs) from the ground up. The materials are organized into a weekly, day-by-day plan that progresses from fundamental concepts to advanced topics. Each week's folder contains self-contained Jupyter notebooks with practical implementations of the concepts discussed.

---

### Visual guide reference - https://bbycroft.net/llm

---

### Week 1: Foundations

- **Topics Covered**: Text Normalization, Tokenization (BPE/WordPiece), Embeddings, and Positional Encodings.
- **Outcome**: Built a foundational understanding of how text is processed and prepared for a transformer model.
- **Code**: See the `week1` directory for hands-on implementations.

### Week 2: Transformer Architecture

- **Topics Covered**: Scaled Dot-Product Attention, Multi-Head Attention, Encoder-Decoder Architectures, and building a minimal transformer block from scratch.
- **Outcome**: Implemented the core components of the transformer architecture.
- **Code**: See the `week2` directory, which includes a complete, minimal transformer implementation.

### Week 3: Pretraining a Language Model

- **Topics Covered**: Language modeling datasets, perplexity, large-scale tokenization, sequence packing, and building a complete training loop with regularization (Dropout, Weight Decay), monitoring, and early stopping.
- **Outcome**: Developed a robust pipeline for pretraining a GPT-style model and implemented a capstone project inspired by `nanoGPT`.
- **Code**: See the `week3` directory for a full pretraining implementation.

### Week 4: Instruction Tuning & Alignment

- **Topics**: Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and Direct Preference Optimization (DPO).
- **Objective**: Learn how to align a pretrained model to follow instructions and adhere to human preferences.

### Week 5: Prompt Engineering & Retrieval-Augmented Generation (RAG)

- **Topics**: Advanced prompting techniques (Chain-of-Thought), function calling, and building RAG pipelines with vector stores and rerankers.
- **Objective**: Enhance model capabilities by integrating external knowledge and tools.

### Week 6: Advanced Fine-Tuning Techniques

- **Topics**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA and QLoRA, and knowledge distillation.
- **Objective**: Learn to efficiently adapt large models for specific tasks on consumer-grade hardware.

### Week 7: Inference, Serving, and Optimization

- **Topics**: Model quantization, KV caching, and deploying models with high-throughput servers like vLLM and TGI.
- **Objective**: Understand how to optimize and serve LLMs for production environments.

### Week 8: Agents, Tools, and Safety

- **Topics**: Building autonomous agents that can use tools, manage memory, and operate within safety guardrails.
- **Objective**: Develop a multi-step agent capable of solving complex tasks and completing the capstone project.
