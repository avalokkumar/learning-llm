# LLM Learning Curriculum

This repository contains a structured, hands-on curriculum for learning Large Language Models (LLMs) from the ground up. The materials are organized into a weekly, day-by-day plan that progresses from fundamental concepts to advanced topics. Each week's folder contains self-contained Jupyter notebooks with practical implementations of the concepts discussed.

---

### Visual guide reference - https://bbycroft.net/llm

---

## 📚 Week 1: Text Processing Fundamentals

### **Day 1: Text Normalization and Tokenization**
- **Core Concepts**: Understanding how raw text becomes model-ready tokens
- **Key Topics**:
  - Text normalization (Unicode, case folding, whitespace handling)
  - Tokenization strategies: Word-level, Character-level, Subword-level
  - **BPE (Byte-Pair Encoding)**: Iterative merging of frequent character pairs
  - **WordPiece**: Likelihood-based tokenization used in BERT
  - Vocabulary size vs. sequence length trade-offs
- **Key Takeaway**: Subword tokenization provides optimal balance between vocabulary size and sequence length

### **Day 2: Advanced Tokenization Libraries**
- **Core Concepts**: Production-grade tokenization with industry tools
- **Key Topics**:
  - **tiktoken**: OpenAI's fast C++ tokenizer for GPT models
  - **HuggingFace tokenizers**: Flexible Rust-based library supporting multiple algorithms
  - Vocabulary construction and optimization
  - BPE merge operations in detail
  - Domain-specific tokenizer design
- **Key Takeaway**: Choose tiktoken for OpenAI models, HuggingFace tokenizers for custom implementations

### **Day 3: Embeddings and Semantic Representations**
- **Core Concepts**: Converting tokens to dense vector representations
- **Key Topics**:
  - **One-hot encodings**: Sparse, high-dimensional representations
  - **Learned embeddings**: Dense vectors capturing semantic relationships
  - **Cosine similarity**: Measuring semantic relationships between embeddings
  - Training embeddings with context (Skip-gram, CBOW)
  - Embedding visualization and analysis
  - Pre-trained embeddings (Word2Vec, GloVe concepts)
- **Key Takeaway**: Learned embeddings capture semantic meaning in lower-dimensional space

### **Day 4: Positional Encodings**
- **Core Concepts**: Adding sequence order information to transformers
- **Key Topics**:
  - Why transformers need positional information
  - **Sinusoidal encodings**: Mathematical position representations using sine/cosine
  - **RoPE (Rotary Position Embedding)**: Modern rotation-based approach
  - **Learned positional embeddings**: Trainable position representations
  - Comparing different positional encoding strategies
- **Key Takeaway**: Positional encodings restore sequence order information lost in parallel processing

### **Day 5: Complete Preprocessing Pipeline**
- **Core Concepts**: Integrating all components into end-to-end pipeline
- **Key Topics**:
  - Building complete text → tokens → embeddings → positional encoding pipeline
  - Special tokens handling (PAD, CLS, SEP, UNK)
  - Attention masks for variable-length sequences
  - Padding and truncation strategies
  - Pipeline validation and testing
- **Key Takeaway**: A well-designed pipeline is crucial for transformer model success

---

## 🏗️ Week 2: Transformer Architecture

### **Day 6: Scaled Dot-Product Attention**
- **Core Concepts**: The heart of transformer models
- **Key Topics**:
  - **Attention formula**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
  - Query, Key, Value intuition (database analogy)
  - **Scaling by √d_k**: Prevents softmax saturation
  - Attention patterns and their meanings
  - Self-attention vs. cross-attention
  - Computational complexity: O(n²d)
- **Key Takeaway**: Attention enables parallel processing while capturing dependencies

### **Day 7: Multi-Head Attention**
- **Core Concepts**: Parallel attention mechanisms for diverse relationships
- **Key Topics**:
  - Multiple attention heads capturing different relationship types
  - Head specialization (syntactic, semantic, positional)
  - Concatenation and projection operations
  - Ensemble learning perspective
  - Computational efficiency through parallelization
- **Key Takeaway**: Multiple heads capture different types of relationships simultaneously

### **Day 8: Transformer Architectures**
- **Core Concepts**: Three fundamental transformer variants
- **Key Topics**:
  - **Encoder-only (BERT-style)**: Bidirectional processing for understanding tasks
  - **Decoder-only (GPT-style)**: Autoregressive generation with causal masking
  - **Encoder-decoder (T5-style)**: Sequence-to-sequence with cross-attention
  - Causal masking mathematics and implementation
  - Architecture selection based on task requirements
- **Key Takeaway**: Architecture choice depends on task: understanding vs. generation vs. seq2seq

### **Day 9: Transformer Components**
- **Core Concepts**: Supporting components for stable training
- **Key Topics**:
  - **Residual connections**: Enable gradient flow in deep networks
  - **Layer normalization**: Stabilizes training across features
  - **GELU activation**: Smooth, probabilistic activation function
  - **Feed-forward networks**: Position-wise transformations
  - Pre-norm vs. post-norm architectures
- **Key Takeaway**: These components work synergistically to enable stable deep network training

### **Day 10: Building Complete Transformer**
- **Core Concepts**: Integrating all components into working model
- **Key Topics**:
  - Complete minimal transformer implementation
  - Proper weight initialization strategies
  - Training and evaluation loops
  - Text generation with autoregressive sampling
  - Performance analysis and optimization
  - Connection to real-world models (GPT, BERT)
- **Key Takeaway**: Understanding implementation details is crucial for working with transformers

---

## 🎯 Week 3: Training and Pretraining

### **Day 11: Datasets and Perplexity**
- **Core Concepts**: Data foundation for language models
- **Key Topics**:
  - Common pretraining datasets (OpenWebText, The Pile, C4, BookCorpus)
  - Data quality considerations (deduplication, filtering, balancing)
  - Train/validation splits for language modeling
  - **Perplexity**: Standard evaluation metric (exponential of cross-entropy)
  - Creating toy datasets for experimentation
- **Key Takeaway**: High-quality, diverse datasets are fundamental to effective LLMs

### **Day 12: Tokenization at Scale and Sequence Preparation**
- **Core Concepts**: Efficient data processing for large-scale training
- **Key Topics**:
  - Tokenization efficiency and optimization
  - Sequence length considerations and memory trade-offs
  - **Sequence packing**: Maximizing GPU utilization
  - **Causal masking**: Preventing future token leakage
  - Efficient data pipelines with PyTorch DataLoader
- **Key Takeaway**: Efficient data processing dramatically improves training throughput

### **Day 13: Training Loop Details**
- **Core Concepts**: Robust and efficient training implementation
- **Key Topics**:
  - **Mixed precision training (AMP)**: FP16/FP32 for speed and memory savings
  - **Gradient clipping**: Preventing exploding gradients
  - **Gradient accumulation**: Effective larger batch sizes
  - **AdamW optimizer**: Decoupled weight decay for LLMs
  - **Learning rate schedules**: Warmup + cosine decay
- **Key Takeaway**: Modern training techniques are essential for stable LLM training

### **Day 14: Regularization and Monitoring**
- **Core Concepts**: Preventing overfitting and tracking progress
- **Key Topics**:
  - **Dropout**: Random neuron deactivation for regularization
  - **Weight decay**: L2 regularization in optimizer
  - **TensorBoard and Weights & Biases**: Monitoring tools
  - **Early stopping**: Preventing overfitting automatically
  - Diagnosing overfitting vs. underfitting
  - Comprehensive logging systems
- **Key Takeaway**: Effective monitoring and regularization ensure optimal model performance

### **Day 15: Pretraining nanoGPT-Style Model**
- **Core Concepts**: End-to-end pretraining from scratch
- **Key Topics**:
  - Complete pretraining pipeline integration
  - Building decoder-only transformer
  - Causal language modeling objective
  - Training on custom corpus
  - Text generation and evaluation
  - Scaling laws and compute-optimal training
- **Key Takeaway**: Pretraining creates base models that learn statistical patterns of language

---

## 🎨 Week 4: Alignment and Evaluation

### **Day 16: Supervised Fine-Tuning (SFT)**
- **Core Concepts**: Teaching models to follow instructions
- **Key Topics**:
  - Instruction-response pair datasets (Alpaca, Dolly, FLAN)
  - SFT pipeline: Pre-trained → Instruction tuning → Instruction-following model
  - Training parameters (lower learning rates, fewer epochs)
  - Common challenges (overfitting, prompt sensitivity)
  - Parameter-efficient fine-tuning (LoRA, QLoRA)
- **Key Takeaway**: SFT bridges gap between language understanding and instruction-following

### **Day 17: Reinforcement Learning from Human Feedback (RLHF)**
- **Core Concepts**: Aligning models with human preferences
- **Key Topics**:
  - **Three-stage pipeline**: SFT → Reward Model → PPO optimization
  - **Reward modeling**: Training models to predict human preferences
  - **PPO (Proximal Policy Optimization)**: RL algorithm for language models
  - KL divergence penalty to prevent drift
  - Comparison to SFT: Better for nuanced preferences
  - Challenges: Reward hacking, training instability
- **Key Takeaway**: RLHF enables nuanced alignment beyond simple imitation

### **Day 18: Direct Preference Optimization (DPO)**
- **Core Concepts**: Simplified alternative to RLHF
- **Key Topics**:
  - **DPO objective**: Direct optimization from preference data
  - Eliminating separate reward model and PPO
  - Mathematical foundations and derivation
  - Comparison to RLHF: Simpler, more stable, comparable results
  - Hyperparameter β controls preference strength
  - Advanced variants (IPO, KTO)
- **Key Takeaway**: DPO simplifies alignment while achieving comparable or better results

### **Day 19: Safety Basics**
- **Core Concepts**: Ensuring responsible model behavior
- **Key Topics**:
  - Safety risk categories (harmful content, misinformation, illegal activities)
  - **Refusal policies**: When and how to decline requests
  - **Red-teaming**: Adversarial testing for vulnerabilities
  - **Content moderation**: Toxicity filters and detection
  - Drafting comprehensive safety policies
  - Constitutional AI and advanced safety techniques
- **Key Takeaway**: Safety is fundamental for responsible LLM deployment

### **Day 20: Evaluation Suites**
- **Core Concepts**: Systematic assessment of model capabilities
- **Key Topics**:
  - Popular benchmarks:
    - **MMLU**: 57 subjects testing knowledge and reasoning
    - **HellaSwag**: Common sense reasoning
    - **TruthfulQA**: Truthfulness and avoiding misconceptions
    - **GSM8K**: Mathematical reasoning
    - **HumanEval**: Code generation
  - Custom task evaluations and scoring rubrics
  - **Measuring hallucinations**: Factual accuracy, context adherence
  - **Reducing hallucinations**: RAG, self-consistency, uncertainty expression
  - Creating model report cards
- **Key Takeaway**: Comprehensive evaluation guides model improvement and ensures reliability

---

## 📊 Architecture Comparison Summary

| Architecture | Best For | Key Feature | Example Models |
|--------------|----------|-------------|----------------|
| **Encoder-only** | Understanding tasks | Bidirectional context | BERT, RoBERTa |
| **Decoder-only** | Generation tasks | Autoregressive with causal masking | GPT-3/4, LLaMA |
| **Encoder-decoder** | Seq2seq tasks | Cross-attention bridge | T5, BART |

---

## 🔧 Training Techniques Summary

| Technique | Purpose | Key Benefit |
|-----------|---------|-------------|
| **Mixed Precision** | Speed & memory | 2x memory reduction, faster training |
| **Gradient Clipping** | Stability | Prevents exploding gradients |
| **Gradient Accumulation** | Effective batch size | Train with larger batches on limited memory |
| **AdamW** | Optimization | Proper weight decay for LLMs |
| **Learning Rate Scheduling** | Convergence | Warmup + decay improves final performance |
| **Dropout** | Regularization | Prevents overfitting |
| **Early Stopping** | Efficiency | Stops training at optimal point |

---

## 🎯 Alignment Techniques Comparison

| Method | Complexity | Compute Cost | Stability | Performance |
|--------|------------|--------------|-----------|-------------|
| **SFT** | Low | Low | High | Good |
| **RLHF** | High | High | Medium | Excellent |
| **DPO** | Medium | Medium | High | Excellent |

---

## 📈 Evaluation Benchmarks Summary

| Benchmark | Tests | Format | Key Metric |
|-----------|-------|--------|------------|
| **MMLU** | Knowledge across 57 subjects | Multiple choice | Accuracy |
| **HellaSwag** | Common sense reasoning | Sentence completion | Accuracy |
| **TruthfulQA** | Truthfulness | Q&A with misconceptions | Truth score |
| **GSM8K** | Math reasoning | Word problems | Accuracy |
| **HumanEval** | Code generation | Function completion | Pass@k |

---

## 🚀 From Theory to Practice: Complete Pipeline

```
1. DATA PREPARATION
   ├─ Collect diverse text corpus
   ├─ Train tokenizer (BPE/WordPiece)
   ├─ Create train/val splits
   └─ Implement efficient data loading

2. MODEL ARCHITECTURE
   ├─ Token embeddings
   ├─ Positional encodings
   ├─ Multi-head attention layers
   ├─ Feed-forward networks
   └─ Output projection

3. PRETRAINING
   ├─ Causal language modeling
   ├─ Mixed precision training
   ├─ Gradient clipping & accumulation
   ├─ Learning rate scheduling
   └─ Regular checkpointing

4. ALIGNMENT
   ├─ Supervised Fine-Tuning (SFT)
   ├─ Collect preference data
   ├─ Apply RLHF or DPO
   └─ Implement safety measures

5. EVALUATION
   ├─ Standard benchmarks (MMLU, etc.)
   ├─ Custom task evaluation
   ├─ Safety testing
   ├─ Hallucination measurement
   └─ Create model report card

6. DEPLOYMENT
   ├─ Optimize for inference
   ├─ Implement content filters
   ├─ Set up monitoring
   └─ Continuous evaluation
```

---

## 💡 Best Practices Learned

### **Data**
- Use diverse, high-quality training data
- Implement proper deduplication
- Balance different domains appropriately
- Respect data privacy and licensing

### **Training**
- Start with strong baselines
- Use appropriate learning rates (lower for fine-tuning)
- Monitor training closely with comprehensive logging
- Save checkpoints regularly
- Implement early stopping

### **Architecture**
- Choose architecture based on task requirements
- Use pre-norm for better stability
- Implement proper initialization
- Balance model size with compute budget

### **Alignment**
- Start with high-quality SFT
- Collect diverse preference data
- Balance helpfulness, harmlessness, and honesty
- Iterate based on evaluation results

### **Safety**
- Implement multiple layers of safety measures
- Conduct thorough red-team testing
- Create clear refusal policies
- Monitor deployed models continuously

### **Evaluation**
- Use multiple evaluation metrics
- Include both automatic and human evaluation
- Test on diverse scenarios and edge cases
- Create comprehensive model documentation

---

## 🔮 Future Directions

### **Emerging Techniques**
- Mixture of Experts (MoE) for efficient scaling
- Sparse attention mechanisms for longer contexts
- Multimodal models (text + images + audio)
- More efficient alignment methods
- Better interpretability tools

### **Open Challenges**
- Reducing hallucinations further
- Improving reasoning capabilities
- Handling very long contexts efficiently
- Ensuring fairness and reducing bias
- Making models more sample-efficient

### **Research Frontiers**
- Constitutional AI and value alignment
- Scalable oversight techniques
- Emergent capabilities in large models
- Efficient training and inference
- Robustness and adversarial resistance

---

## 📚 Key Resources Referenced

### **Papers**
- Attention Is All You Need (Vaswani et al., 2017)
- BERT (Devlin et al., 2018)
- GPT-2/3 (Radford et al., 2019; Brown et al., 2020)
- Training language models to follow instructions (Ouyang et al., 2022)
- Direct Preference Optimization (Rafailov et al., 2023)

### **Tools & Libraries**
- PyTorch
- HuggingFace Transformers & Tokenizers
- TensorBoard / Weights & Biases
- tiktoken
- OpenAI API

### **Datasets**
- The Pile
- OpenWebText
- C4 (Common Crawl)
- Stanford Alpaca
- MMLU, HellaSwag, TruthfulQA

 ---
 ---
 
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
