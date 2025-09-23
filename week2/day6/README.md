# Week 2 · Day 6 — Scaled Dot-Product Attention: Mathematical Foundation and Theory

Complexity: 3 (Medium)  •  Estimated time: 2–3 hours

Today we dive into the heart of transformers: the attention mechanism. You'll understand the mathematical foundation, theoretical principles, and intuitive concepts behind scaled dot-product attention that enables transformers to capture contextual relationships.

## Learning Objectives

- Understand the theoretical foundation of attention mechanisms
- Derive and analyze the scaled dot-product attention formula
- Grasp the intuitive meaning of Query, Key, and Value matrices
- Comprehend why scaling by √d_k is mathematically necessary
- Connect attention theory to transformer architecture principles
- Analyze different types of attention patterns and their interpretations

---

## Theoretical Foundation of Attention

### The Core Concept

Attention mechanisms solve a fundamental problem in sequence processing: **how to selectively focus on relevant parts of the input when processing each position**. Unlike traditional RNNs that process sequences sequentially, attention allows every position to directly access information from every other position.

```mermaid
flowchart TD
    A[Input Sequence] --> B[Query Q: What to look for?]
    A --> C[Key K: What information is available?]
    A --> D[Value V: What is the actual content?]
    
    B --> E[Similarity Computation: Q·K^T]
    C --> E
    E --> F[Scale by √d_k]
    F --> G[Normalize: Softmax]
    G --> H[Attention Weights α]
    
    H --> I[Weighted Aggregation: Σ(α·V)]
    D --> I
    I --> J[Contextualized Output]
    
    K[Core Formula] --> L["Attention(Q,K,V) = softmax(QK^T/√d_k)V"]
```

### Historical Context and Motivation

Before attention mechanisms, sequence-to-sequence models relied on fixed-size context vectors, creating an information bottleneck. The attention mechanism, introduced by Bahdanau et al. (2014) and refined by Vaswani et al. (2017), allows models to:

1. **Access all positions simultaneously** (parallelization)
2. **Learn dynamic focus patterns** (context-dependent attention)
3. **Capture long-range dependencies** (no sequential processing constraint)
4. **Provide interpretability** (attention weights show what the model focuses on)

---

## Mathematical Derivation and Analysis

### The Scaled Dot-Product Formula

**Attention(Q, K, V) = softmax(QK^T / √d_k) V**

Let's break down each component mathematically:

#### 1. Query-Key Similarity (Q·K^T)

The dot product between queries and keys computes similarity scores:
- **Q ∈ ℝ^(n×d_k)**: n queries, each of dimension d_k
- **K ∈ ℝ^(n×d_k)**: n keys, each of dimension d_k  
- **QK^T ∈ ℝ^(n×n)**: similarity matrix where entry (i,j) = q_i · k_j

**Mathematical Intuition**: The dot product measures how "aligned" a query is with each key. Higher values indicate stronger relevance.

#### 2. Scaling Factor (√d_k)

The scaling prevents the dot products from becoming too large, which would cause the softmax to saturate.

**Mathematical Analysis**:
- If Q and K have zero mean and unit variance
- Each element of QK^T has variance d_k
- Scaling by √d_k normalizes the variance to 1
- This keeps the softmax in its sensitive region (avoiding saturation)

**Proof Sketch**: If q_i, k_j ~ N(0,1), then q_i · k_j = Σ(q_i[m] * k_j[m]) has variance d_k. Dividing by √d_k gives unit variance.

#### 3. Softmax Normalization

The softmax function converts similarity scores to a probability distribution:

**softmax(x_i) = exp(x_i) / Σ_j exp(x_j)**

**Properties**:
- Output values sum to 1 (probability distribution)
- Differentiable (enables gradient-based learning)
- Emphasizes larger values (attention sharpening)
- Temperature parameter implicit in scaling

#### 4. Value Aggregation

The final step computes a weighted average of value vectors:
**output_i = Σ_j α_ij * v_j**

Where α_ij are the attention weights from the softmax.

---

## Query, Key, Value: Intuitive Understanding

### The Database Analogy

The Q-K-V mechanism mirrors database operations:

- **Query (Q)**: "What information am I looking for?"
  - Represents the information needs of each position
  - Determines what patterns to search for in the sequence

- **Key (K)**: "What information do I have available?"
  - Represents the searchable attributes of each position
  - Acts as an index for content retrieval

- **Value (V)**: "What is the actual content to retrieve?"
  - Contains the information to be aggregated
  - The payload that gets mixed based on attention weights

### Linguistic Interpretation

In natural language processing:

- **Query**: "As the word 'jumps', what grammatical relationships should I attend to?"
- **Key**: Each word advertises its grammatical role and semantic properties
- **Value**: Each word provides its semantic content and contextual information

### Self-Attention vs Cross-Attention

- **Self-Attention**: Q, K, V all derived from the same sequence
  - Captures intra-sequence relationships
  - Used in encoder-only and decoder-only architectures

- **Cross-Attention**: Q from one sequence, K,V from another
  - Captures inter-sequence relationships  
  - Used in encoder-decoder architectures (e.g., translation)

---

## Theoretical Properties and Analysis

### Computational Complexity

- **Time Complexity**: O(n²d) where n is sequence length, d is model dimension
- **Space Complexity**: O(n²) for storing attention matrix
- **Bottleneck**: Quadratic scaling with sequence length

### Attention Patterns and Their Meanings

#### 1. Uniform Attention
- **Pattern**: Equal weights across all positions
- **Interpretation**: No specific focus, global averaging
- **When it occurs**: Similar or uninformative tokens

#### 2. Peaked Attention  
- **Pattern**: High weight on one position, low elsewhere
- **Interpretation**: Strong focus on specific information
- **When it occurs**: Clear relevance signals

#### 3. Local Attention
- **Pattern**: Higher weights on nearby positions
- **Interpretation**: Local context dependency
- **When it occurs**: Syntactic relationships, local coherence

#### 4. Structured Attention
- **Pattern**: Systematic patterns (diagonal, block structure)
- **Interpretation**: Learned linguistic or domain-specific structures
- **When it occurs**: Grammar, hierarchical relationships

### Information Flow and Representation

Attention enables **information mixing** at each layer:
- Each position's representation becomes a weighted combination of all positions
- Multiple layers allow for complex, hierarchical information integration
- The model learns which information to mix at each layer

---

## Scaling and Numerical Stability

### Why √d_k Scaling is Critical

Without scaling, as d_k increases:
1. **Dot products grow larger** (variance increases)
2. **Softmax saturates** (gradients vanish)
3. **Attention becomes too sharp** (loss of diversity)
4. **Training becomes unstable** (gradient problems)

### Mathematical Analysis of Scaling

For random vectors q, k with components ~ N(0,1):
- **E[q·k] = 0** (expected dot product is zero)
- **Var(q·k) = d_k** (variance grows with dimension)
- **After scaling**: Var(q·k/√d_k) = 1 (normalized variance)

This normalization keeps the softmax input in a reasonable range regardless of dimension.

### Alternative Scaling Approaches

1. **Temperature scaling**: softmax(QK^T / τ) where τ is learnable
2. **Layer-wise scaling**: Different scaling factors per layer
3. **Adaptive scaling**: Scaling based on input statistics

---

## Connection to Information Theory

### Attention as Information Routing

Attention can be viewed through an information-theoretic lens:
- **Attention weights**: Probability distribution over information sources
- **Entropy**: Measures how focused or distributed attention is
- **Mutual information**: Quantifies information shared between positions

### Entropy Analysis

**Attention Entropy**: H(α) = -Σ_j α_j log α_j

- **High entropy**: Distributed attention (uniform focus)
- **Low entropy**: Concentrated attention (selective focus)
- **Zero entropy**: Complete focus on single position

This provides a quantitative measure of attention behavior.

---

## Theoretical Limitations and Extensions

### Current Limitations

1. **Quadratic complexity**: Prohibitive for very long sequences
2. **Fixed context**: Cannot attend beyond training sequence length
3. **Position independence**: Basic attention doesn't inherently understand position
4. **Attention collapse**: Can focus too narrowly, losing diversity

### Theoretical Extensions

1. **Sparse attention**: Attend to subset of positions (O(n√n) or O(n log n))
2. **Linear attention**: Approximate attention with linear complexity
3. **Relative position encoding**: Position-aware attention mechanisms
4. **Hierarchical attention**: Multi-scale attention patterns

---

## Practical Implementation Considerations

All code examples, visualizations, and hands-on exercises are provided in the accompanying Jupyter notebook: `day6_attention_implementation.ipynb`

The notebook includes:
- Step-by-step implementation from scratch
- Manual computation examples
- Attention pattern visualizations
- Scaling factor demonstrations
- Interactive exercises and analysis

---

## Key Theoretical Takeaways

1. **Attention enables parallel processing** of sequences while capturing dependencies
2. **Q-K-V decomposition** provides flexible information routing mechanism  
3. **Scaling by √d_k** is mathematically necessary for numerical stability
4. **Attention patterns** reveal learned linguistic and structural relationships
5. **Information mixing** occurs through weighted aggregation of value vectors
6. **Computational complexity** is the main limitation for long sequences

---

## What's Next (Day 7 Preview)

Tomorrow we'll explore multi-head attention, understanding how multiple attention heads capture different types of relationships simultaneously and how they're combined through concatenation and projection.

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Original attention paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanations
- [Attention Mechanisms in Neural Networks](https://distill.pub/2016/augmented-rnns/#attentional-interfaces) - Distill.pub article

### Why Each Component Matters

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explain_attention_components():
    """Explain each component of attention with intuitive examples."""
    
    print("Understanding Q, K, V with Intuitive Examples")
    print("=" * 50)
    
    # Example: "The cat sat on the mat"
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    
    print("Example sentence: 'The cat sat on the mat'")
    print("\nIntuitive Understanding:")
    print("- Query (Q): 'What should I pay attention to?'")
    print("- Key (K): 'What information do I have?'")
    print("- Value (V): 'What is the actual content?'")
    
    print("\nFor token 'sat':")
    print("- Query: Looking for subject and object relationships")
    print("- Keys: All tokens offer their relationship information")
    print("- Values: Actual semantic content of each token")
    
    print("\nAttention weights tell us:")
    print("- How much 'sat' should focus on 'cat' (subject)")
    print("- How much 'sat' should focus on 'mat' (object)")
    print("- Less attention to articles 'the', 'the'")

explain_attention_components()
```

---

## Step-by-Step Implementation

### Basic Attention Implementation

```python
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention implementation from scratch."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]  
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.size()
        
        # Step 1: Compute attention scores (Q·K^T)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale by √d_k
        scores = scores / np.sqrt(d_model)
        
        # Step 3: Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

# Test the implementation
def test_basic_attention():
    """Test basic attention with simple example."""
    
    batch_size, seq_len, d_model = 1, 4, 8
    
    # Create simple input embeddings
    embeddings = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize attention
    attention = ScaledDotProductAttention(d_model)
    
    # Self-attention: Q, K, V are all the same
    output, weights = attention(embeddings, embeddings, embeddings)
    
    print("Basic Attention Test:")
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {weights.sum(dim=-1)}")
    
    return output, weights

output, weights = test_basic_attention()
```

### Manual Computation Example

```python
def manual_attention_computation():
    """Compute attention manually for a small example."""
    
    print("Manual Attention Computation")
    print("=" * 40)
    
    # Small example: 3 tokens, 4 dimensions
    seq_len, d_model = 3, 4
    
    # Create simple Q, K, V matrices
    Q = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],  # Token 1 query
        [0.0, 1.0, 0.0, 1.0],  # Token 2 query  
        [1.0, 1.0, 0.0, 0.0]   # Token 3 query
    ], dtype=torch.float32)
    
    K = torch.tensor([
        [1.0, 0.0, 0.0, 1.0],  # Token 1 key
        [0.0, 1.0, 1.0, 0.0],  # Token 2 key
        [1.0, 0.0, 1.0, 1.0]   # Token 3 key
    ], dtype=torch.float32)
    
    V = torch.tensor([
        [2.0, 0.0, 1.0, 0.0],  # Token 1 value
        [0.0, 2.0, 0.0, 1.0],  # Token 2 value
        [1.0, 1.0, 2.0, 2.0]   # Token 3 value
    ], dtype=torch.float32)
    
    print("Query matrix Q:")
    print(Q.numpy())
    print("\nKey matrix K:")
    print(K.numpy())
    print("\nValue matrix V:")
    print(V.numpy())
    
    # Step 1: Compute Q·K^T
    scores = torch.matmul(Q, K.transpose(0, 1))
    print(f"\nStep 1 - Attention scores (Q·K^T):")
    print(scores.numpy())
    
    # Step 2: Scale by √d_k
    scaled_scores = scores / np.sqrt(d_model)
    print(f"\nStep 2 - Scaled scores (÷√{d_model} = ÷{np.sqrt(d_model):.2f}):")
    print(scaled_scores.numpy())
    
    # Step 3: Apply softmax
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"\nStep 3 - Attention weights (softmax):")
    print(attention_weights.numpy())
    
    # Verify weights sum to 1
    print(f"\nWeights sum per row: {attention_weights.sum(dim=-1).numpy()}")
    
    # Step 4: Apply to values
    output = torch.matmul(attention_weights, V)
    print(f"\nStep 4 - Final output (weights × V):")
    print(output.numpy())
    
    return Q, K, V, attention_weights, output

Q, K, V, manual_weights, manual_output = manual_attention_computation()
```

---

## Visualizing Attention Patterns

```python
def visualize_attention_patterns():
    """Create comprehensive attention visualizations."""
    
    # Create a more interesting example
    seq_len, d_model = 6, 16
    
    # Simulate embeddings for: "The cat sat on the mat"
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    
    # Create embeddings with some structure
    torch.manual_seed(42)
    embeddings = torch.randn(1, seq_len, d_model)
    
    # Make some tokens more similar (e.g., "The" and "the")
    embeddings[0, 4] = embeddings[0, 0] + 0.1 * torch.randn(d_model)
    
    # Make "cat" and "mat" somewhat similar (both nouns)
    embeddings[0, 5] = embeddings[0, 1] + 0.3 * torch.randn(d_model)
    
    # Apply attention
    attention = ScaledDotProductAttention(d_model)
    output, weights = attention(embeddings, embeddings, embeddings)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Attention heatmap
    sns.heatmap(weights[0].detach().numpy(), 
                xticklabels=tokens, yticklabels=tokens,
                annot=True, fmt='.3f', cmap='Blues',
                ax=axes[0, 0])
    axes[0, 0].set_title('Attention Weights Heatmap')
    axes[0, 0].set_xlabel('Key (attending to)')
    axes[0, 0].set_ylabel('Query (attending from)')
    
    # 2. Attention weights for specific token
    token_idx = 2  # "sat"
    axes[0, 1].bar(tokens, weights[0, token_idx].detach().numpy())
    axes[0, 1].set_title(f'Attention weights for "{tokens[token_idx]}"')
    axes[0, 1].set_ylabel('Attention Weight')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Raw attention scores (before softmax)
    raw_scores = torch.matmul(embeddings, embeddings.transpose(-2, -1)) / np.sqrt(d_model)
    sns.heatmap(raw_scores[0].detach().numpy(),
                xticklabels=tokens, yticklabels=tokens,
                annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=axes[1, 0])
    axes[1, 0].set_title('Raw Attention Scores (before softmax)')
    
    # 4. Attention entropy (how focused/distributed)
    entropy = -torch.sum(weights * torch.log(weights + 1e-9), dim=-1)
    axes[1, 1].bar(tokens, entropy[0].detach().numpy())
    axes[1, 1].set_title('Attention Entropy (higher = more distributed)')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print interpretation
    print("Attention Pattern Interpretation:")
    print("=" * 40)
    
    for i, token in enumerate(tokens):
        top_attention = torch.topk(weights[0, i], 2)
        top_tokens = [tokens[idx] for idx in top_attention.indices]
        top_weights = top_attention.values
        
        print(f"'{token}' attends most to:")
        for j, (att_token, weight) in enumerate(zip(top_tokens, top_weights)):
            print(f"  {j+1}. '{att_token}' (weight: {weight:.3f})")
        print()
    
    return weights, tokens

attention_weights, tokens = visualize_attention_patterns()
```

---

## Understanding the Scaling Factor

```python
def demonstrate_scaling_importance():
    """Show why we need the √d_k scaling factor."""
    
    print("Why Scale by √d_k?")
    print("=" * 30)
    
    # Test with different dimensions
    dimensions = [4, 16, 64, 256]
    seq_len = 4
    
    results = {}
    
    for d_model in dimensions:
        # Create random Q and K
        Q = torch.randn(1, seq_len, d_model)
        K = torch.randn(1, seq_len, d_model)
        
        # Compute scores without scaling
        scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
        
        # Compute scores with scaling
        scores_scaled = scores_unscaled / np.sqrt(d_model)
        
        # Apply softmax
        weights_unscaled = F.softmax(scores_unscaled, dim=-1)
        weights_scaled = F.softmax(scores_scaled, dim=-1)
        
        # Measure how "sharp" the attention is (entropy)
        entropy_unscaled = -torch.sum(weights_unscaled * torch.log(weights_unscaled + 1e-9), dim=-1).mean()
        entropy_scaled = -torch.sum(weights_scaled * torch.log(weights_scaled + 1e-9), dim=-1).mean()
        
        results[d_model] = {
            'scores_std_unscaled': scores_unscaled.std().item(),
            'scores_std_scaled': scores_scaled.std().item(),
            'entropy_unscaled': entropy_unscaled.item(),
            'entropy_scaled': entropy_scaled.item(),
            'max_weight_unscaled': weights_unscaled.max().item(),
            'max_weight_scaled': weights_scaled.max().item()
        }
    
    # Display results
    print("Dimension | Scores Std (Unscaled) | Scores Std (Scaled) | Max Weight (Unscaled) | Max Weight (Scaled)")
    print("-" * 100)
    
    for d_model, stats in results.items():
        print(f"{d_model:8d} | {stats['scores_std_unscaled']:17.3f} | {stats['scores_std_scaled']:16.3f} | "
              f"{stats['max_weight_unscaled']:18.3f} | {stats['max_weight_scaled']:17.3f}")
    
    print("\nObservations:")
    print("- Without scaling: larger dimensions → larger scores → sharper attention")
    print("- With scaling: attention sharpness remains consistent across dimensions")
    print("- Scaling prevents attention from becoming too concentrated")
    
    return results

scaling_results = demonstrate_scaling_importance()
```

---

## Attention Patterns Analysis

```python
def analyze_attention_patterns():
    """Analyze different types of attention patterns."""
    
    print("Types of Attention Patterns")
    print("=" * 35)
    
    seq_len, d_model = 5, 8
    
    # Create different scenarios
    scenarios = {
        'uniform': torch.ones(1, seq_len, d_model),  # All tokens identical
        'sequential': torch.arange(seq_len * d_model).float().view(1, seq_len, d_model),  # Sequential pattern
        'similar_pairs': torch.randn(1, seq_len, d_model)  # Will modify for similarity
    }
    
    # Make pairs similar in the third scenario
    scenarios['similar_pairs'][0, 1] = scenarios['similar_pairs'][0, 0] + 0.1 * torch.randn(d_model)
    scenarios['similar_pairs'][0, 3] = scenarios['similar_pairs'][0, 2] + 0.1 * torch.randn(d_model)
    
    attention = ScaledDotProductAttention(d_model)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, embeddings) in enumerate(scenarios.items()):
        output, weights = attention(embeddings, embeddings, embeddings)
        
        # Visualize attention pattern
        sns.heatmap(weights[0].detach().numpy(),
                   annot=True, fmt='.2f', cmap='Blues',
                   ax=axes[idx])
        axes[idx].set_title(f'{name.title()} Embeddings')
        axes[idx].set_xlabel('Key Position')
        axes[idx].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze patterns
    for name, embeddings in scenarios.items():
        output, weights = attention(embeddings, embeddings, embeddings)
        
        # Compute attention statistics
        self_attention = torch.diag(weights[0]).mean()  # How much tokens attend to themselves
        max_attention = weights[0].max()  # Maximum attention weight
        entropy = -torch.sum(weights[0] * torch.log(weights[0] + 1e-9), dim=-1).mean()
        
        print(f"\n{name.title()} Pattern:")
        print(f"  Average self-attention: {self_attention:.3f}")
        print(f"  Maximum attention weight: {max_attention:.3f}")
        print(f"  Average entropy: {entropy:.3f}")

analyze_attention_patterns()
```

---

## Connecting to docs/llm.md

```python
def connect_to_llm_docs():
    """Connect implementation to concepts in docs/llm.md."""
    
    print("Connection to docs/llm.md Concepts")
    print("=" * 40)
    
    print("1. Self-Attention (from docs/llm.md):")
    print("   'Each token attends to all other tokens in the sequence'")
    print("   → Implemented as Q=K=V=input_embeddings")
    
    print("\n2. Scaled Dot-Product Attention (from docs/llm.md):")
    print("   'Attention(Q, K, V) = softmax(QK^T / √d_k) V'")
    print("   → Exact formula we implemented!")
    
    print("\n3. Contextual Relationships (from docs/llm.md):")
    print("   'Learning contextual relationships between tokens'")
    print("   → Attention weights show these relationships")
    
    # Demonstrate with example from docs
    seq_len, d_model = 4, 8
    
    # Simulate tokens with different relationships
    embeddings = torch.randn(1, seq_len, d_model)
    
    # Make token 0 and 2 similar (like subject-verb relationship)
    embeddings[0, 2] = embeddings[0, 0] + 0.2 * torch.randn(d_model)
    
    attention = ScaledDotProductAttention(d_model)
    output, weights = attention(embeddings, embeddings, embeddings)
    
    print(f"\nExample Attention Matrix:")
    print("Rows = Query tokens, Columns = Key tokens")
    print(weights[0].detach().numpy().round(3))
    
    print(f"\nToken 0 → Token 2 attention: {weights[0, 0, 2]:.3f}")
    print(f"Token 2 → Token 0 attention: {weights[0, 2, 0]:.3f}")
    print("Higher values indicate stronger contextual relationships!")

connect_to_llm_docs()
```

---

## Practical Exercises

### Exercise 1: Hand Computation

```python
def exercise_hand_computation():
    """Exercise: Compute attention by hand for very small example."""
    
    print("Exercise 1: Hand Computation")
    print("=" * 35)
    
    print("Given:")
    print("Q = [[1, 0], [0, 1]]")
    print("K = [[1, 1], [1, 0]]") 
    print("V = [[2, 1], [1, 2]]")
    print("d_k = 2")
    
    print("\nYour task:")
    print("1. Compute QK^T")
    print("2. Scale by √d_k")
    print("3. Apply softmax")
    print("4. Multiply by V")
    
    # Solution
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    K = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    V = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    
    print("\nSolution:")
    
    # Step 1
    QK = torch.matmul(Q, K.T)
    print(f"1. QK^T = \n{QK.numpy()}")
    
    # Step 2
    scaled = QK / np.sqrt(2)
    print(f"2. Scaled = \n{scaled.numpy()}")
    
    # Step 3
    weights = F.softmax(scaled, dim=-1)
    print(f"3. Softmax = \n{weights.numpy()}")
    
    # Step 4
    output = torch.matmul(weights, V)
    print(f"4. Output = \n{output.numpy()}")

exercise_hand_computation()
```

### Exercise 2: Attention Interpretation

```python
def exercise_attention_interpretation():
    """Exercise: Interpret attention patterns in different contexts."""
    
    print("Exercise 2: Attention Interpretation")
    print("=" * 40)
    
    # Create sentence: "The quick brown fox jumps"
    tokens = ["The", "quick", "brown", "fox", "jumps"]
    seq_len, d_model = len(tokens), 12
    
    # Create embeddings with linguistic structure
    torch.manual_seed(123)
    embeddings = torch.randn(1, seq_len, d_model)
    
    # Make adjectives similar to each other
    embeddings[0, 2] = embeddings[0, 1] + 0.3 * torch.randn(d_model)  # brown ~ quick
    
    # Make noun and verb have some relationship
    embeddings[0, 4] = embeddings[0, 3] + 0.4 * torch.randn(d_model)  # jumps ~ fox
    
    attention = ScaledDotProductAttention(d_model)
    output, weights = attention(embeddings, embeddings, embeddings)
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights[0].detach().numpy(),
                xticklabels=tokens, yticklabels=tokens,
                annot=True, fmt='.3f', cmap='Blues')
    plt.title('Attention Pattern: "The quick brown fox jumps"')
    plt.xlabel('Attending to (Key)')
    plt.ylabel('Attending from (Query)')
    plt.show()
    
    print("Questions to consider:")
    print("1. Which words attend most to 'fox'?")
    print("2. Do adjectives ('quick', 'brown') show similar attention patterns?")
    print("3. How much does 'jumps' attend to 'fox' (subject-verb relationship)?")
    
    # Print top attention pairs
    print("\nTop attention relationships:")
    flat_weights = weights[0].flatten()
    top_indices = torch.topk(flat_weights, 5).indices
    
    for idx in top_indices:
        i, j = idx // seq_len, idx % seq_len
        if i != j:  # Skip self-attention
            print(f"'{tokens[i]}' → '{tokens[j]}': {weights[0, i, j]:.3f}")

exercise_attention_interpretation()
```

---

## Key Takeaways

1. **Mathematical Foundation**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
2. **Scaling Importance**: √d_k prevents attention from becoming too sharp
3. **Attention Weights**: Show relationships between tokens
4. **Self-Attention**: Q, K, V all come from the same input sequence
5. **Visualization**: Heatmaps reveal linguistic patterns and relationships

---

## What's Next (Day 7 Preview)

Tomorrow we'll extend single-head attention to multi-head attention, understanding why multiple attention heads capture different types of relationships and how they're combined through concatenation and projection.

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanations
- [Attention Mechanisms](https://distill.pub/2016/augmented-rnns/#attentional-interfaces) - Distill.pub article
- [Understanding Attention](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634)
