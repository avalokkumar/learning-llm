# Week 2 · Day 10 — Building a Complete Minimal Attention Block in PyTorch

Complexity: 4 (Medium-High)  •  Estimated time: 3–4 hours

### 🌟 Layman's Understanding
Today is like the final assembly day in a car factory. We've learned about all the individual parts of a transformer - the attention mechanism (the steering system), layer normalization (the suspension), residual connections (the chassis), and GELU activation (the engine). Now we're going to put all these pieces together to build a complete, working transformer that can actually learn and generate text. It's like taking all the components we've studied and creating a mini version of what powers ChatGPT and other language models.

### 📚 Basic Understanding
Today we integrate everything learned this week into a complete, minimal transformer attention block. You'll build a production-ready implementation with proper initialization, training loops, and evaluation metrics. This represents the culmination of Week 2, where we take all the theoretical knowledge and create a working system that can be trained on real data.

### 🔬 Intermediate Understanding
We'll implement a complete transformer architecture that integrates multi-head attention, position-wise feed-forward networks, layer normalization, and residual connections into a cohesive system. The implementation will include proper weight initialization strategies, training dynamics analysis, and performance optimization techniques. This bridges the gap between understanding individual components and building production-ready transformer models.

### 🎓 Advanced Understanding
This implementation serves as a reference architecture that demonstrates best practices in transformer design, including pre-norm vs post-norm trade-offs, initialization schemes that maintain gradient flow, and training loop optimizations. The code provides insights into the engineering considerations that make transformers scalable and stable, serving as a foundation for understanding larger models like GPT and BERT.

## Learning Objectives
- Build a complete transformer block from scratch
- Implement proper weight initialization
- Create training and evaluation loops
- Test on a simple language modeling task
- Understand performance optimization techniques
- Connect to real-world transformer implementations

---

## Complete Minimal Transformer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import math

class MinimalAttention(nn.Module):
    """Minimal scaled dot-product attention implementation."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x)  # (batch, seq_len, d_model)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

class LayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class TransformerBlock(nn.Module):
    """Complete transformer block."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MinimalAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-norm architecture
        # Self-attention
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.attention(normed_x, mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x, attn_weights

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MinimalTransformer(nn.Module):
    """Complete minimal transformer for language modeling."""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0
    
    def forward(self, x, targets=None):
        seq_len = x.size(1)
        
        # Embeddings and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Apply transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)
        
        # Final layer norm and language modeling head
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shift targets for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'attention_weights': attention_weights
        }

def test_minimal_transformer():
    """Test the complete minimal transformer."""
    
    # Model configuration
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 4
    d_ff = 512
    seq_len = 32
    batch_size = 2
    
    # Create model
    model = MinimalTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids, targets)
    
    print("Minimal Transformer Test:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Number of attention layers: {len(outputs['attention_weights'])}")
    print(f"Attention weights shape: {outputs['attention_weights'][0].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, outputs

model, test_outputs = test_minimal_transformer()
```

---

## Simple Training Loop

```python
class SimpleTextDataset(Dataset):
    """Simple dataset for language modeling."""
    
    def __init__(self, text, vocab_size, seq_len):
        self.seq_len = seq_len
        
        # Simple tokenization (character-level)
        chars = sorted(list(set(text)))
        self.vocab_size = min(len(chars), vocab_size)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars[:self.vocab_size])}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        # Tokenize text
        self.tokens = [self.char_to_idx.get(ch, 0) for ch in text]
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        targets = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, targets

def train_minimal_transformer():
    """Train the minimal transformer on a simple task."""
    
    # Sample text data
    text = """
    The transformer architecture has revolutionized natural language processing.
    It uses self-attention mechanisms to process sequences in parallel.
    The key innovation is the attention mechanism that allows the model to focus
    on different parts of the input sequence when making predictions.
    """ * 10  # Repeat for more data
    
    # Configuration
    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256
    seq_len = 32
    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 10
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(text, vocab_size, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = MinimalTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    losses = []
    
    print("Training Minimal Transformer...")
    print("=" * 40)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(input_ids, targets)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, dataset, losses

trained_model, dataset, training_losses = train_minimal_transformer()
```

---

## Text Generation and Evaluation

```python
def generate_text(model, dataset, prompt="The", max_length=100, temperature=1.0):
    """Generate text using the trained model."""
    
    model.eval()
    
    # Convert prompt to tokens
    tokens = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_ids = torch.tensor([tokens], dtype=torch.long)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Get next token probabilities
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            
            # Stop if we hit unknown token or repeat too much
            if next_token == 0:
                break
    
    # Convert back to text
    generated_text = ''.join([dataset.idx_to_char.get(token, '') for token in tokens])
    return generated_text

def evaluate_model(model, dataset):
    """Evaluate model performance."""
    
    print("Text Generation Examples:")
    print("=" * 30)
    
    prompts = ["The", "Trans", "Atten"]
    
    for prompt in prompts:
        generated = generate_text(model, dataset, prompt, max_length=50)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: {generated}")
        print("-" * 30)

# Generate some text
evaluate_model(trained_model, dataset)
```

---

## Attention Visualization

```python
def visualize_attention_patterns(model, dataset, text="The transformer"):
    """Visualize attention patterns for a given input."""
    
    model.eval()
    
    # Tokenize input
    tokens = [dataset.char_to_idx.get(ch, 0) for ch in text]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(input_ids)
        attention_weights = outputs['attention_weights']
    
    # Visualize attention for each layer and head
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].size(1)
    
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4 * num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer in range(num_layers):
        # Average attention across heads
        avg_attention = attention_weights[layer][0].mean(0).cpu().numpy()
        
        # Individual head (head 0)
        head_attention = attention_weights[layer][0, 0].cpu().numpy()
        
        # Plot average attention
        sns.heatmap(avg_attention, 
                   xticklabels=list(text), 
                   yticklabels=list(text),
                   ax=axes[layer, 0], 
                   cmap='Blues',
                   cbar=True)
        axes[layer, 0].set_title(f'Layer {layer+1}: Average Attention')
        
        # Plot head 0 attention
        sns.heatmap(head_attention,
                   xticklabels=list(text),
                   yticklabels=list(text),
                   ax=axes[layer, 1],
                   cmap='Reds',
                   cbar=True)
        axes[layer, 1].set_title(f'Layer {layer+1}: Head 0 Attention')
    
    plt.tight_layout()
    plt.show()
    
    return attention_weights

# Visualize attention patterns
attention_patterns = visualize_attention_patterns(trained_model, dataset, "The transformer")
```

---

## Performance Analysis

```python
def analyze_performance(model):
    """Analyze model performance and efficiency."""
    
    print("Performance Analysis")
    print("=" * 25)
    
    # Model size analysis
    total_params = sum(p.numel() for p in model.parameters())
    
    component_params = {}
    component_params['Embedding'] = model.embedding.weight.numel()
    component_params['Positional'] = model.pos_encoding.pe.numel()
    
    transformer_params = 0
    for block in model.transformer_blocks:
        transformer_params += sum(p.numel() for p in block.parameters())
    component_params['Transformer Blocks'] = transformer_params
    
    component_params['Output Head'] = model.lm_head.weight.numel()
    
    print("Parameter Distribution:")
    for component, params in component_params.items():
        percentage = (params / total_params) * 100
        print(f"{component:20s}: {params:8,} ({percentage:5.1f}%)")
    
    print(f"{'Total':20s}: {total_params:8,} (100.0%)")
    
    # Memory analysis
    model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"\nModel Size: {model_size_mb:.1f} MB")
    
    # Inference speed test
    model.eval()
    seq_len = 128
    batch_size = 1
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids)
    
    # Timing
    start_time = time.time()
    num_runs = 100
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Tokens per second: {seq_len/avg_time:.0f}")

analyze_performance(trained_model)
```

---

## Key Implementation Insights

```python
def implementation_insights():
    """Key insights from building the minimal transformer."""
    
    insights = {
        "Architecture Choices": [
            "Pre-norm vs post-norm: Pre-norm generally more stable",
            "GELU vs ReLU: GELU provides smoother gradients",
            "Causal masking: Essential for autoregressive generation",
            "Residual connections: Critical for deep network training"
        ],
        
        "Initialization": [
            "Xavier/Glorot for linear layers",
            "Small normal distribution for embeddings",
            "Zero initialization for biases",
            "Proper scaling prevents gradient explosion"
        ],
        
        "Training Considerations": [
            "Learning rate scheduling often helpful",
            "Gradient clipping prevents instability",
            "Dropout prevents overfitting",
            "Batch size affects training dynamics"
        ],
        
        "Performance Optimization": [
            "Attention is O(n²) in sequence length",
            "Multi-head attention parallelizes well",
            "Memory usage dominated by attention matrices",
            "Inference can be optimized with caching"
        ]
    }
    
    print("Implementation Insights")
    print("=" * 25)
    
    for category, points in insights.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  • {point}")

implementation_insights()
```

---

## Connection to Real-World Models

```python
def compare_to_real_models():
    """Compare our minimal implementation to real-world models."""
    
    models = {
        'Our Minimal': {'params': '50K', 'layers': 2, 'd_model': 64, 'heads': 4},
        'GPT-2 Small': {'params': '117M', 'layers': 12, 'd_model': 768, 'heads': 12},
        'BERT Base': {'params': '110M', 'layers': 12, 'd_model': 768, 'heads': 12},
        'GPT-3': {'params': '175B', 'layers': 96, 'd_model': 12288, 'heads': 96},
        'T5 Base': {'params': '220M', 'layers': 12, 'd_model': 768, 'heads': 12}
    }
    
    print("Model Comparison")
    print("=" * 20)
    print("Model        | Parameters | Layers | d_model | Heads")
    print("-" * 55)
    
    for model, specs in models.items():
        print(f"{model:12s} | {specs['params']:>10s} | {specs['layers']:>6d} | "
              f"{specs['d_model']:>7d} | {specs['heads']:>5d}")
    
    print("\nScaling Insights:")
    print("• Our implementation captures core transformer mechanics")
    print("• Real models use much larger dimensions and more layers")
    print("• Performance scales with model size and training data")
    print("• Architecture principles remain consistent across scales")

compare_to_real_models()
```

---

## Week 2 Summary and Next Steps

### 🌟 Layman's Understanding
Congratulations! You've just built your own mini version of the technology that powers ChatGPT, Google Translate, and many other AI systems. Think of it like learning to build a car - you started with understanding the engine (attention), then the transmission (feed-forward networks), the suspension (layer normalization), and the chassis (residual connections). Now you have a complete, working vehicle that can actually drive (process and generate text)!

### 📚 Basic Understanding
Congratulations! You've completed Week 2 and built a complete transformer from scratch. You now have hands-on experience with every major component of transformer architecture and understand how they work together to create powerful language models.

### 🔬 Intermediate Understanding
You've successfully implemented a production-ready transformer architecture with proper initialization, training dynamics, and performance optimization. This foundation provides the technical depth needed to understand and work with larger models like GPT and BERT, and gives you the skills to modify and extend transformer architectures for specific applications.

### 🎓 Advanced Understanding
Your implementation demonstrates mastery of transformer engineering principles, including gradient flow optimization, architectural design choices, and training stability considerations. This knowledge positions you to contribute to cutting-edge research in transformer architectures, optimization techniques, and scaling laws for large language models.

### Week 2 Achievements
- ✅ **Day 6**: Scaled dot-product attention mechanism
- ✅ **Day 7**: Multi-head attention with parallel processing
- ✅ **Day 8**: Three transformer architectures (encoder-only, decoder-only, encoder-decoder)
- ✅ **Day 9**: Supporting components (residuals, LayerNorm, GELU)
- ✅ **Day 10**: Complete minimal transformer implementation

### Key Skills Developed
1. **Attention Mechanisms**: Deep understanding of self-attention and cross-attention
2. **Architecture Design**: When to use different transformer variants
3. **Implementation Skills**: Building transformers from scratch in PyTorch
4. **Training Loops**: End-to-end model training and evaluation
5. **Performance Analysis**: Understanding computational and memory requirements

### What's Next (Week 3 Preview)
Week 3 will focus on **Training Dynamics and Optimization**:
- Advanced optimization techniques (Adam, learning rate scheduling)
- Training stability and gradient flow analysis
- Loss functions and training objectives
- Regularization techniques and preventing overfitting
- Distributed training and scaling considerations

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers) - Implementation details
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) - Official tutorial
