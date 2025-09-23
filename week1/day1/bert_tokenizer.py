#!/usr/bin/env python3
"""
BERT Tokenizer Implementation
A complete end-to-end BERT tokenizer with WordPiece algorithm
"""

import re
import unicodedata
from typing import List, Dict, Set, Optional, Tuple
import json
import os


class BertTokenizer:
    """
    A complete BERT tokenizer implementation with WordPiece algorithm.
    Includes normalization, basic tokenization, and WordPiece tokenization.
    """
    
    def __init__(self, vocab_file: Optional[str] = None, do_lower_case: bool = True, 
                 max_len: int = 512, unk_token: str = "[UNK]", sep_token: str = "[SEP]",
                 pad_token: str = "[PAD]", cls_token: str = "[CLS]", mask_token: str = "[MASK]"):
        """
        Initialize BERT tokenizer.
        
        Args:
            vocab_file: Path to vocabulary file (if None, uses a basic vocab)
            do_lower_case: Whether to lowercase input text
            max_len: Maximum sequence length
            unk_token: Unknown token
            sep_token: Separator token
            pad_token: Padding token
            cls_token: Classification token
            mask_token: Mask token for MLM
        """
        self.do_lower_case = do_lower_case
        self.max_len = max_len
        
        # Special tokens
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        
        # Load or create vocabulary
        if vocab_file and os.path.exists(vocab_file):
            self.vocab = self._load_vocab(vocab_file)
        else:
            self.vocab = self._create_basic_vocab()
            
        # Create reverse mapping
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Initialize WordPiece tokenizer
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
        
        # Special token IDs
        self.unk_token_id = self.vocab[self.unk_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.mask_token_id = self.vocab[self.mask_token]
    
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from file."""
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                vocab[token] = idx
        return vocab
    
    def _create_basic_vocab(self) -> Dict[str, int]:
        """Create a basic vocabulary for demonstration."""
        # Start with special tokens
        special_tokens = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]
        
        # Add common English words and subwords
        common_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "is", "was", "are", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
            "what", "when", "where", "why", "how", "who", "which", "there", "here",
            "hello", "world", "example", "test", "tokenization", "bert", "model", "language",
            "processing", "natural", "machine", "learning", "ai", "artificial", "intelligence"
        ]
        
        # Add common prefixes and suffixes (WordPiece style)
        subwords = [
            "##ing", "##ed", "##er", "##est", "##ly", "##tion", "##ness", "##ment", "##able",
            "##ful", "##less", "##ize", "##ization", "##al", "##ic", "##ous", "##ive",
            "un##", "re##", "pre##", "de##", "dis##", "over##", "under##", "out##",
            "##s", "##es", "##'s", "##n't", "##'re", "##'ve", "##'ll", "##'d"
        ]
        
        # Add individual characters and common substrings
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
        punctuation = list(".,!?;:()[]{}\"'-_/\\@#$%^&*+=<>|`~")
        
        # Combine all tokens
        all_tokens = special_tokens + common_words + subwords + chars + punctuation
        
        # Create vocabulary mapping
        vocab = {}
        for idx, token in enumerate(all_tokens):
            vocab[token] = idx
            
        return vocab
    
    def normalize_text(self, text: str) -> str:
        """Normalize text (Unicode normalization, lowercasing)."""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Lowercase if required
        if self.do_lower_case:
            text = text.lower()
            
        return text
    
    def basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization - split on whitespace and punctuation."""
        text = self.normalize_text(text)
        
        # Split on whitespace
        tokens = []
        for token in text.split():
            # Further split on punctuation
            tokens.extend(self._split_on_punctuation(token))
            
        return [token for token in tokens if token.strip()]
    
    def _split_on_punctuation(self, text: str) -> List[str]:
        """Split text on punctuation characters."""
        tokens = []
        current_token = ""
        
        for char in text:
            if self._is_punctuation(char):
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
                
        if current_token:
            tokens.append(current_token)
            
        return tokens
    
    def _is_punctuation(self, char: str) -> bool:
        """Check if character is punctuation."""
        cp = ord(char)
        # ASCII punctuation
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        # Unicode punctuation categories
        cat = unicodedata.category(char)
        return cat.startswith("P")
    
    def tokenize(self, text: str) -> List[str]:
        """Complete tokenization: basic tokenization + WordPiece."""
        basic_tokens = self.basic_tokenize(text)
        wordpiece_tokens = []
        
        for token in basic_tokens:
            wordpiece_tokens.extend(self.wordpiece_tokenizer.tokenize(token))
            
        return wordpiece_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True, 
               max_length: Optional[int] = None, padding: bool = False,
               truncation: bool = False) -> Dict[str, List[int]]:
        """
        Encode text to token IDs.
        
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'token_type_ids'
        """
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # Handle max_length
        if max_length is None:
            max_length = self.max_len
            
        # Truncation
        if truncation and len(tokens) > max_length:
            if add_special_tokens:
                # Keep [CLS] and [SEP], truncate middle
                tokens = tokens[:max_length-1] + [self.sep_token]
            else:
                tokens = tokens[:max_length]
        
        # Convert to IDs
        input_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Padding
        if padding and len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        # Token type IDs (all 0s for single sentence)
        token_type_ids = [0] * len(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'tokens': tokens
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            token = self.ids_to_tokens.get(token_id, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.cls_token, self.sep_token]:
                continue
            tokens.append(token)
        
        # Join tokens and clean up WordPiece artifacts
        text = " ".join(tokens)
        text = text.replace(" ##", "")  # Remove WordPiece continuation markers
        
        return text.strip()
    
    def encode_pair(self, text_a: str, text_b: str, add_special_tokens: bool = True,
                   max_length: Optional[int] = None, padding: bool = False,
                   truncation: bool = False) -> Dict[str, List[int]]:
        """Encode a pair of texts (for sentence pair tasks)."""
        tokens_a = self.tokenize(text_a)
        tokens_b = self.tokenize(text_b)
        
        if max_length is None:
            max_length = self.max_len
        
        # Handle truncation for pairs
        if truncation:
            total_length = len(tokens_a) + len(tokens_b)
            if add_special_tokens:
                total_length += 3  # [CLS], [SEP], [SEP]
            
            if total_length > max_length:
                # Truncate longer sequence first
                excess = total_length - max_length
                if len(tokens_a) > len(tokens_b):
                    tokens_a = tokens_a[:-excess] if excess < len(tokens_a) else []
                else:
                    tokens_b = tokens_b[:-excess] if excess < len(tokens_b) else []
        
        # Build token sequence
        if add_special_tokens:
            tokens = [self.cls_token] + tokens_a + [self.sep_token] + tokens_b + [self.sep_token]
            token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        else:
            tokens = tokens_a + tokens_b
            token_type_ids = [0] * len(tokens_a) + [1] * len(tokens_b)
        
        # Convert to IDs
        input_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        attention_mask = [1] * len(input_ids)
        
        # Padding
        if padding and len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            token_type_ids.extend([0] * padding_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'tokens': tokens
        }


class WordPieceTokenizer:
    """WordPiece tokenizer implementation."""
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_input_chars_per_word: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using WordPiece algorithm."""
        output_tokens = []
        
        for token in text.split():
            if len(token) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
                
            # Greedy longest-match
            is_bad = False
            start = 0
            sub_tokens = []
            
            while start < len(token):
                end = len(token)
                cur_substr = None
                
                # Find longest matching substring
                while start < end:
                    substr = token[start:end]
                    if start > 0:
                        substr = "##" + substr
                    
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                
                if cur_substr is None:
                    is_bad = True
                    break
                    
                sub_tokens.append(cur_substr)
                start = end
            
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
                
        return output_tokens


def save_tokenizer(tokenizer: BertTokenizer, save_dir: str):
    """Save tokenizer vocabulary and config."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save vocabulary
    vocab_file = os.path.join(save_dir, "vocab.txt")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    
    # Save config
    config = {
        "do_lower_case": tokenizer.do_lower_case,
        "max_len": tokenizer.max_len,
        "unk_token": tokenizer.unk_token,
        "sep_token": tokenizer.sep_token,
        "pad_token": tokenizer.pad_token,
        "cls_token": tokenizer.cls_token,
        "mask_token": tokenizer.mask_token
    }
    
    config_file = os.path.join(save_dir, "tokenizer_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Tokenizer saved to {save_dir}")


def load_tokenizer(load_dir: str) -> BertTokenizer:
    """Load tokenizer from saved directory."""
    vocab_file = os.path.join(load_dir, "vocab.txt")
    config_file = os.path.join(load_dir, "tokenizer_config.json")
    
    # Load config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Create tokenizer
    tokenizer = BertTokenizer(
        vocab_file=vocab_file,
        **config
    )
    
    print(f"Tokenizer loaded from {load_dir}")
    return tokenizer


if __name__ == "__main__":
    # Demo usage
    tokenizer = BertTokenizer(do_lower_case=True)
    
    text = "Hello, world! This is a test of BERT tokenization."
    print(f"Original text: {text}")
    
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    
    encoded = tokenizer.encode(text, padding=True, max_length=20)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: {decoded}")
