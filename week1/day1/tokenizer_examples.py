#!/usr/bin/env python3
"""
BERT Tokenizer Examples and Validation
Comprehensive test cases and examples for the BERT tokenizer implementation
"""

from bert_tokenizer import BertTokenizer, save_tokenizer, load_tokenizer
import os
import json


def print_separator(title: str):
    """Print a formatted separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_tokenization_result(text: str, tokenizer: BertTokenizer, example_num: int):
    """Print detailed tokenization results."""
    print(f"\nExample {example_num}:")
    print(f"Input: '{text}'")
    
    # Basic tokenization
    basic_tokens = tokenizer.basic_tokenize(text)
    print(f"Basic tokens: {basic_tokens}")
    
    # Full tokenization
    tokens = tokenizer.tokenize(text)
    print(f"WordPiece tokens: {tokens}")
    
    # Encoding
    encoded = tokenizer.encode(text, add_special_tokens=True, padding=True, max_length=32)
    print(f"Token IDs: {encoded['input_ids']}")
    print(f"Attention mask: {encoded['attention_mask']}")
    print(f"Tokens with special: {encoded['tokens']}")
    
    # Decoding
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: '{decoded}'")
    
    # Verify round-trip (should match original after normalization)
    original_normalized = tokenizer.normalize_text(text)
    round_trip_match = original_normalized.replace(" ", "").replace("'", "") == decoded.replace(" ", "").replace("'", "")
    print(f"Round-trip quality: {'✓' if round_trip_match or abs(len(original_normalized) - len(decoded)) <= 2 else '✗'}")


def run_basic_examples():
    """Run basic tokenization examples."""
    print_separator("Basic Tokenization Examples")
    
    tokenizer = BertTokenizer(do_lower_case=True)
    
    examples = [
        "Hello, world!",
        "This is a simple sentence.",
        "BERT tokenization works great!",
        "Testing punctuation: hello, world; how are you?",
        "Numbers and symbols: 123 + 456 = 579 @ #hashtag"
    ]
    
    for i, text in enumerate(examples, 1):
        print_tokenization_result(text, tokenizer, i)


def run_wordpiece_examples():
    """Run WordPiece specific examples."""
    print_separator("WordPiece Tokenization Examples")
    
    tokenizer = BertTokenizer(do_lower_case=True)
    
    examples = [
        "internationalization",  # Should split into subwords
        "unhappiness",           # Should split with prefixes/suffixes  
        "preprocessing",         # Should handle prefixes
        "tokenization",          # Should split meaningfully
        "antidisestablishmentarianism",  # Very long word
        "COVID-19",              # Mixed case with numbers
        "don't can't won't",     # Contractions
        "New York City"          # Proper nouns
    ]
    
    for i, text in enumerate(examples, 1):
        print_tokenization_result(text, tokenizer, i)


def run_sentence_pair_examples():
    """Run sentence pair tokenization examples."""
    print_separator("Sentence Pair Tokenization Examples")
    
    tokenizer = BertTokenizer(do_lower_case=True)
    
    pairs = [
        ("The cat sat on the mat.", "The dog ran in the park."),
        ("BERT is a transformer model.", "It uses attention mechanisms."),
        ("Question: What is NLP?", "Answer: Natural Language Processing."),
        ("First sentence is short.", "This second sentence is considerably longer and should test truncation."),
    ]
    
    for i, (text_a, text_b) in enumerate(pairs, 1):
        print(f"\nSentence Pair Example {i}:")
        print(f"Text A: '{text_a}'")
        print(f"Text B: '{text_b}'")
        
        encoded = tokenizer.encode_pair(
            text_a, text_b, 
            add_special_tokens=True, 
            padding=True, 
            max_length=32,
            truncation=True
        )
        
        print(f"Tokens: {encoded['tokens']}")
        print(f"Token IDs: {encoded['input_ids']}")
        print(f"Token type IDs: {encoded['token_type_ids']}")
        print(f"Attention mask: {encoded['attention_mask']}")
        
        # Decode
        decoded = tokenizer.decode(encoded['input_ids'])
        print(f"Decoded: '{decoded}'")


def run_edge_cases():
    """Run edge case examples."""
    print_separator("Edge Cases and Special Scenarios")
    
    tokenizer = BertTokenizer(do_lower_case=True)
    
    edge_cases = [
        "",                          # Empty string
        " ",                         # Just whitespace
        "a",                         # Single character
        "A"*200,                     # Very long repeated character
        "🤗 😊 🚀",                   # Emojis
        "café naïve résumé",         # Accented characters
        "Hello\nworld\ttab",         # Whitespace characters
        "!!!???",                    # Repeated punctuation
        "don't",                     # Contractions
        "U.S.A.",                    # Abbreviations with periods
        "user@example.com",          # Email
        "https://www.example.com",   # URL
        "$100.50",                   # Currency
        "3.14159",                   # Decimal numbers
        "C++",                       # Programming languages
        "re-evaluate",               # Hyphenated words
    ]
    
    for i, text in enumerate(edge_cases, 1):
        print_tokenization_result(text, tokenizer, i)


def run_comparison_with_huggingface():
    """Compare with Hugging Face BERT tokenizer (if available)."""
    print_separator("Comparison with Hugging Face BERT Tokenizer")
    
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        our_tokenizer = BertTokenizer(do_lower_case=True)
        
        test_sentences = [
            "Hello, world!",
            "BERT tokenization example",
            "internationalization preprocessing",
            "don't can't won't"
        ]
        
        for i, text in enumerate(test_sentences, 1):
            print(f"\nComparison Example {i}: '{text}'")
            
            # Our tokenizer
            our_tokens = our_tokenizer.tokenize(text)
            our_encoded = our_tokenizer.encode(text, add_special_tokens=True)
            
            # Hugging Face tokenizer
            hf_tokens = hf_tokenizer.tokenize(text)
            hf_encoded = hf_tokenizer(text, return_tensors="pt")
            
            print(f"Our tokens:     {our_tokens}")
            print(f"HF tokens:      {hf_tokens}")
            print(f"Our IDs:        {our_encoded['input_ids'][:10]}")  # First 10 IDs
            print(f"HF IDs:         {hf_encoded['input_ids'][0][:10].tolist()}")
            
            # Note: IDs will be different due to different vocabularies
            similarity = len(set(our_tokens) & set(hf_tokens)) / max(len(our_tokens), len(hf_tokens))
            print(f"Token similarity: {similarity:.2%}")
            
    except ImportError:
        print("Hugging Face transformers not available. Skipping comparison.")
        print("Install with: pip install transformers")


def test_save_load_functionality():
    """Test saving and loading tokenizer."""
    print_separator("Save/Load Functionality Test")
    
    # Create and save tokenizer
    original_tokenizer = BertTokenizer(do_lower_case=True)
    save_dir = "/tmp/bert_tokenizer_test"
    
    print("Saving tokenizer...")
    save_tokenizer(original_tokenizer, save_dir)
    
    # Load tokenizer
    print("Loading tokenizer...")
    loaded_tokenizer = load_tokenizer(save_dir)
    
    # Test that they work the same
    test_text = "Testing save and load functionality!"
    
    original_tokens = original_tokenizer.tokenize(test_text)
    loaded_tokens = loaded_tokenizer.tokenize(test_text)
    
    original_encoded = original_tokenizer.encode(test_text)
    loaded_encoded = loaded_tokenizer.encode(test_text)
    
    print(f"Test text: '{test_text}'")
    print(f"Original tokens: {original_tokens}")
    print(f"Loaded tokens: {loaded_tokens}")
    print(f"Tokens match: {'✓' if original_tokens == loaded_tokens else '✗'}")
    print(f"Encodings match: {'✓' if original_encoded['input_ids'] == loaded_encoded['input_ids'] else '✗'}")
    
    # Clean up
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print("Cleaned up temporary files.")


def run_performance_test():
    """Run basic performance test."""
    print_separator("Performance Test")
    
    import time
    
    tokenizer = BertTokenizer(do_lower_case=True)
    
    # Test with various text lengths
    texts = [
        "Short text.",
        "This is a medium length text that should take a bit longer to tokenize than the short one.",
        " ".join(["This is a long text."] * 50),  # Very long text
    ]
    
    for i, text in enumerate(texts, 1):
        start_time = time.time()
        tokens = tokenizer.tokenize(text)
        encoded = tokenizer.encode(text, add_special_tokens=True, padding=True, max_length=512)
        decoded = tokenizer.decode(encoded['input_ids'])
        end_time = time.time()
        
        print(f"\nPerformance Test {i}:")
        print(f"Text length: {len(text)} characters")
        print(f"Token count: {len(tokens)}")
        print(f"Processing time: {(end_time - start_time)*1000:.2f}ms")
        print(f"Tokens per second: {len(tokens)/(end_time - start_time):.0f}")


def run_validation_checks():
    """Run validation checks to ensure tokenizer works correctly."""
    print_separator("Validation Checks")
    
    tokenizer = BertTokenizer(do_lower_case=True)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Special tokens are in vocabulary
    total_checks += 1
    special_tokens = [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, 
                     tokenizer.unk_token, tokenizer.mask_token]
    all_special_in_vocab = all(token in tokenizer.vocab for token in special_tokens)
    if all_special_in_vocab:
        checks_passed += 1
        print("✓ All special tokens are in vocabulary")
    else:
        print("✗ Some special tokens missing from vocabulary")
    
    # Check 2: Encoding/decoding round trip
    total_checks += 1
    test_text = "Hello world"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded['input_ids'])
    if test_text.lower().replace(" ", "") in decoded.replace(" ", ""):
        checks_passed += 1
        print("✓ Encoding/decoding round trip works")
    else:
        print("✗ Encoding/decoding round trip failed")
    
    # Check 3: Attention mask length matches input_ids
    total_checks += 1
    encoded = tokenizer.encode("Test sentence", padding=True, max_length=20)
    if len(encoded['input_ids']) == len(encoded['attention_mask']):
        checks_passed += 1
        print("✓ Attention mask length matches input_ids")
    else:
        print("✗ Attention mask length mismatch")
    
    # Check 4: Padding works correctly
    total_checks += 1
    encoded = tokenizer.encode("Short", padding=True, max_length=10)
    if len(encoded['input_ids']) == 10 and tokenizer.pad_token_id in encoded['input_ids']:
        checks_passed += 1
        print("✓ Padding works correctly")
    else:
        print("✗ Padding failed")
    
    # Check 5: Truncation works correctly
    total_checks += 1
    long_text = " ".join(["word"] * 100)
    encoded = tokenizer.encode(long_text, truncation=True, max_length=10)
    if len(encoded['input_ids']) <= 10:
        checks_passed += 1
        print("✓ Truncation works correctly")
    else:
        print("✗ Truncation failed")
    
    # Check 6: Sentence pair encoding
    total_checks += 1
    encoded = tokenizer.encode_pair("First sentence", "Second sentence")
    if 0 in encoded['token_type_ids'] and 1 in encoded['token_type_ids']:
        checks_passed += 1
        print("✓ Sentence pair encoding works")
    else:
        print("✗ Sentence pair encoding failed")
    
    print(f"\nValidation Results: {checks_passed}/{total_checks} checks passed")
    if checks_passed == total_checks:
        print("🎉 All validation checks passed! Tokenizer is working correctly.")
    else:
        print("⚠️  Some validation checks failed. Please review the implementation.")
    
    return checks_passed == total_checks


def main():
    """Run all examples and tests."""
    print("BERT Tokenizer - Comprehensive Examples and Validation")
    print("=" * 60)
    
    try:
        # Run all example categories
        run_basic_examples()
        run_wordpiece_examples()
        run_sentence_pair_examples()
        run_edge_cases()
        run_comparison_with_huggingface()
        test_save_load_functionality()
        run_performance_test()
        
        # Final validation
        validation_passed = run_validation_checks()
        
        if validation_passed:
            print("\n🎉 SUCCESS: BERT tokenizer implementation is working correctly!")
        else:
            print("\n⚠️  ATTENTION: Some issues detected in the tokenizer implementation.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
