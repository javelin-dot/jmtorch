> **文档来源**：`src/10_tokenization/10_tokenization.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 10: Tokenization - Converting Text to Numbers

Welcome to Module 10! You're about to build tokenization - the bridge that converts human-readable text into numerical representations that machine learning models can process.

## 🔗 Prerequisites & Progress
**You've Built**: Neural networks, layers, training loops, and data loading
**You'll Build**: Text tokenization systems (character and BPE-based)
**You'll Enable**: Text processing for language models and NLP tasks

**Connection Map**:
```
DataLoader → Tokenization → Embeddings
(batching)   (text→numbers)  (learnable representations)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement character-based tokenization for simple text processing
2. Build a BPE (Byte Pair Encoding) tokenizer for efficient text representation
3. Understand vocabulary management and encoding/decoding operations
4. Create the foundation for text processing in neural networks

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/10_tokenization/tokenization_dev.py
**Building Side:** Code exports to tinytorch.core.tokenization

```python
# Final package structure:
from tinytorch.core.tokenization import Tokenizer, CharTokenizer, BPETokenizer
```

**Why this matters:**
- **Learning:** Complete tokenization system in one focused module for deep understanding
- **Production:** Proper organization like Hugging Face's tokenizers with all text processing together
- **Consistency:** All tokenization operations and vocabulary management in core.tokenization
- **Integration:** Works seamlessly with embeddings and data loading for complete NLP pipeline

## 📋 Module Dependencies

**Prerequisites**: Module 10 is relatively independent - it mainly works with strings and numbers!

**External Dependencies**:
- `numpy` (for numerical operations and statistics)
- `collections.Counter` (for frequency counting)

**TinyTorch Dependencies**:
- Module 01 (Tensor): Optional - only needed if converting tokens to Tensor format

**Important**: This module focuses on text processing fundamentals that work independently.
The tokenization algorithms use only standard Python and NumPy.

**Dependency Flow**:
```
Module 10 (Tokenization) → Module 11 (Embeddings)
     ↓
  Text to numbers (token IDs) for neural network input
```

Students completing this module will have built the text processing foundation
that enables all NLP tasks in TinyTorch.

## 💡 Introduction: Why Tokenization?

Neural networks operate on numbers, but humans communicate with text. Tokenization is the crucial bridge that converts text into numerical sequences that models can process.

### The Text-to-Numbers Challenge

Consider the sentence: "Hello, world!" - how do we turn this into numbers a neural network can process?

```
┌─────────────────────────────────────────────────────────────────┐
│  TOKENIZATION PIPELINE: Text → Numbers                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (Human Text):     "Hello, world!"                        │
│           │                                                     │
│           ├─ Step 1: Split into tokens                          │
│           │         ['H','e','l','l','o',',', ...']             │
│           │                                                     │
│           ├─ Step 2: Map to vocabulary IDs                      │
│           │         [72, 101, 108, 108, 111, ...]               │
│           │                                                     │
│           ├─ Step 3: Handle unknowns                            │
│           │         Unknown chars → special <UNK> token         │
│           │                                                     │
│           └─ Step 4: Enable decoding                            │
│                     IDs → original text                         │
│                                                                 │
│  Output (Token IDs):  [72, 101, 108, 108, 111, 44, 32, ...]     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Four-Step Process

How do we represent text for a neural network? We need a systematic pipeline:

**1. Split text into tokens** - Break text into meaningful units (words, subwords, or characters)
**2. Map tokens to integers** - Create a vocabulary that assigns each token a unique ID
**3. Handle unknown text** - Deal gracefully with tokens not seen during training
**4. Enable reconstruction** - Convert numbers back to readable text for interpretation

### Why This Matters

The choice of tokenization strategy dramatically affects:
- **Model performance** - How well the model understands text
- **Vocabulary size** - Memory requirements for embedding tables
- **Computational efficiency** - Sequence length affects processing time
- **Robustness** - How well the model handles new/rare words

## 📐 Foundations: Tokenization Strategies

Different tokenization approaches make different trade-offs between vocabulary size, sequence length, and semantic understanding.

### Character-Level Tokenization
**Approach**: Each character gets its own token

```
┌──────────────────────────────────────────────────────────────┐
│ CHARACTER TOKENIZATION PROCESS                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Build Vocabulary from Unique Characters             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Corpus: ["hello", "world"]                             │  │
│  │                ↓                                       │  │
│  │ Unique chars: ['h', 'e', 'l', 'o', 'w', 'r', 'd']      │  │
│  │                ↓                                       │  │
│  │ Vocabulary:  ['<UNK>','h','e','l','o','w','r','d']     │  │
│  │ IDs:            0      1   2   3   4   5   6   7       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 2: Encode Text Character by Character                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Text: "hello"                                         │  │
│  │                                                        │  │
│  │   'h' → 1    (lookup in vocabulary)                    │  │
│  │   'e' → 2                                              │  │
│  │   'l' → 3                                              │  │
│  │   'l' → 3                                              │  │
│  │   'o' → 4                                              │  │
│  │                                                        │  │
│  │  Result: [1, 2, 3, 3, 4]                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 3: Decode by Reversing ID Lookup                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  IDs: [1, 2, 3, 3, 4]                                  │  │
│  │                                                        │  │
│  │   1 → 'h'    (reverse lookup)                          │  │
│  │   2 → 'e'                                              │  │
│  │   3 → 'l'                                              │  │
│  │   3 → 'l'                                              │  │
│  │   4 → 'o'                                              │  |
│  │                                                        │  │
│  │  Result: "hello"                                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Pros**:
- Small vocabulary (~100 chars)
- Handles any text perfectly
- No unknown tokens (every character can be mapped)
- Simple implementation

**Cons**:
- Long sequences (1 character = 1 token)
- Limited semantic understanding (no word boundaries)
- More compute (longer sequences to process)

### Word-Level Tokenization
**Approach**: Each word gets its own token

```
Text: "Hello world"
       ↓
Tokens: ['Hello', 'world']
       ↓
IDs:    [5847, 1254]
```

**Pros**: Semantic meaning preserved, shorter sequences
**Cons**: Huge vocabularies (100K+), many unknown tokens

### Subword Tokenization (BPE)
**Approach**: Learn frequent character pairs, build subword units

```
Text: "tokenization"
       ↓ Character level
Initial: ['t', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']
       ↓ Learn frequent pairs
Merged: ['to', 'ken', 'ization']
       ↓
IDs:    [142, 1847, 2341]
```

**Pros**: Balance between vocabulary size and sequence length
**Cons**: More complex training process

### Strategy Comparison

```
Text: "tokenization" (12 characters)

Character: ['t','o','k','e','n','i','z','a','t','i','o','n'] → 12 tokens, vocab ~100
Word:      ['tokenization']                                   → 1 token, vocab 100K+
BPE:       ['token','ization']                               → 2 tokens, vocab 10-50K
```

The sweet spot for most applications is BPE with 10K-50K vocabulary size.

## 🏗️ Implementation: Building Tokenization Systems

Let's build our tokenization systems step by step, testing each component as we go.

### Tokenization Class Architecture

```
Tokenization System Structure:
┌─────────────────────────────────┐
│ Base Tokenizer Interface:       │
│ • encode(text) → token_ids      │
│ • decode(token_ids) → text      │
├─────────────────────────────────┤
│ CharTokenizer (Simple):         │
│ • vocab: list of characters     │
│ • char_to_id: lookup mapping    │
│ • id_to_char: reverse mapping   │
├─────────────────────────────────┤
│ BPETokenizer (Advanced):        │
│ • vocab: learned subwords       │
│ • merges: learned pair rules    │
│ • token_to_id/id_to_token maps  │
├─────────────────────────────────┤
│ Utility Functions:              │
│ • create_tokenizer()            │
│ • tokenize_dataset()            │
│ • analyze_tokenization()        │
└─────────────────────────────────┘
```

This clean design provides a consistent interface across different tokenization strategies.

### Base Tokenizer Interface

All tokenizers need to provide two core operations: encoding text to numbers and decoding numbers back to text. Let's define the common interface.

```
Tokenizer Interface:
    encode(text) → [id1, id2, id3, ...]
    decode([id1, id2, id3, ...]) → text
```

This ensures consistent behavior across different tokenization strategies.

### 🧪 Unit Test: Base Tokenizer Interface

This test validates our base tokenizer defines the correct interface for all implementations.

**What we're testing**: Abstract interface definition with NotImplementedError
**Why it matters**: Ensures consistent API across all tokenizer types
**Expected**: Base class raises NotImplementedError for both encode and decode

## 🏗️ Character-Level Tokenizer

The simplest tokenization approach: each character becomes a token. This gives us perfect coverage of any text but produces long sequences.

```
Character Tokenization Process:

Step 1: Build vocabulary from unique characters
Text corpus: ["hello", "world"]
Unique chars: ['h', 'e', 'l', 'o', 'w', 'r', 'd']
Vocabulary: ['<UNK>', 'h', 'e', 'l', 'o', 'w', 'r', 'd']  # <UNK> for unknown
                0      1    2    3    4    5    6    7

Step 2: Encode text character by character
Text: "hello"
  'h' → 1
  'e' → 2
  'l' → 3
  'l' → 3
  'o' → 4
Result: [1, 2, 3, 3, 4]

Step 3: Decode by looking up each ID
IDs: [1, 2, 3, 3, 4]
  1 → 'h'
  2 → 'e'
  3 → 'l'
  3 → 'l'
  4 → 'o'
Result: "hello"
```

### 🧪 Unit Test: Character Tokenizer

This test validates our character tokenizer works correctly with vocabulary building, encoding, and decoding.

**What we're testing**: Character-level tokenization with vocabulary management
**Why it matters**: Foundation for text processing with perfect coverage
**Expected**: Correct encoding/decoding, unknown character handling, vocabulary building

Character tokenization provides a simple, robust foundation for text processing. The key insight is that with a small vocabulary (typically <100 characters), we can represent any text without unknown tokens.

**Trade-offs**:
- **Pro**: No out-of-vocabulary issues, handles any language
- **Con**: Long sequences (1 char = 1 token), limited semantic understanding
- **Use case**: When robustness is more important than efficiency

## 🏗️ Byte Pair Encoding (BPE) Tokenizer

BPE is the secret sauce behind modern language models (GPT, BERT, etc.). It learns to merge frequent character pairs, creating subword units that balance vocabulary size with sequence length.

```
┌───────────────────────────────────────────────────────────────────────┐
│ BPE TRAINING ALGORITHM: Learning Subword Units                        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ STEP 1: Initialize with Character Vocabulary                          │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Training Data: ["hello", "hello", "help"]                         │ │
│ │                                                                   │ │
│ │ Initial Tokens (with end-of-word markers):                        │ │
│ │   ['h','e','l','l','o</w>']    (hello)                            │ │
│ │   ['h','e','l','l','o</w>']    (hello)                            │ │
│ │   ['h','e','l','p</w>']        (help)                             │ │
│ │                                                                   │ │
│ │ Starting Vocab: ['h', 'e', 'l', 'o', 'p', '</w>']                 │ │
│ │                   ↑ All unique characters                         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 2: Count All Adjacent Pairs                                      │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Pair Frequency Analysis:                                          │ │
│ │                                                                   │ │
│ │   ('h', 'e'): ██████  3 occurrences  ← MOST FREQUENT!             │ │
│ │   ('e', 'l'): ██████  3 occurrences                               │ │
│ │   ('l', 'l'): ████    2 occurrences                               │ │
│ │   ('l', 'o'): ████    2 occurrences                               │ │
│ │   ('o', '<'): ████    2 occurrences                               │ │
│ │   ('l', 'p'): ██      1 occurrence                                │ │
│ │   ('p', '<'): ██      1 occurrence                                │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 3: Merge Most Frequent Pair                                      │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Merge Operation: ('h', 'e') → 'he'                                │ │
│ │                                                                   │ │
│ │ BEFORE:                          AFTER:                           │ │
│ │   ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']            │ │
│ │   ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']            │ │
│ │   ['h','e','l','p</w>']      →  ['he','l','p</w>']                │ │
│ │                                                                   │ │
│ │ Updated Vocab: ['h','e','l','o','p','</w>', 'he']                 │ │
│ │                                              ↑ NEW TOKEN!         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 4: Repeat Until Target Vocab Size Reached                        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Iteration 2: Next most frequent is ('l', 'l')                     │ │
│ │ Merge ('l','l') → 'll'                                            │ │
│ │                                                                   │ │
│ │   ['he','l','l','o</w>']     →  ['he','ll','o</w>']               │ │
│ │   ['he','l','l','o</w>']     →  ['he','ll','o</w>']               │ │
│ │   ['he','l','p</w>']         →  ['he','l','p</w>']                │ │
│ │                                                                   │ │
│ │ Updated Vocab: ['h','e','l','o','p','</w>','he','ll']             │ │
│ │                                                  ↑ NEW!           │ │
│ │                                                                   │ │
│ │ Continue merging until vocab_size target...                       │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ FINAL RESULTS:                                                        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Trained BPE can now encode efficiently:                           │ │
│ │                                                                   │ │
│ │ "hello" → ['he', 'll', 'o</w>']  = 3 tokens (vs 5 chars)          │ │
│ │ "help"  → ['he', 'l', 'p</w>']   = 3 tokens (vs 4 chars)          │ │
│ │                                                                   │ │
│ │  Key Insights: BPE automatically discovers:                       │ │
│ │    - Common prefixes ('he')                                       │ │
│ │    - Morphological patterns ('ll')                                │ │
│ │    - Natural word boundaries (</w>)                               │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Why BPE Works**: By starting with characters and iteratively merging frequent pairs, BPE discovers the natural statistical patterns in language. Common words become single tokens, rare words split into recognizable subword pieces!

### Counting Byte Pairs

The first step in each BPE iteration is counting how often each adjacent token pair
appears across all words, weighted by word frequency. This tells us which pair to merge next.

```
Count Pairs Across All Words (weighted by frequency):

  word_tokens:                     word_freq:
  "hello" → ['h','e','l','l','o</w>']    freq=3
  "help"  → ['h','e','l','p</w>']        freq=1

  Pair counting (freq-weighted):
    ('h','e'):  3+1 = 4   ← appears in both words
    ('e','l'):  3+1 = 4   ← appears in both words
    ('l','l'):  3   = 3   ← only in "hello"
    ('l','o</w>'): 3 = 3  ← only in "hello"
    ('l','p</w>'): 1 = 1  ← only in "help"
```

### 🧪 Unit Test: Count Byte Pairs

**What we're testing**: Frequency-weighted pair counting across multiple words
**Why it matters**: The most frequent pair determines which merge to perform next
**Expected**: Pairs appearing in frequent words get higher counts

### Merging a Byte Pair

Once we identify the most frequent pair, we need to merge it everywhere it appears.
This scans through every word's token list and replaces adjacent occurrences of the
pair with a single concatenated token.

```
Merge Operation: ('h', 'e') → 'he'

BEFORE merging:                    AFTER merging:
  "hello" → ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']
  "help"  → ['h','e','l','p</w>']      →  ['he','l','p</w>']

Algorithm (linear scan per word):
  i=0: tokens[0]='h', tokens[1]='e' → match! append 'he', skip 2
  i=2: tokens[2]='l' → no match, append 'l', advance 1
  ...continue until end of tokens
```

### 🧪 Unit Test: Merge Pair

**What we're testing**: In-place merging of a specific pair across all word token lists
**Why it matters**: This is the core operation that builds the BPE vocabulary
**Expected**: Adjacent pair occurrences replaced by concatenated token, non-matching tokens preserved

### 🧪 Unit Test: BPE Tokenizer

This test validates our BPE tokenizer learns merge rules and correctly encodes/decodes text.

**What we're testing**: BPE training, merge rule application, encoding and decoding
**Why it matters**: BPE is the standard tokenization for modern language models
**Expected**: Vocabulary building, proper merging, reasonable round-trip on training data

BPE provides a balance between vocabulary size and sequence length. By learning frequent subword patterns, it can handle new words through decomposition while maintaining reasonable sequence lengths.

```
BPE Merging Visualization:

Original: "tokenization" → ['t','o','k','e','n','i','z','a','t','i','o','n','</w>']
                                                       ↓ Merge frequent pairs
Step 1:   ('t','o') is frequent → ['to','k','e','n','i','z','a','t','i','o','n','</w>']
Step 2:   ('i','o') is frequent → ['to','k','e','n','io','z','a','t','io','n','</w>']
Step 3:   ('io','n') is frequent → ['to','k','e','n','io','z','a','t','ion','</w>']
Step 4:   ('to','k') is frequent → ['tok','e','n','io','z','a','t','ion','</w>']
                                                       ↓ Continue merging...
Final:    "tokenization" → ['token','ization']  # 2 tokens vs 13 characters!
```

**Key insights**:
- **Adaptive vocabulary**: Learns from data, not hand-crafted
- **Subword robustness**: Handles rare/new words through decomposition
- **Efficiency trade-off**: Larger vocabulary → shorter sequences → faster processing
- **Morphological awareness**: Naturally discovers prefixes, suffixes, roots

## 🔧 Integration: Bringing It Together

Let's test how our tokenization components work together in realistic scenarios that mirror NLP pipelines. This integration demonstrates that our individual tokenizers combine correctly for complete text processing workflows.

### Tokenization Pipeline

```
Tokenization Workflow:

1. Choose Strategy → 2. Train Tokenizer → 3. Process Dataset → 4. Analyze Results
      ↓                      ↓                    ↓                   ↓
   char/bpe           corpus training        batch encoding      stats/metrics

Real NLP Pipeline:
Raw Text → Tokenization → Token IDs → Embedding Layer → Neural Network
   ↓           ↓             ↓              ↓                ↓
 Strings    Vocabulary    Integers     Dense Vectors    Predictions
```

### Why This Integration Matters

This simulation shows how our tokenization components combine to create the preprocessing building blocks of NLP:

- **Factory Pattern**: Easily create and configure tokenizers
- **Dataset Processing**: Tokenize batches with consistent length limits
- **Analysis Tools**: Understand vocabulary coverage and compression
- **Round-Trip Validation**: Ensure text can be recovered from tokens

### 🧪 Unit Test: Tokenization Utilities

This test validates our utility functions for tokenizer creation, dataset processing, and analysis.

**What we're testing**: Factory pattern, batch tokenization, and analysis statistics
**Why it matters**: Essential for building NLP pipelines with consistent preprocessing
**Expected**: Correct tokenizer creation, length limits respected, meaningful statistics

## 📊 Systems Analysis: Tokenization Trade-offs

Let's understand the key systems concepts in tokenization: **vocabulary size vs sequence length trade-offs** and **memory implications**.

This analysis reveals why different tokenization strategies make different choices for different use cases.

### Memory Profiling: Actual Tokenizer Memory Usage

Let's measure the real memory footprint of different tokenization strategies. This is crucial for understanding resource requirements in production systems.

### Performance Benchmarking: Encoding/Decoding Speed

Speed matters in production! Let's measure how fast different tokenizers can process text.
This helps understand computational bottlenecks in NLP pipelines.

### Scaling Analysis: How BPE Training Time Grows

Understanding algorithmic complexity helps us predict performance on larger datasets.
Let's measure how BPE training time scales with corpus size.

### 📊 Performance Analysis: Vocabulary Size vs Sequence Length

The fundamental trade-off in tokenization creates a classic systems engineering challenge:

```
Tokenization Trade-off Spectrum:

Character          BPE-Small         BPE-Large         Word-Level
vocab: ~100    →   vocab: ~1K    →   vocab: ~50K   →   vocab: ~100K+
seq: very long →   seq: long     →   seq: medium   →   seq: short
memory: low    →   memory: med   →   memory: high  →   memory: very high
compute: high  →   compute: med  →   compute: low  →   compute: very low
coverage: 100% →   coverage: 99% →   coverage: 95% →   coverage: <80%
```

**Character tokenization (vocab ~100)**:
- Pro: Universal coverage, simple implementation, small embedding table
- Con: Long sequences (high compute), limited semantic units
- Use case: Morphologically rich languages, robust preprocessing

**BPE tokenization (vocab 10K-50K)**:
- Pro: Balanced efficiency, handles morphology, good coverage
- Con: Training complexity, domain-specific vocabularies
- Use case: Most modern language models (GPT, BERT family)

**Real-world scaling examples**:
```
GPT-3/4:     ~50K BPE tokens, avg 3-4 chars/token
BERT:        ~30K WordPiece tokens, avg 4-5 chars/token
T5:          ~32K SentencePiece tokens, handles 100+ languages
ChatGPT:     ~100K tokens with extended vocabulary
```

**Downstream memory impact of vocabulary size**:

Each token ID will eventually map to a vector of numbers (you'll build this in Module 11: Embeddings).
Larger vocabularies mean more mappings to store -- a vocab_size of 50K vs 100K doubles this storage.

```
Key Trade-off:
  Larger vocab → Shorter sequences → Less compute
  BUT larger vocab → More mappings to store → Harder to train
```

## 🧪 Module Integration Test

Let's test our complete tokenization system to ensure everything works together.

## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of tokenization and its systems implications:

### 1. Vocabulary Size and Storage
**Question**: You implemented tokenizers with different vocabulary sizes.

**Calculate**:
- A character tokenizer with vocab ~100 needs to store ~100 token-to-ID mappings
- A BPE tokenizer with vocab_size=50,000 needs to store ~50,000 token-to-ID mappings
- How much memory does each vocabulary lookup dictionary require? (Hint: each entry stores a string key and an integer value)

**Consider**:
- How does vocabulary storage scale with vocab size?
- BPE tokens are variable-length strings — how does this affect dictionary memory?
- What's the trade-off between vocabulary size and sequence length for downstream processing?

---

### 2. Sequence Length Trade-offs
**Question**: Your character tokenizer produces longer sequences than BPE. For the text "machine learning" (16 characters):

**Compare**:
- Character tokenizer produces ~16 tokens
- BPE tokenizer might produce ~3-4 tokens
- If processing batch_size=32 with max_length=512:
  - Character model processes: _____ total tokens per batch
  - BPE model processes: _____ total tokens per batch

**Think about**:
- Which produces longer sequences that downstream processing must handle?
- How does this affect training time for large language models?
- Why might longer sequences create computational challenges for the models that process them?

---

### 3. Tokenization Coverage and Robustness
**Question**: Your BPE tokenizer handles unknown words by decomposing into subwords.

**Consider**:
- Why is this better than word-level tokenization for real applications?
- What happens to model performance when many tokens map to <UNK>?
- How does vocabulary size affect the number of unknown decompositions?
- What languages or domains might struggle with English-trained BPE?

**Real-world context**: OpenAI's tiktoken uses ~100K tokens to handle multilingual text. Google's SentencePiece was designed specifically for language-agnostic tokenization.

---

### 4. Production Scale Considerations
**Question**: A production language model serves 1 million requests per day, each with average 500 tokens.

**Calculate**:
- Total tokens processed daily: _____ million
- If tokenization takes 0.1ms per token, how much time is spent tokenizing? _____ hours

**Design implications**:
- Why do production systems use Rust-based tokenizers (10-100x faster)?
- How does caching tokenized prompts help for repeated queries?
- What's the memory cost of keeping tokenizer vocabulary in RAM?

## ⭐ Aha Moment: Text Becomes Tokens

**What you built:** Tokenizers that convert text into numerical sequences that neural networks can process.

**Why it matters:** Neural networks can't read text - they need numbers! Your tokenizer bridges
this gap, converting words into token IDs that can be embedded and processed. Every language
model from GPT to Claude uses tokenization as the first step in understanding text.

Your tokenization system is ready for NLP applications.

## 🚀 MODULE SUMMARY: Tokenization

Congratulations! You've built a complete tokenization system for converting text to numerical representations!

### Key Accomplishments
- **Built a character-level tokenizer** with perfect text coverage and simple implementation
- **Implemented BPE tokenizer** that learns efficient subword representations from data
- **Created vocabulary management** with encoding/decoding and unknown token handling
- **Discovered the vocabulary size vs sequence length trade-off** through systems analysis
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: Embedding table size = vocab_size x embed_dim (can be 100+ MB)
- **Sequence length trade-offs**: BPE compresses text, reducing compute by 3-4x
- **Training complexity**: BPE training scales O(n^2) with corpus size
- **Production patterns**: Rust tokenizers are 10-100x faster than pure Python

### Ready for Next Steps
Your tokenization implementation enables text processing for language models.
Export with: `tito module complete 10`

**Next**: Module 11 will add learnable embeddings that convert your token IDs into rich vector representations!
