# Stopwords Filtering in NLP Analysis

## What are Stopwords?

Stopwords are common words that appear frequently in text but carry little semantic meaning for analysis. Examples include:
- **Articles**: the, a, an
- **Prepositions**: in, on, at, to, for, from, with
- **Pronouns**: he, she, it, we, they, you, I
- **Conjunctions**: and, or, but
- **Auxiliary verbs**: is, was, are, been, have, has

## Why Remove Stopwords?

✅ **Improved Signal-to-Noise Ratio**
- Focus on meaningful keywords rather than grammatical function words
- Stopwords like "the" and "and" can distort frequency analysis

✅ **Better Semantic Understanding**
- Word frequency analysis becomes more interpretable
- Topic modeling and keyword extraction work better

✅ **Standard NLP Practice**
- Widely used in information retrieval, text mining, and machine learning
- Industry standard in search engines and document clustering

✅ **Reduced Computational Cost**
- Fewer words to process and analyze
- Smaller vocabulary size

## How to Use in This Project

### Basic Usage (Stopwords Removed - Default)
```python
dataset = RadicalisationDataset(root_dir)
# Stopwords are removed by default
dataset.analyze_content_statistics(top_n=30, save_dir='./analysis_results', remove_stopwords=True)
```

### Without Stopwords Removal
```python
# Keep all words including common ones
dataset.analyze_content_statistics(top_n=30, save_dir='./analysis_results', remove_stopwords=False)
```

## Implementation Details

### Available Stopwords Lists

1. **NLTK Stopwords** (Recommended)
   - Requires: `pip install nltk`
   - Download data: `python -m nltk.downloader stopwords`
   - Comprehensive list of 179 common English words
   - Highest quality, widely used in NLP research

2. **Default Fallback List**
   - Included in code if NLTK not available
   - ~73 most common English words
   - Covers articles, prepositions, pronouns, auxiliary verbs

### How the Code Works

```python
# Get stopwords
stopwords_set = get_stopwords()  # Returns set of stopwords

# During word processing
for content in contents:
    words = str(content).lower().split()
    words = [re.sub(r'[^\w\s]', '', word) for word in words]  # Remove punctuation
    words = [w for w in words if w not in stopwords_set]      # Remove stopwords
    all_words.extend(words)
```

## Example Results

### Word Frequency Analysis

**With Stopwords:**
```
Rank | Word    | Frequency
  1  | the     | 2,543
  2  | a       | 1,892
  3  | and     | 1,654
  4  | to      | 1,421
  5  | ISIS    | 892      ← First meaningful word!
```

**Without Stopwords (This Project):**
```
Rank | Word      | Frequency
  1  | ISIS      | 892
  2  | fighter   | 456
  3  | Syria     | 378
  4  | groups    | 312
  5  | support   | 289
```

## Output

The analysis generates:
- `word_frequency.png` - Bar chart of top N words (with/without stopwords)
- `text_length_distribution.png` - Histogram of text lengths
- `wordcloud.png` - Visual word cloud of most common terms
- `category_distribution.png` - Sample count per category

Chart titles automatically indicate if stopwords were removed:
- "Word Frequency Distribution (Top 30 words)(Stopwords removed)"

## Best Practices

1. **Always use stopwords for frequency analysis** ✅
   - More interpretable results
   - Matches standard NLP practices

2. **Can optionally keep stopwords for:**
   - Document length analysis
   - Grammatical structure study
   - Stylometry analysis

3. **Customize stopwords if needed**
   - Add domain-specific words to filter
   - Remove words that shouldn't be filtered

## References

- NLTK Stopwords: https://www.nltk.org/howto/wordnet.html
- Common stopwords: https://en.wikipedia.org/wiki/Stop_words
- NLP best practices: https://nlp.stanford.edu/

## Installation

If NLTK stopwords are not available, install them:
```bash
pip install nltk
python -m nltk.downloader stopwords
```

Or the code will automatically fall back to the built-in stopwords list.
