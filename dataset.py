import os
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import itertools
from collections import Counter
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Stopwords for filtering common words
try:
    from nltk.corpus import stopwords
    nltk_available = True
except ImportError:
    nltk_available = False
    # Default English stopwords if nltk not available
    DEFAULT_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'out', 'if', 'as', 'is',
        'was', 'are', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'so', 'than',
        'then', 'now', 'just', 'only', 'very', 'too', 'also', 'no', 'not',
        'yes', 'be', 'such', 'same', 'own', 'each', 'other', 'more', 'most'
    }

# Configure matplotlib for Chinese font support
import matplotlib
import platform
import warnings

# Auto-detect and configure Chinese font
def setup_matplotlib_font():
    """
    Auto-detect system fonts and configure matplotlib
    """
    system = platform.system()
    font_found = False
    
    if system == 'Windows':
        # Common Windows fonts (by priority)
        possible_fonts = [
            'SimHei',           # SimHei
            'Microsoft YaHei',  # Microsoft YaHei
            'SimSun',           # SimSun
            'KaiTi',            # KaiTi
            'FangSong',         # FangSong
        ]
        
        # Get system available fonts
        import matplotlib.font_manager as fm
        system_fonts = {f.name for f in fm.fontManager.ttflist}
        
        # Find first available font
        for font in possible_fonts:
            if font in system_fonts:
                matplotlib.rcParams['font.sans-serif'] = font
                font_found = True
                print(f"✓ Font detected: {font}")
                break
    
    # Set other parameters
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    if not font_found:
        print("⚠ No font detected. Please install one:")
        print("  Option 1: Download open-source font (recommended)")
        print("    - Visit: https://github.com/noto-fonts/noto-cjk/releases")
        print("    - Download: NotoSansCJKsc-Regular.otf")
        print("    - Save to: C:\\Users\\<username>\\AppData\\Local\\Microsoft\\Windows\\Fonts")
        print("  Option 2: Use system fonts")
        print("    - Copy any .ttf file to C:\\Windows\\Fonts")

# Execute font configuration
# setup_matplotlib_font()

CHINESE_FONT_PATH = None
# Detect system font file path
def get_chinese_font_path():
    """Auto-detect font file in Windows system"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        # Common Windows font paths
        possible_fonts = [
            'C:\\Windows\\Fonts\\simhei.ttf',      # SimHei
            'C:\\Windows\\Fonts\\msyh.ttc',        # Microsoft YaHei
            'C:\\Windows\\Fonts\\simsun.ttc',      # SimSun
            'C:\\Windows\\Fonts\\SimSun_18030.ttc' # SimSun
        ]
        for font_path in possible_fonts:
            if os.path.exists(font_path):
                return font_path
    
    return None

# CHINESE_FONT_PATH = get_chinese_font_path()

def get_stopwords(use_nltk=True):
    """
    Get English stopwords for filtering common words.
    Combines NLTK stopwords with custom Twitter/social media terms.
    
    Args:
        use_nltk (bool): Use nltk stopwords if available
    
    Returns:
        set: Set of stopwords
    """
    stopwords_set = set()
    
    # Get NLTK stopwords if available
    if use_nltk and nltk_available:
        try:
            stopwords_set.update(stopwords.words('english'))
        except LookupError:
            print("    ⚠ NLTK stopwords not found. Using default stopwords only.")
            stopwords_set.update(DEFAULT_STOPWORDS)
    else:
        stopwords_set.update(DEFAULT_STOPWORDS)
    
    # Add custom social media and domain-specific stopwords
    custom_stopwords = {
        # Twitter-specific
        'rt',           # Retweet
        'twitter',      # Twitter mention
        'amp',          # HTML entity &amp;
        'nbsp',         # Non-breaking space
        'gt', 'lt',     # HTML entities > <
        
        # Common abbreviations and month names
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'st', 'nd', 'rd', 'th',  # Date suffixes
        
        # Common internet terms
        'http', 'https', 'www', 'com', 'org', 'net',
        'url', 'link', 'click',
        
        # Single letters (handled separately too but add here)
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    }
    
    stopwords_set.update(custom_stopwords)
    return stopwords_set


def clean_text_for_analysis(text, stopwords_set=None, min_word_length=2, remove_numbers=True):
    """
    Advanced text cleaning for word frequency analysis.
    
    Args:
        text (str): Input text
        stopwords_set (set): Stopwords to remove (None = no removal)
        min_word_length (int): Minimum word length to keep (default: 2, removes single letters)
        remove_numbers (bool): Remove pure numbers (default: True)
    
    Returns:
        list: Cleaned word list
    """
    if not text or pd.isna(text):
        return []
    
    text = str(text).lower()
    
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', '', text)  # &amp; → '', &nbsp; → ''
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove mentions and hashtags (keep the word part)
    text = re.sub(r'@\w+', '', text)       # @username → ''
    text = re.sub(r'#', '', text)          # Remove # but keep word
    
    # Remove punctuation but keep internal apostrophes
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r"\s+", ' ', text)       # Normalize whitespace
    
    # Split into words
    words = text.split()
    
    # Filter words
    filtered_words = []
    for word in words:
        # Skip if contains apostrophe at start/end (contractions)
        word = word.strip("'")
        if not word:
            continue
        
        # Skip if too short
        if len(word) < min_word_length:
            continue
        
        # Skip if pure number
        if remove_numbers and word.isdigit():
            continue
        
        # Skip if in stopwords
        if stopwords_set and word in stopwords_set:
            continue
        
        filtered_words.append(word)
    
    return filtered_words


def detect_likely_handles(word_freq, threshold=0.7):
    """
    Detect likely Twitter handles or usernames based on frequency patterns
    
    Args:
        word_freq (Counter): Word frequency counter
        threshold (float): Likelihood threshold (0-1)
    
    Returns:
        dict: Suspected handles and confidence scores
    """
    suspected_handles = {}
    total_words = sum(word_freq.values())
    
    for word, freq in word_freq.items():
        confidence = 0.0
        reasons = []
        
        # Check 1: Unusually high concentration in single word
        word_percentage = freq / total_words
        if word_percentage > 0.07:  # More than 7% is suspicious
            confidence += 0.4
            reasons.append(f"High concentration ({word_percentage*100:.1f}%)")
        
        # Check 2: Looks like a username (lowercase, no spaces, contains letters/numbers)
        if re.match(r'^[a-z0-9_]+$', word) and len(word) < 20:
            confidence += 0.3
            reasons.append("Username-like pattern")
        
        # Check 3: Contains common Twitter handle patterns
        if 'jin' in word.lower() or 'user' in word.lower():
            confidence += 0.2
            reasons.append("Contains known handle patterns")
        
        if confidence >= threshold:
            suspected_handles[word] = {
                'confidence': confidence,
                'frequency': freq,
                'percentage': word_percentage * 100,
                'reasons': reasons
            }
    
    return suspected_handles

def find_files_multiple_types(directory, extensions, recursive=True):
    """
    Find multiple file types in the specified directory.

    :param directory: Search directory
    :param extensions: List of file extensions, e.g. ['.txt', '.py']
    :param recursive: Whether to search subdirectories recursively
    :return: List of matched file paths
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    # Generate matching patterns
    patterns = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        if recursive:
            patterns.append(os.path.join(directory, '**', f'*{ext}'))
        else:
            patterns.append(os.path.join(directory, f'*{ext}'))

    # Use itertools.chain to merge multiple glob results
    files_iter = itertools.chain.from_iterable(
        glob.iglob(pattern, recursive=recursive) for pattern in patterns
    )

    # Deduplicate and return sorted results
    return sorted(set(files_iter))


def collate_fn_series(batch):
    """
    Custom collate function for processing pd.Series data
    
    Returns batch in dictionary format, which is the standard approach in modern NLP research.
    Each key corresponds to a feature column, and value is a list of all sample values in that column.
    
    Advantages:
    - Facilitates vectorization and GPU acceleration
    - Compatible with mainstream frameworks like HuggingFace and PyTorch
    - Enables batch operations on specific columns
    
    Args:
        batch: List containing pd.Series
        
    Returns:
        dict: Convert Series in batch to DataFrame, then to dictionary
    """
    # Convert pd.Series list to DataFrame
    df = pd.DataFrame(batch)
    
    # Convert DataFrame to dictionary, with each column as a key
    batch_dict = {}
    for col in df.columns:
        values = df[col].values
        # Try to convert to torch tensor (if numeric type)
        try:
            batch_dict[col] = torch.tensor(values, dtype=torch.float32)
        except:
            # If cannot convert to tensor, keep as numpy array or list
            batch_dict[col] = values
    
    return batch_dict


# Column name mapping dictionary
COLUMN_MAPPING = {
    # Content text mapping
    'Posting': 'content',
    'Postings': 'content',
    'Text': 'content',
    'Post': 'content',
    'Description of Posting': 'content',
    
    # Date mapping
    'Date (GMT)': 'date',
    'Date of Posting': 'date',
    
    # Avatar mapping
    'Avatar used': 'avatar_used',
    
    # Background mapping
    'Background/Header picture': 'background_used',
    
    # Image description mapping
    'Description of Image': 'image_description',
    
    # Format mapping
    'Format of Post': 'post_format',
    
    # Identity information mapping
    'Handle': 'handle',
    'Name': 'name',
    'Social media handle of user': 'handle',
    
    # Encoding status mapping
    'Coded': 'coded',
    'coded': 'coded',
}

# Standard column order
STANDARD_COLUMNS = [
    'content', 'date', 'avatar_used', 'background_used', 
    'image_description', 'post_format', 'handle', 'name', 'coded',
    'category', 'person', 'sheet', 'file_path'
]

class RadicalisationDataset(Dataset):
    """
    PyTorch Dataset class for loading data under "Fighter and sympathiser"
    Supports loading data from XLSX and XLSM files, and adds folder names and people names to each data row
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the Fighter and sympathiser directory
            transform (callable, optional): Data processing function
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data = []  # Store all row data
        self.column_info = {}  # Store column information for each file
        
        # Scan directory and read all Excel files
        self._load_all_data()
    
    def _load_all_data(self):
        """Scan directory tree, read all data from XLSX and XLSM files"""
        for category_dir in self.root_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            # Iterate through all subdirectories (person directories) under category
            for person_dir in category_dir.iterdir():
                if not person_dir.is_dir():
                    continue
                
                person_name = person_dir.name
                
                # Find all XLSX and XLSM files
                for file_path in person_dir.glob('*.xlsx'):
                    self._read_excel_file(file_path, category_name, person_name)
                
                for file_path in person_dir.glob('*.xlsm'):
                    self._read_excel_file(file_path, category_name, person_name)
        
        print(f"Total loaded {len(self.data)} rows")

    
    def _read_excel_file(self, file_path, category_name, person_name):
        """
        Read XLSX or XLSM file, collect all row data to self.data, and normalize column names
        
        Args:
            file_path (str): File path
            category_name (str): Folder name
            person_name (str): Person name
        """
        try:
            xls = pd.ExcelFile(file_path)
            file_key = f"{category_name}/{person_name}/{file_path.name}"
            self.column_info[file_key] = {}
            
            # Read all Sheets in this file
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Record original column names for this sheet
                self.column_info[file_key][sheet_name] = list(df.columns)
                
                # Rename columns based on mapping
                rename_dict = {}
                for original_col in df.columns:
                    if original_col in COLUMN_MAPPING:
                        rename_dict[original_col] = COLUMN_MAPPING[original_col]
                
                df = df.rename(columns=rename_dict)
                
                # Ensure all standard columns exist (fill missing with NaN)
                for std_col in STANDARD_COLUMNS:
                    if std_col not in df.columns:
                        df[std_col] = np.nan
                
                # Add metadata columns
                df['category'] = category_name
                df['person'] = person_name
                df['sheet'] = sheet_name
                df['file_path'] = str(file_path)
                
                # Keep only standard columns and metadata columns, in specified order
                df = df[STANDARD_COLUMNS]
                
                # Add all rows from this sheet to self.data
                for _, row in df.iterrows():
                    self.data.append(row)
            
            print(f"Successfully read file: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            pd.Series: Single row data with standardized columns and metadata
        """
        row = self.data[idx]
        
        if self.transform:
            row = self.transform(row)
        
        return row
    
    def get_statistics(self):
        """Get data statistics"""
        print("\n=== Data Statistics ===")
        print(f"Total rows: {len(self)}")
        
        if len(self.data) > 0:
            # Get unique values and counts for category column
            categories = [row.get('category', 'Unknown') for row in self.data]
            from collections import Counter
            category_counts = Counter(categories)
            
            print("\nRows per category:")
            for category, count in sorted(category_counts.items()):
                print(f"  {category}: {count} rows")
    
    def analyze_suspicious_words(self, words_list=None, context_length=50):
        """
        Analyze suspicious high-frequency words (likely handles, mentions, or artifacts)
        
        Args:
            words_list (list): Words to analyze (if None, analyzes top 20 words)
            context_length (int): Number of characters before/after to show
        """
        if words_list is None:
            words_list = ['jinny', 'itsljinny']
        
        print("\n" + "="*80)
        print("Suspicious Words Context Analysis")
        print("="*80)
        
        for word in words_list:
            print(f"\n{'='*80}")
            print(f"Word: '{word}'")
            print(f"{'='*80}")
            
            # Find all occurrences
            occurrences = []
            category_counts = {}
            person_counts = {}
            
            for idx, row in enumerate(self.data):
                content = str(row.get('content', '')).lower()
                if word.lower() in content:
                    occurrences.append({
                        'idx': idx,
                        'content': row.get('content', ''),
                        'category': row.get('category', 'Unknown'),
                        'person': row.get('person', 'Unknown'),
                        'date': row.get('date', 'Unknown')
                    })
                    
                    # Count by category and person
                    cat = row.get('category', 'Unknown')
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    
                    pers = row.get('person', 'Unknown')
                    person_counts[pers] = person_counts.get(pers, 0) + 1
            
            # Print statistics
            print(f"\nTotal occurrences: {len(occurrences)}")
            print(f"\nOccurrences by category:")
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")
            
            print(f"\nTop 10 people mentioning/using '{word}':")
            for pers, count in sorted(person_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {pers}: {count}")
            
            # Show sample contexts
            print(f"\nSample contexts (first 5):")
            for i, occ in enumerate(occurrences[:5]):
                content = str(occ['content'])
                idx_in_content = content.lower().find(word.lower())
                
                # Extract context around the word
                start = max(0, idx_in_content - context_length)
                end = min(len(content), idx_in_content + len(word) + context_length)
                context = content[start:end]
                
                print(f"\n  Example {i+1}:")
                print(f"    Person: {occ['person']}")
                print(f"    Category: {occ['category']}")
                print(f"    Date: {occ['date']}")
                print(f"    Context: ...{context}...")
        
        print("\n" + "="*80)
        
        if not self.column_info:
            print("No column information found. Make sure data is loaded.")
            return
        
        for file_key, sheets_info in sorted(self.column_info.items()):
            print(f"\n[File]: {file_key}")
            for sheet_name, columns in sheets_info.items():
                print(f"  └─ [Sheet]: {sheet_name}")
                print(f"     Column count: {len(columns)}")
                print(f"     Column names: {columns}")
        
        print("\n" + "="*80)
    
    def analyze_content_statistics(self, top_n=20, save_dir='./analysis', remove_stopwords=True, 
                                     remove_numbers=True, min_word_length=2):
        """
        Analyze content column statistics, including word frequency distribution.
        
        Args:
            top_n (int): Number of top frequent words to display
            save_dir (str): Directory to save charts
            remove_stopwords (bool): Whether to remove common stopwords (default: True)
            remove_numbers (bool): Whether to remove pure numbers (default: True)
            min_word_length (int): Minimum word length to keep (default: 2, removes single letters)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect all content data
        contents = [row.get('content', '') for row in self.data if pd.notna(row.get('content'))]
        
        print("\n" + "="*80)
        print("Content Statistics (Advanced Cleaning)")
        print("="*80)
        print(f"Total samples: {len(self.data)}")
        print(f"Valid content count: {len(contents)}")
        print(f"Empty content count: {len(self.data) - len(contents)}")
        print(f"\nCleaning options:")
        print(f"  Remove stopwords: {'Yes' if remove_stopwords else 'No'}")
        print(f"  Remove numbers: {'Yes' if remove_numbers else 'No'}")
        print(f"  Min word length: {min_word_length} characters")
        
        if len(contents) == 0:
            print("No valid content data")
            return
        
        # Get stopwords if needed
        stopwords_set = get_stopwords() if remove_stopwords else None
        
        # Calculate text statistics
        content_lengths = [len(str(c).split()) for c in contents]
        print(f"\nText length statistics (unit: words):")
        print(f"  Max: {max(content_lengths)} words")
        print(f"  Min: {min(content_lengths)} words")
        print(f"  Average: {np.mean(content_lengths):.2f} words")
        print(f"  Median: {np.median(content_lengths):.2f} words")
        
        # Tokenize and count word frequency with advanced cleaning
        all_words = []
        for content in contents:
            # Use advanced cleaning function
            words = clean_text_for_analysis(
                content, 
                stopwords_set=stopwords_set,
                min_word_length=min_word_length,
                remove_numbers=remove_numbers
            )
            all_words.extend(words)
        
        # Calculate word frequency
        word_freq = Counter(all_words)
        most_common_words = word_freq.most_common(top_n)
        
        print(f"\nHigh-frequency word statistics (top {top_n} words):")
        print(f"  Total words (before dedup): {len(all_words)}")
        print(f"  Vocabulary size (after dedup): {len(word_freq)}")
        if remove_stopwords:
            print(f"  Stopwords removed: ~{len(stopwords_set)} common words")
        print(f"\nRank | Word | Frequency")
        print("  " + "-" * 30)
        for rank, (word, freq) in enumerate(most_common_words, 1):
            print(f"  {rank:2d}  | {word:15s} | {freq:5d}")
        
        # Plot charts
        self._plot_word_frequency(most_common_words, save_dir, top_n, remove_stopwords)
        self._plot_text_length_distribution(content_lengths, save_dir)
        self._plot_wordcloud(all_words, save_dir)
        self._plot_category_distribution(save_dir)
        
        # Generate comparison data for frequency analysis
        self._generate_frequency_comparison(most_common_words, save_dir)
        
        print("\n✓ Analysis complete, charts saved to " + save_dir)
        print("="*80)
    
    def _plot_word_frequency(self, most_common_words, save_dir, top_n, remove_stopwords=True):
        """Plot word frequency bar chart for top N words"""
        words, freqs = zip(*most_common_words)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(len(words)), freqs, color='steelblue', alpha=0.8)
        ax.set_xlabel('Word', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        title_suffix = " (Stopwords removed)" if remove_stopwords else ""
        ax.set_title(f'Word Frequency Distribution (Top {top_n} words){title_suffix}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/word_frequency.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Word frequency chart saved: {save_dir}/word_frequency.png")
        plt.close()
    
    def _plot_text_length_distribution(self, content_lengths, save_dir):
        """Plot text length distribution histogram"""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(content_lengths, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Text Length (words)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Text Length Distribution', fontsize=14, fontweight='bold')
        ax.axvline(np.mean(content_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(content_lengths):.2f}')
        ax.axvline(np.median(content_lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(content_lengths):.2f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/text_length_distribution.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Text length distribution chart saved: {save_dir}/text_length_distribution.png")
        plt.close()
    
    def _plot_wordcloud(self, all_words, save_dir):
        """
        Plot word cloud with font detection and error handling
        
        Note: WordCloud visualization may not be perfectly linear with word frequency
        due to font size limits and layout optimization. Use the bar chart for accurate frequency analysis.
        """
        try:
            # Convert word list to text string
            text = ' '.join(all_words)
            
            # Generate word cloud with adjusted parameters for better frequency representation
            kwargs = {
                'width': 1400, 
                'height': 700,
                'background_color': 'white',
                'colormap': 'viridis',
                'max_words': 100,  # Reduced from 200 to better show top words
                'relative_scaling': 0.5,  # Controls frequency impact on size (0-1, lower = more linear)
                'min_font_size': 10,  # Minimum font size
            }
            
            # Add font_path if Chinese font found
            # if CHINESE_FONT_PATH:
            #     kwargs['font_path'] = CHINESE_FONT_PATH
            #     wordcloud = WordCloud(**kwargs).generate(text)
            # else:
            #     print("    ⚠ No Chinese font found, word cloud may display as boxes")
            #     wordcloud = WordCloud(**kwargs).generate(text)
            wordcloud = WordCloud(**kwargs).generate(text)
            
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud (Note: Frequency visualization, not perfectly linear)', 
                        fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
            print(f"  ✓ Word cloud saved: {save_dir}/wordcloud.png")
            plt.close()
        except Exception as e:
            print(f"  ✗ Word cloud generation failed: {e}")
    
    def _plot_category_distribution(self, save_dir):
        """Plot sample distribution across categories"""
        categories = [row.get('category', 'Unknown') for row in self.data]
        category_counts = Counter(categories)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        cats = list(category_counts.keys())
        counts = list(category_counts.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
        ax.bar(range(len(cats)), counts, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Sample Distribution by Category', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Display values on bars
        for i, (cat, count) in enumerate(zip(cats, counts)):
            ax.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/category_distribution.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Category distribution chart saved: {save_dir}/category_distribution.png")
        plt.close()
    
    
    def analyze_with_dual_reports(self, top_n=30, save_dir='./analysis_dual', remove_stopwords=True, 
                                   remove_numbers=True, min_word_length=2):
        """
        Generate dual reports: one with all words (including handles), one with handles excluded
        
        Args:
            top_n (int): Number of top frequent words to display
            save_dir (str): Directory to save charts
            remove_stopwords (bool): Whether to remove common stopwords
            remove_numbers (bool): Whether to remove pure numbers
            min_word_length (int): Minimum word length to keep
        """
        import os
        
        # Create main directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subdirectories for each report
        report_with_handles = os.path.join(save_dir, '01_With_Handles')
        report_without_handles = os.path.join(save_dir, '02_Without_Handles')
        os.makedirs(report_with_handles, exist_ok=True)
        os.makedirs(report_without_handles, exist_ok=True)
        
        # Collect all content data
        contents = [row.get('content', '') for row in self.data if pd.notna(row.get('content'))]
        
        print("\n" + "="*80)
        print("DUAL REPORT ANALYSIS: With Handles vs Without Handles")
        print("="*80)
        print(f"Total samples: {len(self.data)}")
        print(f"Valid content count: {len(contents)}")
        
        # Get stopwords if needed
        stopwords_set = get_stopwords() if remove_stopwords else None
        
        # Calculate text statistics (shared for both reports)
        content_lengths = [len(str(c).split()) for c in contents]
        
        # ==== REPORT 1: WITH HANDLES (原始数据) ====
        print(f"\n{'='*80}")
        print("REPORT 1: WITH HANDLES (All words including usernames)")
        print("="*80)
        
        all_words_with_handles = []
        for content in contents:
            words = clean_text_for_analysis(
                content, 
                stopwords_set=stopwords_set,
                min_word_length=min_word_length,
                remove_numbers=remove_numbers
            )
            all_words_with_handles.extend(words)
        
        word_freq_with = Counter(all_words_with_handles)
        most_common_with = word_freq_with.most_common(top_n)
        
        print(f"Total words: {len(all_words_with_handles)}")
        print(f"Vocabulary size: {len(word_freq_with)}")
        print(f"\nRank | Word | Frequency | Percentage")
        print("  " + "-" * 45)
        for rank, (word, freq) in enumerate(most_common_with, 1):
            pct = (freq / len(all_words_with_handles)) * 100
            print(f"  {rank:2d}  | {word:15s} | {freq:5d}    | {pct:5.2f}%")
        
        # Generate charts for Report 1
        self._plot_word_frequency(most_common_with, report_with_handles, top_n, remove_stopwords)
        self._plot_text_length_distribution(content_lengths, report_with_handles)
        self._plot_wordcloud(all_words_with_handles, report_with_handles)
        self._plot_category_distribution(report_with_handles)
        self._generate_frequency_comparison(most_common_with, report_with_handles)
        
        # Detect likely handles
        suspected_handles = detect_likely_handles(word_freq_with, threshold=0.5)
        
        print(f"\n⚠️  Suspected Handles Detected: {len(suspected_handles)}")
        for handle, info in sorted(suspected_handles.items(), key=lambda x: -x[1]['frequency'])[:10]:
            print(f"    '{handle}': {info['frequency']} times ({info['percentage']:.2f}%)")
        
        # ==== REPORT 2: WITHOUT HANDLES (清洁数据) ====
        print(f"\n{'='*80}")
        print("REPORT 2: WITHOUT HANDLES (Content words only)")
        print("="*80)
        
        # Create stopwords that includes suspected handles
        stopwords_with_handles = stopwords_set.copy() if stopwords_set else set()
        stopwords_with_handles.update(suspected_handles.keys())
        
        all_words_without_handles = []
        for content in contents:
            words = clean_text_for_analysis(
                content, 
                stopwords_set=stopwords_with_handles,
                min_word_length=min_word_length,
                remove_numbers=remove_numbers
            )
            all_words_without_handles.extend(words)
        
        word_freq_without = Counter(all_words_without_handles)
        most_common_without = word_freq_without.most_common(top_n)
        
        print(f"Total words: {len(all_words_without_handles)}")
        print(f"Vocabulary size: {len(word_freq_without)}")
        print(f"Words removed (handles): {len(all_words_with_handles) - len(all_words_without_handles)}")
        print(f"Reduction: {((len(all_words_with_handles) - len(all_words_without_handles)) / len(all_words_with_handles) * 100):.1f}%")
        print(f"\nRank | Word | Frequency | Percentage")
        print("  " + "-" * 45)
        for rank, (word, freq) in enumerate(most_common_without, 1):
            pct = (freq / len(all_words_without_handles)) * 100
            print(f"  {rank:2d}  | {word:15s} | {freq:5d}    | {pct:5.2f}%")
        
        # Generate charts for Report 2
        self._plot_word_frequency(most_common_without, report_without_handles, top_n, remove_stopwords)
        self._plot_text_length_distribution(content_lengths, report_without_handles)
        self._plot_wordcloud(all_words_without_handles, report_without_handles)
        self._plot_category_distribution(report_without_handles)
        self._generate_frequency_comparison(most_common_without, report_without_handles)
        
        # ==== 对比分析 ====
        print(f"\n{'='*80}")
        print("COMPARISON ANALYSIS")
        print("="*80)
        
        print("\nTop 10 Words Comparison:")
        print(f"{'Rank':<5} {'Report 1 (With)':<25} {'Report 2 (Without)':<25}")
        print("-" * 55)
        for i in range(min(10, len(most_common_with), len(most_common_without))):
            w1, f1 = most_common_with[i] if i < len(most_common_with) else ('N/A', 0)
            w2, f2 = most_common_without[i] if i < len(most_common_without) else ('N/A', 0)
            
            if w1 in suspected_handles:
                w1_str = f"{w1}* ({f1})"  # Mark as handle
            else:
                w1_str = f"{w1} ({f1})"
            
            w2_str = f"{w2} ({f2})"
            
            print(f"{i+1:<5} {w1_str:<25} {w2_str:<25}")
        
        print("\n* = Suspected handle/username")
        
        # 生成对比总结报告
        self._generate_comparison_summary(
            suspected_handles, 
            most_common_with, 
            most_common_without,
            save_dir
        )
        
        print(f"\n✓ Dual analysis complete!")
        print(f"  Report 1 (with handles): {report_with_handles}/")
        print(f"  Report 2 (without handles): {report_without_handles}/")
        print(f"  Summary: {save_dir}/comparison_summary.txt")
        print("="*80)
    
    def _generate_comparison_summary(self, suspected_handles, most_common_with, most_common_without, save_dir):
        """Generate a detailed comparison summary report"""
        import os
        
        summary_path = os.path.join(save_dir, 'comparison_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DUAL REPORT ANALYSIS SUMMARY\n")
            f.write("Comparing: With Handles vs Without Handles\n")
            f.write("="*80 + "\n\n")
            
            # 1. Detected Handles
            f.write("1. SUSPECTED HANDLES/USERNAMES DETECTED\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total: {len(suspected_handles)}\n\n")
            for handle, info in sorted(suspected_handles.items(), key=lambda x: -x[1]['frequency']):
                f.write(f"  '{handle}':\n")
                f.write(f"    Frequency: {info['frequency']} times ({info['percentage']:.2f}% of total)\n")
                f.write(f"    Confidence: {info['confidence']:.0%}\n")
                f.write(f"    Reasons: {', '.join(info['reasons'])}\n")
                f.write(f"\n")
            
            # 2. Top Words Comparison
            f.write("\n" + "="*80 + "\n")
            f.write("2. TOP 30 WORDS COMPARISON\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("REPORT 1: WITH HANDLES\n")
            f.write("Rank | Word              | Frequency | Percentage\n")
            f.write("-" * 50 + "\n")
            for rank, (word, freq) in enumerate(most_common_with[:30], 1):
                pct = (freq / sum(f for _, f in most_common_with)) * 100
                is_handle = " ←HANDLE" if word in suspected_handles else ""
                f.write(f"{rank:3d}  | {word:17s} | {freq:7d}  | {pct:6.2f}%{is_handle}\n")
            
            f.write("\n\nREPORT 2: WITHOUT HANDLES (CLEANED)\n")
            f.write("Rank | Word              | Frequency | Percentage\n")
            f.write("-" * 50 + "\n")
            for rank, (word, freq) in enumerate(most_common_without[:30], 1):
                pct = (freq / sum(f for _, f in most_common_without)) * 100
                f.write(f"{rank:3d}  | {word:17s} | {freq:7d}  | {pct:6.2f}%\n")
            
            # 3. Key Insights
            f.write("\n" + "="*80 + "\n")
            f.write("3. KEY INSIGHTS\n")
            f.write("-" * 80 + "\n\n")
            
            total_words_with = sum(f for _, f in most_common_with)
            total_words_without = sum(f for _, f in most_common_without)
            removed_words = total_words_with - total_words_without
            
            f.write(f"Words removed due to handles: {removed_words}\n")
            f.write(f"Percentage of total: {(removed_words/total_words_with*100):.1f}%\n\n")
            
            f.write("Impact on Top 5 Words:\n")
            for i in range(min(5, len(most_common_with))):
                word_with, freq_with = most_common_with[i]
                # Find this word in without list
                rank_without = next((j+1 for j, (w, _) in enumerate(most_common_without) if w == word_with), None)
                
                if rank_without:
                    f.write(f"  {word_with}: Rank {i+1} (with) → Rank {rank_without} (without)\n")
                else:
                    f.write(f"  {word_with}: Rank {i+1} (with) → Not in top {len(most_common_without)} (without)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("-" * 80 + "\n\n")
            f.write("For academic research and serious analysis, use REPORT 2 (Without Handles).\n")
            f.write("Report 2 provides cleaner content word analysis, free from user identifiers.\n")
            f.write("Use Report 1 only for completeness or when studying handle patterns.\n")
            f.write("\n" + "="*80 + "\n")
        
        print(f"  ✓ Comparison summary saved: {summary_path}")
    
    def _generate_frequency_comparison(self, most_common_words, save_dir):
        """
        Generate frequency comparison chart to diagnose WordCloud representation accuracy
        
        This helps verify that the WordCloud visual size actually matches the word frequency data.
        """
        import os
        
        # Top 20 words
        top_20 = most_common_words[:20]
        words, freqs = zip(*top_20)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Left: Absolute frequency
        ax1.barh(range(len(words)), freqs, color='steelblue', alpha=0.8)
        ax1.set_yticks(range(len(words)))
        ax1.set_yticklabels(words, fontsize=11)
        ax1.set_xlabel('Frequency (Count)', fontsize=12, fontweight='bold')
        ax1.set_title('Top 20 Words by Frequency (Absolute)', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add frequency values on bars
        for i, (word, freq) in enumerate(top_20):
            ax1.text(freq + max(freqs)*0.01, i, str(freq), va='center', fontweight='bold', fontsize=10)
        
        # Right: Normalized frequency (percentage)
        total_freq = sum(freqs)
        percentages = [f / total_freq * 100 for f in freqs]
        colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(words)))
        ax2.barh(range(len(words)), percentages, color=colors_gradient, alpha=0.8)
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels(words, fontsize=11)
        ax2.set_xlabel('Frequency (Percentage %)', fontsize=12, fontweight='bold')
        ax2.set_title('Top 20 Words by Frequency (Percentage)', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add percentage values on bars
        for i, (word, pct) in enumerate(zip(words, percentages)):
            ax2.text(pct + max(percentages)*0.01, i, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/frequency_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Frequency comparison chart saved: {save_dir}/frequency_comparison.png")
        plt.close()
        
        # Also save detailed CSV for inspection
        csv_path = os.path.join(save_dir, 'top_words_frequency.csv')
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Word', 'Frequency', 'Percentage %'])
            for rank, (word, freq) in enumerate(most_common_words[:50], 1):
                pct = (freq / sum(freq for _, freq in most_common_words)) * 100
                writer.writerow([rank, word, freq, f'{pct:.2f}'])
        
        print(f"  ✓ Top 50 words frequency data saved: {csv_path}")
    
    def extract_coded_samples(self, output_path='./coded_samples.csv'):
        """
        提取所有 coded=1 的样本并保存到 CSV 文件
        
        Args:
            output_path (str): 输出CSV文件的路径，默认为 './coded_samples.csv'
        
        Returns:
            pd.DataFrame: 包含所有 coded=1 样本的DataFrame
        """
        # 将所有数据转换为DataFrame
        df = pd.DataFrame(self.data)
        
        # 过滤 coded=1 的样本
        coded_samples = df[df['coded'] == 1].copy()
        
        # 保存到CSV
        coded_samples.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n✓ 已提取 coded=1 的样本")
        print(f"  总数: {len(coded_samples)} 条记录")
        print(f"  保存位置: {output_path}")
        print(f"  列数: {len(coded_samples.columns)}")
        print(f"  包含列: {', '.join(coded_samples.columns.tolist())}")
        
        return coded_samples
    
    def extract_non_coded_samples(self, output_path='./non_coded_samples.csv'):
        """
        提取所有 coded=0 的样本并保存到 CSV 文件
        
        Args:
            output_path (str): 输出CSV文件的路径，默认为 './non_coded_samples.csv'
        
        Returns:
            pd.DataFrame: 包含所有 coded=0 样本的DataFrame
        """
        # 将所有数据转换为DataFrame
        df = pd.DataFrame(self.data)
        
        # 过滤 coded=0 的样本
        non_coded_samples = df[df['coded'] == 0].copy()
        
        # 保存到CSV
        non_coded_samples.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n✓ 已提取 coded=0 的样本")
        print(f"  总数: {len(non_coded_samples)} 条记录")
        print(f"  保存位置: {output_path}")
        print(f"  列数: {len(non_coded_samples.columns)}")
        print(f"  包含列: {', '.join(non_coded_samples.columns.tolist())}")
        
        return non_coded_samples
    
    def extract_all_coded_status(self, coded_output='./coded_samples.csv', non_coded_output='./non_coded_samples.csv'):
        """
        同时提取并保存 coded=1 和 coded=0 的所有样本
        
        Args:
            coded_output (str): coded=1 样本的输出路径
            non_coded_output (str): coded=0 样本的输出路径
        
        Returns:
            tuple: (coded_samples_df, non_coded_samples_df)
        """
        print("\n" + "="*80)
        print("EXTRACTING SAMPLES BY CODED STATUS")
        print("="*80)
        
        coded_df = self.extract_coded_samples(coded_output)
        non_coded_df = self.extract_non_coded_samples(non_coded_output)
        
        print(f"\n✓ 提取完成")
        print(f"  Coded=1 样本: {len(coded_df)} 条 ({len(coded_df)/len(self)*100:.1f}%)")
        print(f"  Coded=0 样本: {len(non_coded_df)} 条 ({len(non_coded_df)/len(self)*100:.1f}%)")
        print(f"  总计: {len(self)} 条记录")
        
        return coded_df, non_coded_df


def create_dataloaders(dataset, batch_size=4, num_workers=0, train_ratio=0.8):
    """
    Create training and validation DataLoaders
    
    Args:
        dataset (RadicalisationDataset): Loaded dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker processes for data loading
        train_ratio (float): Training set ratio (0-1)
    
    Returns:
        tuple: (train_loader, val_loader, dataset)
    """
    
    # Split dataset into training and validation sets
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # Create DataLoaders with custom collate_fn for pd.Series
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_series,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_series,
    )
    
    return train_loader, val_loader, dataset


# Usage Example
def usage_example():
    root_dir = r"c:\Users\shanghong.li\Desktop\AI for radicalisation\Fighter and sympathiser"
    
    # Create Dataset
    print("Loading data...")
    dataset = RadicalisationDataset(root_dir)
    
    # Get data statistics
    dataset.get_statistics()
    
    # ========== EXTRACT CODED AND NON-CODED SAMPLES ==========
    print("\n" + "="*80)
    print("EXTRACTING CODED AND NON-CODED SAMPLES")
    print("="*80)
    
    # 提取所有coded=1和coded=0的样本并保存到CSV
    coded_df, non_coded_df = dataset.extract_all_coded_status(
        coded_output='./coded_samples.csv',
        non_coded_output='./non_coded_samples.csv'
    )
    
    # Analyze suspicious words (handles, mentions, etc.)
    print("\n" + "="*80)
    print("Analyzing High-Frequency Words Context...")
    print("="*80)
    dataset.analyze_suspicious_words(words_list=['jinny', 'itsljinny', 'tabanacle'])
    
    # ========== DUAL REPORT ANALYSIS (选项2: 分别报告) ==========
    print("\n" + "="*100)
    print("GENERATING DUAL REPORTS: With Handles vs Without Handles")
    print("This will help you compare content analysis with and without username/handle filtering")
    print("="*100)
    
    dataset.analyze_with_dual_reports(
        top_n=30,
        save_dir='./analysis_dual_reports',
        remove_stopwords=True,
        remove_numbers=True,
        min_word_length=2
    )
    
    # Original single report (for reference)
    print("\nPerforming single NLP data analysis with advanced text cleaning (for reference)...")
    dataset.analyze_content_statistics(
        top_n=30, 
        save_dir='./analysis_results',
        remove_stopwords=True,      # Remove common words (the, a, and, etc.)
        remove_numbers=True,         # Remove pure numbers (0000, 123, etc.)
        min_word_length=2            # Remove single letters (l, r, etc.)
    )
    
    # Display standardized data examples
    print("\n=== Standardized Data Example ===")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nColumns in first data point:")
        print(f"  {list(sample.index)}")
        print(f"\nFirst data point content preview:")
        for col in ['content', 'coded', 'category', 'person', 'date']:
            value = sample.get(col, 'N/A')
            if pd.notna(value):
                value_str = str(value)[:80] if col == 'content' else str(value)
                print(f"  {col}: {value_str}")
    
    print("\n=== Creating DataLoaders ===")
    train_loader, val_loader, final_dataset = create_dataloaders(
        dataset,
        batch_size=4,
        num_workers=0,
        train_ratio=0.8
    )
    
    print(f"\nTraining set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Total data: {len(dataset)}")
    
    # View one batch
    print("\n=== First Batch Content ===")
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch size: {len(batch[list(batch.keys())[0]])}")
        print(f"Number of columns: {len(batch)}")
        print(f"Columns in batch: {list(batch.keys())}")
        break
    
    print("\n✓ All tasks completed")

if __name__ == "__main__":
    root_dir = r"/home/user/workspace/SHLi/AI for radicalisation/Fighter and sympathiser"
    
    # Create Dataset
    print("Loading data...")

    dataset = RadicalisationDataset(root_dir)
    # 同时提取coded和non-coded样本
    coded_df, non_coded_df = dataset.extract_all_coded_status(
        coded_output='./Fighter and sympathiser/coded_samples.csv',
        non_coded_output='./Fighter and sympathiser/non_coded_samples.csv'
    )
