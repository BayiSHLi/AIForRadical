"""
Analyze generated samples by radicality level.
Generates word frequency analysis, word clouds, and statistical reports.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np


class RadicalityAnalyzer:
    def __init__(self, input_file, output_dir="analysis_radicality"):
        self.input_file = input_file
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.radicality_data = {
            "Neutral": [],
            "Low": [],
            "Medium": [],
            "High": []
        }
        self.radicality_order = ["Neutral", "Low", "Medium", "High"]
        
    def load_samples(self):
        """Load samples from JSONL file and group by radicality."""
        print(f"Loading samples from {self.input_file}...")
        count = 0
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    radicality = sample.get('Radicality', 'Unknown')
                    if radicality in self.radicality_data:
                        self.radicality_data[radicality].append(sample)
                        count += 1
        
        print(f"Loaded {count} samples")
        for rad, samples in self.radicality_data.items():
            print(f"  {rad}: {len(samples)} samples")
        return count
    
    def tokenize_text(self, text):
        """Simple tokenization: split by non-alphanumeric characters."""
        text = text.lower()
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        # Split into words
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def get_stopwords(self):
        """Return common English stopwords."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'it', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'our', 'their',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'as', 'if', 'just',
            'so', 'than', 'too', 'very', 'more', 'most', 'some', 'any', 'all', 'each',
            'every', 'both', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'then', 'there', 'now', 'also', 'like', 'about', 'up', 'out', 'down',
            'off', 'into', 'through', 'during', 'before', 'after', 'between', 'under',
            'again', 'further', 'once', 'here', 'been', 'being', 'having', 'it', 'em'
        }
    
    def calculate_word_frequencies(self, radicality):
        """Calculate word frequencies for a radicality level."""
        samples = self.radicality_data[radicality]
        if not samples:
            return {}
        
        all_tokens = []
        stopwords = self.get_stopwords()
        
        for sample in samples:
            content = sample.get('Content', '')
            tokens = self.tokenize_text(content)
            # Filter out stopwords and short tokens
            tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
            all_tokens.extend(tokens)
        
        # Count frequencies
        freq_counter = Counter(all_tokens)
        return freq_counter
    
    def generate_wordcloud(self, radicality, frequencies):
        """Generate word cloud for a radicality level."""
        if not frequencies:
            print(f"No data for {radicality}")
            return
        
        # Convert counter to dict for wordcloud
        freq_dict = dict(frequencies.most_common(200))
        
        if not freq_dict:
            print(f"Insufficient data for wordcloud: {radicality}")
            return
        
        plt.figure(figsize=(14, 8))
        wordcloud = WordCloud(
            width=1400,
            height=800,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10,
            max_words=100
        ).generate_from_frequencies(freq_dict)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - Radicality: {radicality}', fontsize=16, fontweight='bold')
        
        output_path = os.path.join(self.output_dir, f'wordcloud_{radicality.lower()}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved wordcloud: {output_path}")
    
    def save_word_frequency_csv(self, radicality, frequencies, top_n=100):
        """Save top N word frequencies to CSV."""
        if not frequencies:
            return
        
        top_words = frequencies.most_common(top_n)
        df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'Word', 'Frequency']]
        
        output_path = os.path.join(self.output_dir, f'word_frequency_{radicality.lower()}.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved word frequency: {output_path}")
        return df
    
    def generate_radicality_comparison_report(self):
        """Generate comparative statistics across radicality levels."""
        report_data = []
        
        for radicality in self.radicality_order:
            samples = self.radicality_data[radicality]
            if not samples:
                continue
            
            # Calculate statistics
            contents = [s.get('Content', '') for s in samples]
            word_counts = [len(self.tokenize_text(c)) for c in contents]
            char_counts = [len(c) for c in contents]
            
            unique_indicators = len(set(s.get('indicator') for s in samples))
            
            report_data.append({
                'Radicality': radicality,
                'Sample_Count': len(samples),
                'Unique_Indicators': unique_indicators,
                'Avg_Word_Count': round(np.mean(word_counts), 2),
                'Avg_Char_Count': round(np.mean(char_counts), 2),
                'Max_Word_Count': max(word_counts),
                'Min_Word_Count': min(word_counts),
                'Max_Char_Count': max(char_counts),
                'Min_Char_Count': min(char_counts),
                'Median_Word_Count': round(np.median(word_counts), 2),
                'Std_Word_Count': round(np.std(word_counts), 2),
            })
        
        df_report = pd.DataFrame(report_data)
        output_path = os.path.join(self.output_dir, 'radicality_statistics.csv')
        df_report.to_csv(output_path, index=False)
        print(f"Saved radicality statistics: {output_path}")
        return df_report
    
    def generate_distribution_chart(self):
        """Generate distribution chart of samples across radicality levels."""
        counts = [len(self.radicality_data[rad]) for rad in self.radicality_order]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        bars = ax.bar(self.radicality_order, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_xlabel('Radicality Level', fontsize=12, fontweight='bold')
        ax.set_title('Sample Distribution by Radicality Level', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        output_path = os.path.join(self.output_dir, 'radicality_distribution.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved distribution chart: {output_path}")
    
    def generate_frequency_comparison(self):
        """Generate comparison chart of top words across radicality levels."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, radicality in enumerate(self.radicality_order):
            frequencies = self.calculate_word_frequencies(radicality)
            top_words = frequencies.most_common(15)
            
            if top_words:
                words, counts = zip(*top_words)
                ax = axes[idx]
                bars = ax.barh(words, counts, color=plt.cm.viridis(idx / len(self.radicality_order)))
                ax.set_xlabel('Frequency', fontweight='bold')
                ax.set_title(f'Top 15 Words - {radicality}', fontweight='bold', fontsize=12)
                ax.invert_yaxis()
                
                # Add value labels
                for i, (word, count) in enumerate(top_words):
                    ax.text(count, i, f' {count}', va='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'word_frequency_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved frequency comparison: {output_path}")
    
    def generate_text_length_comparison(self):
        """Generate text length comparison across radicality levels."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        word_lengths = []
        char_lengths = []
        
        for radicality in self.radicality_order:
            samples = self.radicality_data[radicality]
            contents = [s.get('Content', '') for s in samples]
            word_len = [len(self.tokenize_text(c)) for c in contents]
            char_len = [len(c) for c in contents]
            word_lengths.append(word_len)
            char_lengths.append(char_len)
        
        # Word count distribution
        ax1.boxplot(word_lengths, labels=self.radicality_order)
        ax1.set_ylabel('Word Count', fontweight='bold')
        ax1.set_title('Word Count Distribution by Radicality', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Character count distribution
        ax2.boxplot(char_lengths, labels=self.radicality_order)
        ax2.set_ylabel('Character Count', fontweight='bold')
        ax2.set_title('Character Count Distribution by Radicality', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'text_length_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved text length comparison: {output_path}")
    
    def generate_summary_report(self):
        """Generate comprehensive text summary report."""
        report_lines = [
            "="*80,
            "RADICALITY LEVEL ANALYSIS REPORT",
            "="*80,
            f"Generated Samples: {sum(len(self.radicality_data[rad]) for rad in self.radicality_order)}",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for radicality in self.radicality_order:
            samples = self.radicality_data[radicality]
            if not samples:
                continue
            
            contents = [s.get('Content', '') for s in samples]
            word_counts = [len(self.tokenize_text(c)) for c in contents]
            char_counts = [len(c) for c in contents]
            
            frequencies = self.calculate_word_frequencies(radicality)
            top_10 = frequencies.most_common(10)
            
            report_lines.extend([
                "-"*80,
                f"RADICALITY: {radicality.upper()}",
                "-"*80,
                f"Sample Count: {len(samples)}",
                f"Unique Indicators: {len(set(s.get('indicator') for s in samples))}",
                f"Average Word Count: {np.mean(word_counts):.2f}",
                f"Average Character Count: {np.mean(char_counts):.2f}",
                f"Word Count Range: {min(word_counts)} - {max(word_counts)}",
                f"Character Count Range: {min(char_counts)} - {max(char_counts)}",
                "",
                "Top 10 Words:",
            ])
            
            for rank, (word, freq) in enumerate(top_10, 1):
                report_lines.append(f"  {rank:2d}. {word:20s} - {freq:5d} occurrences")
            
            report_lines.append("")
        
        report_lines.extend([
            "="*80,
            "KEY OBSERVATIONS",
            "="*80,
            "- Word frequency analysis reveals distinctive vocabulary patterns per radicality level",
            "- Text length metrics (words/characters) show distribution characteristics",
            "- Word clouds provide visual representation of dominant themes",
            "",
        ])
        
        report_text = "\n".join(report_lines)
        
        output_path = os.path.join(self.output_dir, 'radicality_analysis_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Saved summary report: {output_path}")
        print("\n" + report_text)
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*80)
        print("Starting Radicality Distribution Analysis")
        print("="*80 + "\n")
        
        # Load samples
        self.load_samples()
        
        # Generate analysis for each radicality level
        print("\nGenerating analysis for each radicality level...")
        for radicality in self.radicality_order:
            print(f"\n--- Processing {radicality} ---")
            frequencies = self.calculate_word_frequencies(radicality)
            
            if frequencies:
                self.generate_wordcloud(radicality, frequencies)
                self.save_word_frequency_csv(radicality, frequencies)
            else:
                print(f"No data for {radicality}")
        
        # Generate comparative reports and visualizations
        print("\nGenerating comparative reports...")
        self.generate_radicality_comparison_report()
        self.generate_distribution_chart()
        self.generate_frequency_comparison()
        self.generate_text_length_comparison()
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("Analysis Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")


if __name__ == "__main__":
    input_file = "generated_samples/samples_79x4x20.jsonl"
    analyzer = RadicalityAnalyzer(input_file)
    analyzer.run_analysis()
