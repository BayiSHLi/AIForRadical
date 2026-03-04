"""
Coded Text Statistics Module
结合text内容和coded标记的统计分析
在indicator分析前的基础数据探索
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')


def detect_likely_handles(word_freq, threshold=0.7):
    """
    Detect likely Twitter handles or usernames based on frequency patterns
    从dataset.py中复用的方法
    
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


class CodedTextStatistics:
    """
    结合Coded标记的文本统计分析器
    
    功能：
    1. 数据集基础统计 (coded=0 vs coded=1的分布)
    2. 文本质量评估 (长度、单词数、有效性)
    3. 词频对比分析 (两组的词汇差异)
    4. 类别分布分析 (各类别的coded比例)
    5. 时间或人物维度的coded分析
    """
    
    def __init__(self, dataset=None, data=None):
        """
        初始化分析器
        
        Args:
            dataset: RadicalisationDataset对象
            data: pandas DataFrame
        """
        self.data = None
        self.stopwords = set()
        self.stats = {}
        
        if dataset is not None:
            self._load_from_dataset(dataset)
        elif data is not None:
            self.data = data.copy()
        
        self._initialize_stopwords()
    
    def _load_from_dataset(self, dataset):
        """从RadicalisationDataset加载数据"""
        print("正在加载数据集...")
        
        rows = []
        for idx, row in enumerate(dataset.data):
            content = str(row.get('content', '')).strip()
            rows.append({
                'id': idx,
                'content': content,
                'coded': row.get('coded', 0),
                'category': row.get('category', 'unknown'),
                'person': row.get('person', 'unknown'),
                'date': row.get('date', 'unknown'),
                'handle': row.get('handle', 'unknown'),
            })
        
        self.data = pd.DataFrame(rows)
        print(f"✓ 加载完成: {len(self.data)} 条posts")
    
    def _initialize_stopwords(self):
        """初始化停用词"""
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        except:
            # 降级：使用基础停用词
            self.stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'was', 'are', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 'as', 'just', 'now'
            }
    
    def basic_statistics(self):
        """生成数据集基础统计"""
        print("\n" + "="*80)
        print("基础统计分析")
        print("="*80)
        
        # 总体统计
        total = len(self.data)
        coded_1 = (self.data['coded'] == 1).sum()
        coded_0 = (self.data['coded'] == 0).sum()
        
        print(f"\n1. 数据集规模")
        print(f"   总posts数: {total:,}")
        print(f"   Coded=1:   {coded_1:,} ({coded_1/total*100:.1f}%)")
        print(f"   Coded=0:   {coded_0:,} ({coded_0/total*100:.1f}%)")
        
        # 按category分布
        print(f"\n2. 按Category的分布")
        category_stats = self.data.groupby('category').agg({
            'coded': ['count', lambda x: (x == 1).sum(), lambda x: (x == 0).sum()]
        }).round(2)
        category_stats.columns = ['Total', 'Coded=1', 'Coded=0']
        category_stats['Coded=1 %'] = (category_stats['Coded=1'] / category_stats['Total'] * 100).round(1)
        
        for category, row in category_stats.iterrows():
            print(f"   {category:30} Total:{row['Total']:5.0f}  Coded=1:{row['Coded=1']:4.0f} ({row['Coded=1 %']:5.1f}%)")
        
        # 按person分布 (只显示前10个)
        print(f"\n3. 按Person的分布 (Top 10)")
        person_stats = self.data.groupby('person').agg({
            'coded': ['count', lambda x: (x == 1).sum()]
        }).round(2)
        person_stats.columns = ['Total', 'Coded=1']
        person_stats['Coded=1 %'] = (person_stats['Coded=1'] / person_stats['Total'] * 100).round(1)
        person_stats = person_stats.sort_values('Total', ascending=False).head(10)
        
        for person, row in person_stats.iterrows():
            print(f"   {person:30} Total:{row['Total']:5.0f}  Coded=1:{row['Coded=1']:4.0f} ({row['Coded=1 %']:5.1f}%)")
        
        self.stats['category_stats'] = category_stats
        self.stats['person_stats'] = person_stats
        
        return category_stats, person_stats
    
    def text_quality_analysis(self):
        """分析文本质量和特征"""
        print("\n" + "="*80)
        print("文本质量分析")
        print("="*80)
        
        # 计算文本指标
        self.data['text_length'] = self.data['content'].str.len()
        self.data['word_count'] = self.data['content'].str.split().str.len()
        self.data['has_content'] = (self.data['text_length'] > 10).astype(int)
        
        # 按coded分组统计
        print("\nCoded=0 的文本特征:")
        coded_0_stats = self.data[self.data['coded'] == 0].describe()[['text_length', 'word_count']]
        print(coded_0_stats)
        
        print("\nCoded=1 的文本特征:")
        coded_1_stats = self.data[self.data['coded'] == 1].describe()[['text_length', 'word_count']]
        print(coded_1_stats)
        
        # 详细对比
        print("\n详细对比:")
        for metric in ['text_length', 'word_count']:
            coded_0_data = self.data[self.data['coded'] == 0][metric]
            coded_1_data = self.data[self.data['coded'] == 1][metric]
            
            print(f"\n{metric}:")
            print(f"  Coded=0: 平均={coded_0_data.mean():.1f}, 中位数={coded_0_data.median():.1f}, 标准差={coded_0_data.std():.1f}")
            print(f"  Coded=1: 平均={coded_1_data.mean():.1f}, 中位数={coded_1_data.median():.1f}, 标准差={coded_1_data.std():.1f}")
            print(f"  差异:    {coded_1_data.mean() - coded_0_data.mean():+.1f}")
        
        # 有效文本比例
        print(f"\n有效文本比例 (length > 10):")
        valid_0 = (self.data[self.data['coded'] == 0]['has_content'] == 1).sum()
        valid_1 = (self.data[self.data['coded'] == 1]['has_content'] == 1).sum()
        total_0 = (self.data['coded'] == 0).sum()
        total_1 = (self.data['coded'] == 1).sum()
        
        print(f"  Coded=0: {valid_0:,} / {total_0:,} ({valid_0/total_0*100:.1f}%)")
        print(f"  Coded=1: {valid_1:,} / {total_1:,} ({valid_1/total_1*100:.1f}%)")
        
        self.stats['text_length'] = {
            'coded_0': self.data[self.data['coded'] == 0]['text_length'],
            'coded_1': self.data[self.data['coded'] == 1]['text_length']
        }
        self.stats['word_count'] = {
            'coded_0': self.data[self.data['coded'] == 0]['word_count'],
            'coded_1': self.data[self.data['coded'] == 1]['word_count']
        }
    
    def word_frequency_analysis(self, top_n=30):
        """对比coded=0和coded=1的词频"""
        print("\n" + "="*80)
        print("词频对比分析")
        print("="*80)
        
        def extract_words(text):
            """提取单词"""
            text = text.lower()
            # 移除URL
            text = re.sub(r'http\S+|www\S+', '', text)
            # 移除@mention和#hashtag符号
            text = re.sub(r'@\S+|#', '', text)
            # 分词
            words = re.findall(r'\b[a-z]+\b', text)
            # 过滤停用词和长度
            words = [w for w in words if w not in self.stopwords and len(w) >= 3]
            return words
        
        # 提取coded=0和coded=1的单词
        coded_0_texts = self.data[self.data['coded'] == 0]['content']
        coded_1_texts = self.data[self.data['coded'] == 1]['content']
        
        coded_0_words = []
        coded_1_words = []
        
        print(f"正在处理 {len(coded_0_texts)} 条coded=0的文本...")
        for text in coded_0_texts:
            coded_0_words.extend(extract_words(text))
        
        print(f"正在处理 {len(coded_1_texts)} 条coded=1的文本...")
        for text in coded_1_texts:
            coded_1_words.extend(extract_words(text))
        
        # 计算词频
        freq_0 = Counter(coded_0_words)
        freq_1 = Counter(coded_1_words)
        
        # 检测并移除handles
        print(f"\n正在检测handles...")
        handles_0 = detect_likely_handles(freq_0, threshold=0.5)
        handles_1 = detect_likely_handles(freq_1, threshold=0.5)
        all_handles = set(list(handles_0.keys()) + list(handles_1.keys()))
        
        if all_handles:
            print(f"  发现 {len(all_handles)} 个suspected handles:")
            for handle in sorted(all_handles):
                conf_0 = handles_0.get(handle, {}).get('confidence', 0)
                conf_1 = handles_1.get(handle, {}).get('confidence', 0)
                freq_count_0 = freq_0.get(handle, 0)
                freq_count_1 = freq_1.get(handle, 0)
                print(f"    {handle:20} Coded0:{freq_count_0:6,} (conf:{conf_0:.2f})  Coded1:{freq_count_1:6,} (conf:{conf_1:.2f})")
            
            # 移除handles
            freq_0 = Counter({w: c for w, c in freq_0.items() if w not in all_handles})
            freq_1 = Counter({w: c for w, c in freq_1.items() if w not in all_handles})
            
            # 重新计算总单词数（移除handles后）
            len_coded_0_words = sum(freq_0.values())
            len_coded_1_words = sum(freq_1.values())
            print(f"  ✓ 已移除handles")
            print(f"    Coded=0: {len(coded_0_words):,} → {len_coded_0_words:,} 单词 (-{len(coded_0_words) - len_coded_0_words:,})")
            print(f"    Coded=1: {len(coded_1_words):,} → {len_coded_1_words:,} 单词 (-{len(coded_1_words) - len_coded_1_words:,})")
            print()
        else:
            len_coded_0_words = len(coded_0_words)
            len_coded_1_words = len(coded_1_words)
            all_handles = set()  # 初始化空set
        
        print(f"\nCoded=0 文本统计:")
        print(f"  总单词数: {len_coded_0_words:,}")
        print(f"  不同单词数: {len(freq_0):,}")
        print(f"  Top {top_n} 单词:")
        for word, count in freq_0.most_common(top_n):
            pct = count / len_coded_0_words * 100
            print(f"    {word:20} {count:6,} ({pct:.2f}%)")
        
        print(f"\nCoded=1 文本统计:")
        print(f"  总单词数: {len_coded_1_words:,}")
        print(f"  不同单词数: {len(freq_1):,}")
        print(f"  Top {top_n} 单词:")
        for word, count in freq_1.most_common(top_n):
            pct = count / len_coded_1_words * 100
            print(f"    {word:20} {count:6,} ({pct:.2f}%)")
        
        # 对比：coded=1独有或更高频的单词
        print(f"\nCoded=1 相对 Coded=0 更高频的单词 (Score = (f1 - f0) / f0):")
        print(f"解释: Score = (Coded=1的相对频率 - Coded=0的相对频率) / Coded=0的相对频率")
        print(f"      衡量Coded=1相对于Coded=0的超额频率倍数\n")
        
        # 归一化频率
        freq_0_norm = {w: c / len_coded_0_words for w, c in freq_0.items()}
        freq_1_norm = {w: c / len_coded_1_words for w, c in freq_1.items()}
        
        # 计算相对频率（TF-IDF式）
        # Score = (f1 - f0) / f0，当f0=0时略过（在Coded=0中不存在的词）
        relative_freq = {}
        for word in freq_1:
            f1 = freq_1_norm.get(word, 0)
            f0 = freq_0_norm.get(word, 0)
            if f0 > 0:  # 只计算在Coded=0中也出现的词
                relative_freq[word] = (f1 - f0) / f0
        
        sorted_relative = sorted(relative_freq.items(), key=lambda x: -x[1])
        for word, score in sorted_relative[:15]:
            freq_1_count = freq_1.get(word, 0)
            freq_0_count = freq_0.get(word, 0)
            f0 = freq_0_norm.get(word, 0)
            f1 = freq_1_norm.get(word, 0)
            print(f"  {word:20} f0={f0*100:.3f}% f1={f1*100:.3f}%  Coded0:{freq_0_count:6,}  Coded1:{freq_1_count:6,}  Score:{score:+8.2f}")
        
        # 反向分析：coded=0独有或更高频的单词
        print(f"\nCoded=0 相对 Coded=1 更高频的单词 (Score = (f0 - f1) / f1):")
        print(f"解释: Score = (Coded=0的相对频率 - Coded=1的相对频率) / Coded=1的相对频率")
        print(f"      衡量Coded=0相对于Coded=1的超额频率倍数\n")
        
        # 计算反向相对频率
        # Score = (f0 - f1) / f1，当f1=0时略过（在Coded=1中不存在的词）
        reverse_relative_freq = {}
        for word in freq_0:
            f0 = freq_0_norm.get(word, 0)
            f1 = freq_1_norm.get(word, 0)
            if f1 > 0:  # 只计算在Coded=1中也出现的词
                reverse_relative_freq[word] = (f0 - f1) / f1
        
        sorted_reverse_relative = sorted(reverse_relative_freq.items(), key=lambda x: -x[1])
        for word, score in sorted_reverse_relative[:15]:
            freq_0_count = freq_0.get(word, 0)
            freq_1_count = freq_1.get(word, 0)
            f0 = freq_0_norm.get(word, 0)
            f1 = freq_1_norm.get(word, 0)
            print(f"  {word:20} f0={f0*100:.3f}% f1={f1*100:.3f}%  Coded0:{freq_0_count:6,}  Coded1:{freq_1_count:6,}  Score:{score:+8.2f}")
        
        self.stats['word_freq_0'] = freq_0
        self.stats['word_freq_1'] = freq_1
        self.stats['detected_handles'] = all_handles
        
        return freq_0, freq_1
    
    def visualize_frequency_comparison(self, top_n=20, save_dir='./analysis_coded_text_statistic'):
        """生成词频对比柱状图（按Coded=0频率由高到低排序）"""
        if 'word_freq_0' not in self.stats or 'word_freq_1' not in self.stats:
            print("  ⚠ 跳过词频对比图：未执行词频分析")
            return
        
        freq_0 = self.stats['word_freq_0']
        freq_1 = self.stats['word_freq_1']
        
        # 计算总单词数
        total_0 = sum(freq_0.values())
        total_1 = sum(freq_1.values())
        
        # 转换为相对频率（百分比）
        freq_0_norm = {w: (c / total_0 * 100) for w, c in freq_0.items()}
        freq_1_norm = {w: (c / total_1 * 100) for w, c in freq_1.items()}

        # 获取top_n的词（按Coded=0的相对频率排序）
        top_words = [w for w, _ in sorted(freq_0_norm.items(), key=lambda x: -x[1])[:top_n]]
        
        # 准备数据（相对频率）
        freq_0_values = [freq_0_norm.get(w, 0) for w in top_words]
        freq_1_values = [freq_1_norm.get(w, 0) for w in top_words]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(top_words))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, freq_0_values, width, label='Coded=0', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, freq_1_values, width, label='Coded=1', color='#e74c3c', alpha=0.8)
        
        # 添加数值标签（显示百分比）
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:  # 只显示非常小的值
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}%',
                           ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        ax.set_xlabel('Words', fontweight='bold', fontsize=12)
        ax.set_ylabel('Relative Frequency (%)', fontweight='bold', fontsize=12)
        ax.set_title('Word Frequency Comparison: Coded=0 vs Coded=1 (Normalized by Group)\nTop 20 Words Sorted by Coded=0 Frequency', 
                     fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(top_words, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/word_frequency_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存: word_frequency_comparison.png")
        plt.close()
    
    def visualize_wordclouds(self, save_dir='./analysis_coded_text_statistic'):
        """生成词云图（Coded=0和Coded=1各一个）"""
        if 'word_freq_0' not in self.stats or 'word_freq_1' not in self.stats:
            print("  ⚠ 跳过词云图：未执行词频分析")
            return
        
        freq_0 = self.stats['word_freq_0']
        freq_1 = self.stats['word_freq_1']
        
        # 创建图表：两个词云并排
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Coded=0的词云
        if len(freq_0) > 0:
            wc_0 = WordCloud(width=800, height=600, 
                            background_color='white',
                            colormap='Greens',
                            relative_scaling=0.5).generate_from_frequencies(freq_0)
            axes[0].imshow(wc_0, interpolation='bilinear')
            axes[0].set_title('Word Cloud: Coded=0', fontweight='bold', fontsize=13)
            axes[0].axis('off')
        
        # Coded=1的词云
        if len(freq_1) > 0:
            wc_1 = WordCloud(width=800, height=600,
                            background_color='white',
                            colormap='Reds',
                            relative_scaling=0.5).generate_from_frequencies(freq_1)
            axes[1].imshow(wc_1, interpolation='bilinear')
            axes[1].set_title('Word Cloud: Coded=1', fontweight='bold', fontsize=13)
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/word_clouds.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存: word_clouds.png")
        plt.close()
    
    def category_coded_analysis(self):
        """按category分析coded的分布"""
        print("\n" + "="*80)
        print("按Category的Coded分布分析")
        print("="*80)
        
        # 识别fighters和sympathisers
        fighters = self.data[self.data['category'].str.contains('fighter', case=False, na=False)]
        sympathisers = self.data[~self.data['category'].str.contains('fighter', case=False, na=False)]
        
        print(f"\n群体分类:")
        print(f"  Fighters: {len(fighters):,} posts")
        print(f"  Sympathisers: {len(sympathisers):,} posts")
        
        # coded分布
        fighter_coded_1 = (fighters['coded'] == 1).sum()
        sympathiser_coded_1 = (sympathisers['coded'] == 1).sum()
        
        print(f"\nCoded=1 分布:")
        print(f"  Fighters: {fighter_coded_1:,} / {len(fighters):,} ({fighter_coded_1/len(fighters)*100:.1f}%)")
        print(f"  Sympathisers: {sympathiser_coded_1:,} / {len(sympathisers):,} ({sympathiser_coded_1/len(sympathisers)*100:.1f}%)")
        
        # 详细的category级别分析
        print(f"\n详细的Category级别分析:")
        category_coded = self.data.groupby('category').agg({
            'coded': [
                'count',
                lambda x: (x == 1).sum(),
                lambda x: (x == 0).sum(),
                lambda x: (x == 1).sum() / len(x) * 100 if len(x) > 0 else 0
            ]
        })
        category_coded.columns = ['Total', 'Coded=1', 'Coded=0', 'Coded=1 %']
        category_coded = category_coded.sort_values('Coded=1 %', ascending=False)
        
        for category, row in category_coded.iterrows():
            bar_length = int(row['Coded=1 %'] / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"  {category:35} {bar} {row['Coded=1 %']:5.1f}% ({int(row['Coded=1']):4.0f}/{int(row['Total']):5.0f})")
        
        self.stats['category_coded'] = category_coded
    
    def visualize_coded_statistics(self, save_dir='./analysis_coded_text_statistic'):
        """生成统计可视化"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n生成可视化图表...")
        
        # 1. Coded分布饼图
        if 'category_stats' in self.stats:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 总体分布
            coded_counts = [
                (self.data['coded'] == 0).sum(),
                (self.data['coded'] == 1).sum()
            ]
            colors = ['#2ecc71', '#e74c3c']
            axes[0].pie(coded_counts, labels=['Coded=0', 'Coded=1'], autopct='%1.1f%%',
                       colors=colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
            axes[0].set_title('Overall Coded Distribution', fontsize=13, fontweight='bold')
            
            # Category分布
            category_coded = self.data.groupby('category')['coded'].apply(lambda x: (x == 1).sum())
            category_total = self.data.groupby('category')['coded'].count()
            category_pct = (category_coded / category_total * 100).sort_values(ascending=True)
            
            axes[1].barh(range(len(category_pct)), category_pct.values, color='#3498db', alpha=0.8)
            axes[1].set_yticks(range(len(category_pct)))
            axes[1].set_yticklabels(category_pct.index)
            axes[1].set_xlabel('Percentage of Coded=1 (%)', fontweight='bold')
            axes[1].set_title('Coded=1 Percentage by Category', fontsize=13, fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/coded_distribution.png', dpi=150, bbox_inches='tight')
            print(f"  ✓ 保存: coded_distribution.png")
            plt.close()
        
        # 2. 文本长度分布
        if 'text_length' in self.stats:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].hist([self.stats['text_length']['coded_0'], self.stats['text_length']['coded_1']],
                        label=['Coded=0', 'Coded=1'], bins=50, color=['#2ecc71', '#e74c3c'], alpha=0.7)
            axes[0].set_xlabel('Text Length (characters)', fontweight='bold')
            axes[0].set_ylabel('Frequency', fontweight='bold')
            axes[0].set_title('Text Length Distribution', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            
            # Box plot
            data_to_plot = [self.stats['text_length']['coded_0'], self.stats['text_length']['coded_1']]
            bp = axes[1].boxplot(data_to_plot, labels=['Coded=0', 'Coded=1'], patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[1].set_ylabel('Text Length (characters)', fontweight='bold')
            axes[1].set_title('Text Length Comparison', fontsize=13, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/text_length_distribution.png', dpi=150, bbox_inches='tight')
            print(f"  ✓ 保存: text_length_distribution.png")
            plt.close()
        
        # 3. 单词数分布
        if 'word_count' in self.stats:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].hist([self.stats['word_count']['coded_0'], self.stats['word_count']['coded_1']],
                        label=['Coded=0', 'Coded=1'], bins=50, color=['#2ecc71', '#e74c3c'], alpha=0.7)
            axes[0].set_xlabel('Word Count', fontweight='bold')
            axes[0].set_ylabel('Frequency', fontweight='bold')
            axes[0].set_title('Word Count Distribution', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            
            # Box plot
            data_to_plot = [self.stats['word_count']['coded_0'], self.stats['word_count']['coded_1']]
            bp = axes[1].boxplot(data_to_plot, labels=['Coded=0', 'Coded=1'], patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[1].set_ylabel('Word Count', fontweight='bold')
            axes[1].set_title('Word Count Comparison', fontsize=13, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/word_count_distribution.png', dpi=150, bbox_inches='tight')
            print(f"  ✓ 保存: word_count_distribution.png")
            plt.close()
        
        # 4. 词频对比柱状图
        self.visualize_frequency_comparison(top_n=20, save_dir=save_dir)
        
        # 5. 词云图
        self.visualize_wordclouds(save_dir=save_dir)
    
    def generate_summary_report(self, save_dir='./analysis_coded_text_statistic', output_file='coded_text_statistics_report.txt'):
        """生成综合统计报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        report_path = os.path.join(save_dir, output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CODED TEXT STATISTICS REPORT\n")
            f.write("Text Analysis Combined with Coded Labels\n")
            f.write("="*80 + "\n\n")
            
            # 基础统计
            f.write("1. BASIC STATISTICS\n")
            f.write("-" * 80 + "\n")
            total = len(self.data)
            coded_1 = (self.data['coded'] == 1).sum()
            coded_0 = (self.data['coded'] == 0).sum()
            f.write(f"Total posts: {total:,}\n")
            f.write(f"Coded=1:     {coded_1:,} ({coded_1/total*100:.1f}%)\n")
            f.write(f"Coded=0:     {coded_0:,} ({coded_0/total*100:.1f}%)\n\n")
            
            # 文本质量
            f.write("2. TEXT QUALITY METRICS\n")
            f.write("-" * 80 + "\n")
            if 'text_length' in self.stats:
                coded_0_len = self.stats['text_length']['coded_0']
                coded_1_len = self.stats['text_length']['coded_1']
                
                f.write("Text Length (characters):\n")
                f.write(f"  Coded=0: Mean={coded_0_len.mean():.1f}, Median={coded_0_len.median():.1f}, Std={coded_0_len.std():.1f}\n")
                f.write(f"  Coded=1: Mean={coded_1_len.mean():.1f}, Median={coded_1_len.median():.1f}, Std={coded_1_len.std():.1f}\n")
                f.write(f"  Difference: {coded_1_len.mean() - coded_0_len.mean():+.1f}\n\n")
            
            if 'word_count' in self.stats:
                coded_0_wc = self.stats['word_count']['coded_0']
                coded_1_wc = self.stats['word_count']['coded_1']
                
                f.write("Word Count:\n")
                f.write(f"  Coded=0: Mean={coded_0_wc.mean():.1f}, Median={coded_0_wc.median():.1f}, Std={coded_0_wc.std():.1f}\n")
                f.write(f"  Coded=1: Mean={coded_1_wc.mean():.1f}, Median={coded_1_wc.median():.1f}, Std={coded_1_wc.std():.1f}\n")
                f.write(f"  Difference: {coded_1_wc.mean() - coded_0_wc.mean():+.1f}\n\n")
            
            # 词频对比
            f.write("3. WORD FREQUENCY COMPARISON\n")
            f.write("-" * 80 + "\n")
            
            # 先输出detected handles信息
            if 'detected_handles' in self.stats:
                handles = self.stats['detected_handles']
                f.write(f"Detected Handles (removed from analysis): {len(handles)}\n")
                if handles:
                    for handle in sorted(handles):
                        f.write(f"  - {handle}\n")
                f.write("\n")
            
            if 'word_freq_0' in self.stats and 'word_freq_1' in self.stats:
                freq_0 = self.stats['word_freq_0']
                freq_1 = self.stats['word_freq_1']
                
                f.write("Coded=0 Top Words (after removing handles):\n")
                for word, count in freq_0.most_common(10):
                    f.write(f"  {word:20} {count:6,}\n")
                
                f.write("\nCoded=1 Top Words (after removing handles):\n")
                for word, count in freq_1.most_common(10):
                    f.write(f"  {word:20} {count:6,}\n")
            
            # Category分析
            f.write("\n4. CATEGORY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            if 'category_coded' in self.stats:
                cat_coded = self.stats['category_coded']
                for category, row in cat_coded.iterrows():
                    f.write(f"{category:35} Coded=1: {int(row['Coded=1']):4.0f} / {int(row['Total']):5.0f} ({row['Coded=1 %']:5.1f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ 报告生成: {report_path}")
        return report_path


def main():
    """主分析流程"""
    from dataset import RadicalisationDataset
    
    print("="*80)
    print("CODED TEXT STATISTICS ANALYSIS")
    print("="*80)
    
    # 加载数据
    root_dir = r"c:\Users\shanghong.li\Desktop\AI for radicalisation\Fighter and sympathiser"
    print(f"\n正在加载数据集: {root_dir}")
    dataset = RadicalisationDataset(root_dir)
    
    # 创建分析器
    analyzer = CodedTextStatistics(dataset=dataset)
    
    # 基础统计
    analyzer.basic_statistics()
    
    # 文本质量分析
    analyzer.text_quality_analysis()
    
    # 词频分析
    analyzer.word_frequency_analysis(top_n=30)
    
    # Category分析
    analyzer.category_coded_analysis()
    
    # 生成可视化
    print("\n" + "="*80)
    print("生成可视化...")
    print("="*80)
    analyzer.visualize_coded_statistics()
    
    # 生成报告
    print("\n" + "="*80)
    print("生成报告...")
    print("="*80)
    analyzer.generate_summary_report()
    
    print("\n✓ 分析完成！")
    print("输出目录: ./analysis_coded_text_statistic/")


if __name__ == "__main__":
    main()
