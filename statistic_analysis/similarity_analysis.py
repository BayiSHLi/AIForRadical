"""
Content Similarity Analysis Module
用于检测重复内容、转发模式和内容相似度分析
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import warnings
import re

warnings.filterwarnings('ignore')


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_dataset_root() -> Path:
    """Resolve dataset directory across possible repository layouts."""
    candidates = [
        PROJECT_ROOT / "data" / "Fighter and sympathiser",
        PROJECT_ROOT / "Fighter and sympathiser",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    checked = "\n".join([f"  - {p}" for p in candidates])
    raise FileNotFoundError(
        "Dataset root not found. Checked:\n"
        f"{checked}"
    )


class SimilarityAnalyzer:
    """
    Content Similarity Analysis Tool
    
    功能:
    1. 计算文本相似度 (TF-IDF + 余弦相似度)
    2. 检测重复和接近重复的内容
    3. 识别转发/复制模式
    4. 生成相似度报告和可视化
    """
    
    def __init__(self, dataset=None, texts=None, metadata=None):
        """
        初始化相似度分析器
        
        Args:
            dataset: RadicalisationDataset 对象或包含 'content' 列的列表
            texts: 文本列表（如果不提供dataset）
            metadata: 与文本对应的元数据（来源、作者等）
        """
        self.texts = []
        self.metadata = []
        self.similarity_matrix = None
        self.tfidf_matrix = None
        self.vectorizer = None
        
        if dataset is not None:
            self._load_from_dataset(dataset)
        elif texts is not None:
            self.texts = texts
            if metadata is None:
                self.metadata = [{"id": i, "source": "unknown"} for i in range(len(texts))]
            else:
                self.metadata = metadata
    
    def _load_from_dataset(self, dataset):
        """从RadicalisationDataset加载数据"""
        self.texts = []
        self.metadata = []
        
        for idx, row in enumerate(dataset.data):
            content = str(row.get('content', '')).strip()
            if content and len(content) > 10:  # 只处理有效的非空文本
                self.texts.append(content)
                self.metadata.append({
                    'id': idx,
                    'person': row.get('person', 'unknown'),
                    'category': row.get('category', 'unknown'),
                    'date': row.get('date', 'unknown'),
                    'handle': row.get('handle', 'unknown'),
                })
        
        print(f"✓ Loaded {len(self.texts)} valid texts from dataset")
    
    def compute_similarity_matrix(self, method='tfidf', top_n=None):
        """
        计算相似度矩阵
        
        Args:
            method (str): 'tfidf' 或 'count'
            top_n (int): 只处理前N个文本（加快计算）
        
        Returns:
            similarity_matrix: 相似度矩阵 (N x N)
        """
        if not self.texts:
            print("❌ No texts loaded. Please provide dataset or texts.")
            return None
        
        # 如果只处理前N个文本
        texts_to_use = self.texts[:top_n] if top_n else self.texts
        
        print(f"\n计算相似度矩阵 (共 {len(texts_to_use)} 个文本)...")
        
        if method == 'tfidf':
            # TF-IDF 向量化
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = vectorizer.fit_transform(texts_to_use)
            self.vectorizer = vectorizer
            
            # 计算余弦相似度
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
        elif method == 'count':
            # 词数向量化
            vectorizer = CountVectorizer(
                max_features=5000,
                stop_words='english',
                min_df=1,
                max_df=0.95
            )
            count_matrix = vectorizer.fit_transform(texts_to_use)
            self.similarity_matrix = cosine_similarity(count_matrix)
            self.vectorizer = vectorizer
        
        print(f"✓ 相似度矩阵计算完成: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def find_similar_pairs(self, threshold=0.7, top_k=50):
        """
        找出相似度高于阈值的文本对
        
        Args:
            threshold (float): 相似度阈值 (0-1)
            top_k (int): 返回的相似对数量
        
        Returns:
            list: 相似文本对列表，按相似度从高到低排序
        """
        if self.similarity_matrix is None:
            print("❌ 需要先计算相似度矩阵。调用 compute_similarity_matrix()")
            return []
        
        similar_pairs = []
        n = self.similarity_matrix.shape[0]  # 使用相似度矩阵的实际大小
        
        # 遍历相似度矩阵的上三角（避免重复和自比较）
        for i in range(n):
            for j in range(i + 1, n):
                sim_score = self.similarity_matrix[i, j]
                if sim_score >= threshold:
                    similar_pairs.append({
                        'idx1': i,
                        'idx2': j,
                        'similarity': sim_score,
                        'text1': self.texts[i][:100] + "...",
                        'text2': self.texts[j][:100] + "...",
                        'person1': self.metadata[i]['person'],
                        'person2': self.metadata[j]['person'],
                        'category1': self.metadata[i]['category'],
                        'category2': self.metadata[j]['category'],
                    })
        
        # 按相似度排序
        similar_pairs.sort(key=lambda x: -x['similarity'])
        
        # 返回top_k对
        similar_pairs = similar_pairs[:top_k]
        
        print(f"\n找到 {len(similar_pairs)} 个相似度 ≥ {threshold} 的文本对")
        return similar_pairs
    
    def find_duplicates(self, similarity_threshold=0.95):
        """
        找出基本重复的内容 (相似度 > 95%)
        通常表示转发或完全复制
        
        Args:
            similarity_threshold (float): 重复度阈值
        
        Returns:
            dict: 重复内容组
        """
        similar_pairs = self.find_similar_pairs(threshold=similarity_threshold, top_k=10000)
        
        # 构建重复内容组
        duplicate_groups = defaultdict(list)
        processed = set()
        
        for pair in similar_pairs:
            idx1, idx2 = pair['idx1'], pair['idx2']
            
            if idx1 in processed or idx2 in processed:
                continue
            
            group_id = len(duplicate_groups)
            duplicate_groups[group_id] = [
                {
                    'idx': idx1,
                    'person': self.metadata[idx1]['person'],
                    'category': self.metadata[idx1]['category'],
                    'date': self.metadata[idx1]['date'],
                    'text': self.texts[idx1][:100] + "..."
                },
                {
                    'idx': idx2,
                    'person': self.metadata[idx2]['person'],
                    'category': self.metadata[idx2]['category'],
                    'date': self.metadata[idx2]['date'],
                    'text': self.texts[idx2][:100] + "..."
                }
            ]
            processed.add(idx1)
            processed.add(idx2)
        
        print(f"\n找到 {len(duplicate_groups)} 个重复内容组")
        return duplicate_groups
    
    def analyze_retweet_patterns(self):
        """
        分析转发/复制模式
        识别：
        - 完全重复的内容
        - 高度相似的内容
        - 常见的短语/主题
        """
        print("\n" + "="*80)
        print("转发/复制模式分析")
        print("="*80)
        
        # 1. 完全重复
        duplicate_texts = defaultdict(list)
        for idx, text in enumerate(self.texts):
            # 规范化文本（移除多余空格）
            normalized = ' '.join(text.lower().split())
            duplicate_texts[normalized].append(idx)
        
        exact_duplicates = {k: v for k, v in duplicate_texts.items() if len(v) > 1}
        
        print(f"\n1. 完全重复的内容: {len(exact_duplicates)} 组")
        for text, indices in sorted(exact_duplicates.items(), key=lambda x: -len(x[1]))[:10]:
            people = [self.metadata[idx]['person'] for idx in indices]
            print(f"   出现 {len(indices)} 次，来自: {', '.join(set(people))}")
            print(f"   内容: {text[:80]}...")
        
        # 2. 高度相似的内容 (80-95%)
        similar_pairs = self.find_similar_pairs(threshold=0.80, top_k=100)
        high_sim_count = sum(1 for p in similar_pairs if p['similarity'] > 0.90)
        
        print(f"\n2. 高度相似的内容 (相似度 > 90%): {high_sim_count} 对")
        for pair in similar_pairs[:5]:
            print(f"   {pair['person1']} ← → {pair['person2']}")
            print(f"   相似度: {pair['similarity']:.2%}")
        
        # 3. 常见短语分析
        print(f"\n3. 常见短语分析 (长度 ≥ 30字符的高频短语)")
        phrases = self._extract_common_phrases(min_length=30, top_n=10)
        for phrase, count, people_set in phrases:
            print(f"   {phrase[:60]}...")
            print(f"   出现 {count} 次，来自 {len(people_set)} 个人")
        
        return {
            'exact_duplicates': exact_duplicates,
            'similar_pairs': similar_pairs,
            'phrases': phrases
        }
    
    def _extract_common_phrases(self, min_length=20, top_n=10):
        """提取常见的长短语"""
        phrases = defaultdict(lambda: {'count': 0, 'people': set()})
        
        # 分割成句子并查找相似的
        for idx, text in enumerate(self.texts):
            person = self.metadata[idx]['person']
            
            # 简单的句子分割
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= min_length:
                    # 规范化
                    normalized = ' '.join(sentence.lower().split())
                    phrases[normalized]['count'] += 1
                    phrases[normalized]['people'].add(person)
        
        # 按频率排序
        common_phrases = [
            (phrase, info['count'], info['people']) 
            for phrase, info in phrases.items()
            if info['count'] > 1  # 只要出现超过1次的
        ]
        common_phrases.sort(key=lambda x: -x[1])
        
        return common_phrases[:top_n]
    
    def visualize_similarity_heatmap(self, save_dir='./analysis', sample_size=50):
        """
        绘制相似度热力图
        
        Args:
            save_dir (str): 保存目录
            sample_size (int): 为了可读性，只显示前N个文本
        """
        if self.similarity_matrix is None:
            print("❌ 需要先计算相似度矩阵")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 只用前sample_size个
        sim_matrix = self.similarity_matrix[:sample_size, :sample_size]
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 绘制热力图
        sns.heatmap(sim_matrix, cmap='YlOrRd', square=True, 
                   cbar_kws={'label': 'Similarity Score'},
                   ax=ax, vmin=0, vmax=1)
        
        ax.set_title(f'Content Similarity Heatmap (First {sample_size} Texts)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Text Index', fontsize=12)
        ax.set_ylabel('Text Index', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/similarity_heatmap.png', dpi=150, bbox_inches='tight')
        print(f"✓ 热力图已保存: {save_dir}/similarity_heatmap.png")
        plt.close()
    
    def visualize_similar_pairs(self, similar_pairs, save_dir='./analysis', top_n=20):
        """
        绘制相似文本对的分布
        
        Args:
            similar_pairs (list): find_similar_pairs() 的返回结果
            save_dir (str): 保存目录
            top_n (int): 显示前N个
        """
        if not similar_pairs:
            print("❌ 没有相似对数据")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取前top_n
        pairs_to_plot = similar_pairs[:top_n]
        similarities = [p['similarity'] for p in pairs_to_plot]
        labels = [f"{p['person1'][:12]} - {p['person2'][:12]}" for p in pairs_to_plot]
        
        # 绘制条形图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(similarities)))
        bars = ax.barh(range(len(similarities)), similarities, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Similar Text Pairs', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, sim) in enumerate(zip(bars, similarities)):
            ax.text(sim + 0.01, i, f'{sim:.2%}', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/similar_pairs_top.png', dpi=150, bbox_inches='tight')
        print(f"✓ 相似对分布图已保存: {save_dir}/similar_pairs_top.png")
        plt.close()
    
    def visualize_person_similarity_network(self, save_dir='./analysis', threshold=0.75):
        """
        绘制个人之间的内容相似度网络
        
        Args:
            save_dir (str): 保存目录
            threshold (float): 相似度阈值
        """
        try:
            import networkx as nx
        except ImportError:
            print("⚠️  需要安装 networkx: pip install networkx")
            return
        
        if self.similarity_matrix is None:
            print("❌ 需要先计算相似度矩阵")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建网络图
        G = nx.Graph()
        
        # 添加节点（人物）
        people = list(set([m['person'] for m in self.metadata]))
        G.add_nodes_from(people)
        
        # 添加边（高相似度的连接）
        n = len(self.texts)
        for i in range(n):
            for j in range(i + 1, n):
                if self.similarity_matrix[i, j] >= threshold:
                    person1 = self.metadata[i]['person']
                    person2 = self.metadata[j]['person']
                    
                    if person1 != person2:  # 不同的人
                        weight = self.similarity_matrix[i, j]
                        if G.has_edge(person1, person2):
                            G[person1][person2]['weight'] += weight
                        else:
                            G.add_edge(person1, person2, weight=weight)
        
        # 绘制网络图
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 布局
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [2 + 5 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                              edge_color='gray', alpha=0.6)
        
        # 绘制节点
        node_sizes = [300 + 100 * G.degree(node) for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                              node_color='lightblue', edgecolors='navy',
                              linewidths=2)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
        
        ax.set_title(f'Person-to-Person Content Similarity Network (threshold={threshold})',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/similarity_network.png', dpi=150, bbox_inches='tight')
        print(f"✓ 网络图已保存: {save_dir}/similarity_network.png")
        plt.close()
    
    def generate_report(self, save_dir='./analysis', output_file='similarity_report.txt'):
        """
        生成完整的相似度分析报告
        
        Args:
            save_dir (str): 保存目录
            output_file (str): 报告文件名
        """
        os.makedirs(save_dir, exist_ok=True)
        
        report_path = os.path.join(save_dir, output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CONTENT SIMILARITY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # 基本统计
            f.write("1. BASIC STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total texts: {len(self.texts)}\n")
            f.write(f"Unique people: {len(set([m['person'] for m in self.metadata]))}\n")
            f.write(f"Average text length: {np.mean([len(t.split()) for t in self.texts]):.1f} words\n\n")
            
            # 相似度统计
            if self.similarity_matrix is not None:
                sim_values = self.similarity_matrix[np.triu_indices_from(
                    self.similarity_matrix, k=1)]
                f.write("2. SIMILARITY STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Min similarity: {sim_values.min():.4f}\n")
                f.write(f"Max similarity: {sim_values.max():.4f}\n")
                f.write(f"Mean similarity: {sim_values.mean():.4f}\n")
                f.write(f"Median similarity: {np.median(sim_values):.4f}\n")
                f.write(f"Std dev: {sim_values.std():.4f}\n\n")
                
                # 分布
                f.write("Distribution of Similarity Scores:\n")
                thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                for threshold in thresholds:
                    count = (sim_values >= threshold).sum()
                    pct = count / len(sim_values) * 100
                    f.write(f"  ≥ {threshold}: {count} pairs ({pct:.1f}%)\n")
            
            # 相似对
            f.write("\n\n3. SIMILAR TEXT PAIRS (similarity ≥ 0.7)\n")
            f.write("-" * 80 + "\n")
            similar_pairs = self.find_similar_pairs(threshold=0.7, top_k=50)
            
            for i, pair in enumerate(similar_pairs[:20], 1):
                f.write(f"\n#{i} Similarity: {pair['similarity']:.4f}\n")
                f.write(f"  Person 1: {pair['person1']} ({pair['category1']})\n")
                f.write(f"  Person 2: {pair['person2']} ({pair['category2']})\n")
                f.write(f"  Text 1: {pair['text1']}\n")
                f.write(f"  Text 2: {pair['text2']}\n")
            
            # 转发模式
            f.write("\n\n4. RETWEET/COPY PATTERNS\n")
            f.write("-" * 80 + "\n")
            patterns = self.analyze_retweet_patterns()
            
            f.write(f"\nExact Duplicates: {len(patterns['exact_duplicates'])} groups\n")
            for text, indices in list(patterns['exact_duplicates'].items())[:5]:
                people = list(set([self.metadata[idx]['person'] for idx in indices]))
                f.write(f"  {len(indices)} copies by: {', '.join(people[:3])}\n")
            
            f.write(f"\n\nCommon Phrases: {len(patterns['phrases'])} phrases detected\n")
            for phrase, count, people_set in patterns['phrases'][:10]:
                f.write(f"  {phrase[:50]}...\n")
                f.write(f"    Appeared {count} times, {len(people_set)} people involved\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ 报告已生成: {report_path}")
        return report_path
    
    def print_similar_pairs_summary(self, similar_pairs, top_n=20):
        """打印相似对的总结"""
        print("\n" + "="*80)
        print("SIMILAR TEXT PAIRS SUMMARY")
        print("="*80)
        
        for i, pair in enumerate(similar_pairs[:top_n], 1):
            print(f"\n#{i} [相似度: {pair['similarity']:.2%}]")
            print(f"  来源1: {pair['person1']} ({pair['category1']})")
            print(f"  来源2: {pair['person2']} ({pair['category2']})")
            print(f"  内容1: {pair['text1']}")
            print(f"  内容2: {pair['text2']}")


def main():
    """使用示例"""
    from dataset import RadicalisationDataset
    
    print("内容相似度分析示例")
    print("="*80)
    
    # 加载数据
    root_dir = str(resolve_dataset_root())
    print("正在加载数据...")
    dataset = RadicalisationDataset(root_dir)
    
    # 创建相似度分析器
    print("\n初始化相似度分析器...")
    analyzer = SimilarityAnalyzer(dataset=dataset)
    
    # 计算相似度（为了演示，只用前200个文本）
    print("\n计算相似度矩阵...")
    analyzer.compute_similarity_matrix(method='tfidf', top_n=200)
    
    # 找出相似对
    print("\n找出相似的文本对...")
    similar_pairs = analyzer.find_similar_pairs(threshold=0.7, top_k=50)
    analyzer.print_similar_pairs_summary(similar_pairs, top_n=10)
    
    # 分析转发模式
    print("\n分析转发/复制模式...")
    patterns = analyzer.analyze_retweet_patterns()
    
    # 生成可视化
    print("\n生成可视化...")
    analyzer.visualize_similarity_heatmap(save_dir='./analysis_results', sample_size=50)
    analyzer.visualize_similar_pairs(similar_pairs, save_dir='./analysis_results')
    
    try:
        analyzer.visualize_person_similarity_network(save_dir='./analysis_results', threshold=0.75)
    except Exception as e:
        print(f"⚠️  网络图生成失败: {e}")
    
    # 生成报告
    print("\n生成详细报告...")
    analyzer.generate_report(save_dir='./analysis_results')
    
    print("\n✓ 相似度分析完成！")
    print("输出目录: ./analysis_results/")


if __name__ == "__main__":
    main()
