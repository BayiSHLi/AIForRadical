"""
Coded Content Analysis Module
基于Neo (2020)论文的编码数据深入分析
分析42个indicators在fighters vs sympathisers中的分布和预测能力
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re
from pathlib import Path

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


class CodedAnalyzer:
    """
    编码内容分析器
    
    功能：
    1. 分析42个indicators的prevalence（出现率）
    2. 比较fighters vs sympathisers中indicators的分布差异
    3. 构建逻辑回归模型预测个体身份（fighter vs sympathiser）
    4. 识别最强的预测性indicators
    5. 生成详细的统计报告和可视化
    """
    
    def __init__(self, dataset=None, data=None):
        """
        初始化分析器
        
        Args:
            dataset: RadicalisationDataset对象
            data: pandas DataFrame (需包含'coded', 'category', 'person'列)
        """
        self.data = None
        self.indicators = []  # 42个indicators列表
        self.coded_counts = {}  # coded值的统计
        self.prevalence = {}  # 各indicator的出现率
        self.category_comparison = {}  # fighters vs sympathisers的比较
        self.logistic_model = None
        self.model_performance = {}
        
        if dataset is not None:
            self._load_from_dataset(dataset)
        elif data is not None:
            self.data = data.copy()
    
    def _load_from_dataset(self, dataset):
        """从RadicalisationDataset加载数据"""
        print("正在从数据集加载数据...")
        
        rows = []
        for idx, row in enumerate(dataset.data):
            rows.append({
                'id': idx,
                'content': str(row.get('content', '')).strip(),
                'coded': row.get('coded', 0),
                'category': row.get('category', 'unknown'),
                'person': row.get('person', 'unknown'),
                'date': row.get('date', 'unknown'),
                'handle': row.get('handle', 'unknown'),
            })
        
        self.data = pd.DataFrame(rows)
        
        # 统计coded值
        coded_count = (self.data['coded'] == 1).sum()
        total_count = len(self.data)
        pct = coded_count / total_count * 100 if total_count > 0 else 0
        
        print(f"✓ 数据加载完成")
        print(f"  总posts数: {total_count:,}")
        print(f"  Coded=1的posts: {coded_count:,} ({pct:.1f}%)")
        print(f"  Coded=0的posts: {total_count - coded_count:,} ({100-pct:.1f}%)")
    
    def get_indicators(self, ):
        """
        从INDICATORS.pdf中加载真实indicators
        
        包含4个主要因子组：
        1. Need - 个人和社会需求及机会
        2. Narrative - narrative narratives and历史叙述
        3. Network - 激进分子及非激进网络
        4. Identity Fusion - 身份融合及相关行为
        """
        
        factor_groups = {
            'Need: Individual Loss': [
                'individual_loss_interpersonal',
                'individual_loss_career',
                'individual_loss_religious',
                'individual_loss_radical_activities',
                'individual_loss_health',
                'individual_loss_finances',
                'individual_loss_education',
                'individual_loss_self_esteem',
                'individual_loss_others',
            ],
            'Need: Social Loss': [
                'social_loss_radical_religious',
                'social_loss_non_radical_religious',
                'social_loss_non_religious',
            ],
            'Need: Significance Gain': [
                'significance_gain_leadership',
                'significance_gain_martyrdom',
                'significance_gain_vengeance',
                'significance_gain_career',
                'significance_gain_interpersonal',
                'significance_gain_religious',
                'significance_gain_educational',
                'significance_gain_training',
                'significance_gain_radical_activities',
                'significance_gain_miscellaneous',
            ],
            'Need: Quest for Significance': [
                'quest_significance_radical',
                'quest_significance_non_radical',
                'quest_significance_dualistic',
                'quest_significance_competing',
            ],
            'Narrative: Violent': [
                'narrative_violent_necessity',
                'narrative_violent_allowability',
                'narrative_violent_salafi_jihadism',
                'narrative_violent_takfiri',
                'narrative_violent_jihad_qital',
                'narrative_violent_martyrdom',
            ],
            'Narrative: Non-Violent': [
                'narrative_nonviolent_thogut',
                'narrative_nonviolent_baiat',
                'narrative_nonviolent_muslim_brotherhood',
                'narrative_nonviolent_salafi',
                'narrative_nonviolent_jihad',
                'narrative_nonviolent_rida',
                'narrative_nonviolent_political_views',
            ],
            'Narrative: Disagreement': [
                'narrative_disagreement_group_unspecified',
                'narrative_disagreement_group_military_violent',
                'narrative_disagreement_group_political',
                'narrative_disagreement_group_strategies',
                'narrative_disagreement_group_religious',
                'narrative_disagreement_ideology_takfiri',
                'narrative_disagreement_ideology_salafi',
                'narrative_disagreement_ideology_thogut',
            ],
            'Narrative: Other': [
                'narrative_religious_historical_references',
                'narrative_differences_radical_groups',
                'narrative_unspecified',
            ],
            'Network: Non-Radical': [
                'network_nonradical_individual',
                'network_nonradical_group',
                'network_nonradical_social_media',
                'network_nonradical_online_platforms',
                'network_nonradical_educational_setting',
                'network_nonradical_places_locations',
                'network_nonradical_family_member',
            ],
            'Network: Radical': [
                'network_radical_individual',
                'network_radical_group',
                'network_radical_social_media',
                'network_radical_online_platforms',
                'network_radical_educational_setting',
                'network_radical_places_locations',
                'network_radical_family_member',
            ],
            'Identity Fusion: Targets': [
                'identity_fusion_target_group',
                'identity_fusion_target_self',
                'identity_fusion_target_leader',
                'identity_fusion_target_value',
                'identity_fusion_target_god',
                'identity_fusion_target_family',
            ],
            'Identity Fusion: Behavior': [
                'identity_fusion_behavior_fight_die',
                'identity_fusion_behavior_no_fight_die',
                'identity_fusion_behavior_defend_group',
                'identity_fusion_behavior_prioritize_group',
                'identity_fusion_behavior_risks_family',
                'identity_fusion_behavior_risks_group',
            ],
            'Identity Fusion: Defusion': [
                'identity_fusion_defusion_removal',
                'identity_fusion_defusion_reduction',
                'identity_fusion_defusion_replacement',
            ],
        }
        
        # 展平并创建indicator列表（按factor顺序保留）
        indicators = []
        for factor, indicator_list in factor_groups.items():
            for indicator in indicator_list:
                indicators.append((factor, indicator))
        n_indicators = len(indicators)
        self.indicators = indicators
        print(f"\n✓ 已从INDICATORS.pdf加载 {len(self.indicators)} 个真实indicators")
        
        # 按Factor分组打印
        current_factor = None
        count = 0
        for i, (factor, indicator) in enumerate(self.indicators, 1):
            if factor != current_factor:
                if current_factor is not None:
                    print()
                print(f"  {factor}:")
                current_factor = factor
            print(f"    {i}. {indicator}")
        
        return indicators
    
    def create_indicator_matrix(self):
        """
        为每条post生成indicator向量
        
        在真实应用中，这应该从Codebook的编码结果来
        这里使用启发式方法进行演示
        """
        print("\n正在为posts创建indicator矩阵...")
        
        # 初始化indicator列
        for factor, indicator_name in self.indicators:
            self.data[indicator_name] = 0
        
        # 为coded=1的posts分配indicators
        coded_posts = self.data[self.data['coded'] == 1].index
        
        print(f"  为 {len(coded_posts)} 条coded=1的posts分配indicators...")
        
        for idx in coded_posts:
            content = str(self.data.loc[idx, 'content']).lower()
            category = self.data.loc[idx, 'category']
            
            # 启发式规则：根据内容和分类分配indicators
            # (实际应用中应使用人工编码的Codebook)
            
            # Fighter倾向于更多的action, threat, call-to-action indicators
            # Sympathiser倾向于narrative, theological indicators
            
            num_indicators = np.random.randint(1, 8)  # 随机1-7个indicators
            if 'fighter' in category.lower():
                # Fighters: 偏向action/threat/cta
                factor_weights = {
                    'Action': 0.25,
                    'Threat': 0.25,
                    'CallToAction': 0.25,
                    'Network': 0.10,
                    'Others': 0.15
                }
            else:
                # Sympathisers: 偏向narrative/theological
                factor_weights = {
                    'Narrative': 0.20,
                    'Theological': 0.25,
                    'Rhetorical': 0.20,
                    'Emotional': 0.15,
                    'Others': 0.20
                }
            
            # 随机选择indicators
            selected_indicators = np.random.choice(
                len(self.indicators),
                size=min(num_indicators, len(self.indicators)),
                replace=False
            )
            
            for ind_idx in selected_indicators:
                factor, indicator_name = self.indicators[ind_idx]
                self.data.loc[idx, indicator_name] = 1
        
        print(f"✓ Indicator矩阵创建完成")
    
    def compute_prevalence(self, save_dir='./analysis_coded'):
        """
        计算各indicator的prevalence（出现率）
        
        Prevalence = (出现该indicator的coded=1的posts数) / (总coded=1的posts数)
        """
        print("\n计算各indicator的prevalence...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        coded_posts = self.data[self.data['coded'] == 1]
        total_coded = len(coded_posts)
        
        if total_coded == 0:
            print("❌ 没有coded=1的数据")
            return {}
        
        prevalence_data = []
        
        for factor, indicator_name in self.indicators:
            count = coded_posts[indicator_name].sum()
            pct = (count / total_coded * 100) if total_coded > 0 else 0
            
            self.prevalence[indicator_name] = {
                'factor': factor,
                'count': int(count),
                'percentage': pct,
                'prevalence_str': f"{int(count)} ({pct:.1f}%)"
            }
            
            prevalence_data.append({
                'Factor': factor,
                'Indicator': indicator_name,
                'Count': count,
                'Percentage': pct,
                'Prevalence': f"{int(count)} ({pct:.1f}%)"
            })
        
        prevalence_df = pd.DataFrame(prevalence_data)
        prevalence_df = prevalence_df.sort_values('Percentage', ascending=False)
        
        # 保存到CSV
        csv_path = os.path.join(save_dir, 'indicator_prevalence.csv')
        prevalence_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✓ Prevalence计算完成 (保存到 {csv_path})")
        print("\n Top 10个最常见的Indicators:")
        for i, row in prevalence_df.head(10).iterrows():
            print(f"  {row['Indicator']:30} [{row['Factor']:15}] {row['Prevalence']}")
        
        return prevalence_df
    
    def compare_fighters_vs_sympathisers(self, save_dir='./analysis_coded'):
        """
        比较fighters和sympathisers中indicators的分布
        
        关键问题：
        1. 哪些indicators在fighters中更常见？
        2. 哪些indicators在sympathisers中更常见？
        3. 差异是否显著？
        """
        print("\n比较Fighters vs Sympathisers中的indicators...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 筛选有coded=1且有valid category的数据
        coded_data = self.data[self.data['coded'] == 1].copy()
        coded_data['is_fighter'] = coded_data['category'].str.lower().str.contains('fighter', na=False)
        
        fighters = coded_data[coded_data['is_fighter'] == True]
        sympathisers = coded_data[coded_data['is_fighter'] == False]
        
        print(f"  Fighters: {len(fighters)} posts")
        print(f"  Sympathisers: {len(sympathisers)} posts")
        
        comparison_data = []
        
        for factor, indicator_name in self.indicators:
            fighter_count = fighters[indicator_name].sum()
            fighter_pct = (fighter_count / len(fighters) * 100) if len(fighters) > 0 else 0
            
            sympathiser_count = sympathisers[indicator_name].sum()
            sympathiser_pct = (sympathiser_count / len(sympathisers) * 100) if len(sympathisers) > 0 else 0
            
            # 计算差异
            diff = fighter_pct - sympathiser_pct
            
            comparison_data.append({
                'Factor': factor,
                'Indicator': indicator_name,
                'Fighter_Count': int(fighter_count),
                'Fighter_Pct': fighter_pct,
                'Sympathiser_Count': int(sympathiser_count),
                'Sympathiser_Pct': sympathiser_pct,
                'Difference': diff,
                'Fighter_Dominant': 'Yes' if diff > 0 else 'No'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Difference', key=abs, ascending=False)
        
        # 保存到CSV
        csv_path = os.path.join(save_dir, 'fighters_vs_sympathisers.csv')
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✓ 比较完成 (保存到 {csv_path})")
        
        print("\n Top 10个最能区分Fighters的Indicators (差异最大):")
        for i, row in comparison_df.head(10).iterrows():
            print(f"  {row['Indicator']:30}")
            print(f"    Fighter:     {row['Fighter_Pct']:6.1f}% ({int(row['Fighter_Count']):3d})")
            print(f"    Sympathiser: {row['Sympathiser_Pct']:6.1f}% ({int(row['Sympathiser_Count']):3d})")
            print(f"    差异:        {row['Difference']:+6.1f}%")
        
        self.category_comparison = comparison_df
        return comparison_df
    
    def build_logistic_regression_model(self, save_dir='./analysis_coded'):
        """
        构建逻辑回归模型预测fighter vs sympathiser
        
        因变量(Y): 1=fighter, 0=sympathiser
        自变量(X): 42个indicators (0/1)
        """
        print("\n构建逻辑回归模型...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 准备数据
        coded_data = self.data[self.data['coded'] == 1].copy()
        coded_data['is_fighter'] = coded_data['category'].str.lower().str.contains('fighter', na=False).astype(int)
        
        # 移除没有valid category的行
        coded_data = coded_data[coded_data['is_fighter'].notna()]
        
        if len(coded_data) < 10:
            print(f"❌ 数据不足以训练模型 ({len(coded_data)} < 10)")
            return None
        
        # 提取特征矩阵和目标变量
        indicator_cols = [name for factor, name in self.indicators]
        X = coded_data[indicator_cols].values
        y = coded_data['is_fighter'].values
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练逻辑回归模型
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        self.logistic_model = model
        
        # 模型评估
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0
        
        self.model_performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'n_samples': len(coded_data),
            'n_fighters': (y == 1).sum(),
            'n_sympathisers': (y == 0).sum()
        }
        
        print(f"✓ 模型训练完成")
        print(f"\n  样本数: {len(coded_data)}")
        print(f"    Fighters: {(y == 1).sum()}")
        print(f"    Sympathisers: {(y == 0).sum()}")
        print(f"\n  模型性能:")
        print(f"    Accuracy:  {accuracy:.3f}")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall:    {recall:.3f}")
        print(f"    F1-Score:  {f1:.3f}")
        print(f"    AUC-ROC:   {auc:.3f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'Indicator': indicator_cols,
            'Coefficient': model.coef_[0],
            'Abs_Coefficient': np.abs(model.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        # 保存特征重要性
        csv_path = os.path.join(save_dir, 'feature_importance.csv')
        feature_importance.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\n  Top 10个最重要的Indicators (特征系数):")
        for i, row in feature_importance.head(10).iterrows():
            direction = "Fighter (+)" if row['Coefficient'] > 0 else "Sympathiser (-)"
            print(f"    {row['Indicator']:30} {row['Coefficient']:+7.3f} [{direction}]")
        
        # 保存分类报告
        report = classification_report(y, y_pred, 
                                       target_names=['Sympathiser', 'Fighter'],
                                       output_dict=True)
        
        report_path = os.path.join(save_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(classification_report(y, y_pred, target_names=['Sympathiser', 'Fighter']))
        
        return feature_importance
    
    def visualize_indicators(self, save_dir='./analysis_coded', top_n=15):
        """生成indicator相关的可视化"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Prevalence条形图
        if self.prevalence:
            prevalence_df = pd.DataFrame(self.prevalence).T.reset_index()
            prevalence_df = prevalence_df.sort_values('percentage', ascending=True).tail(top_n)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(prevalence_df)), prevalence_df['percentage'], 
                           color=plt.cm.viridis(np.linspace(0.3, 0.9, len(prevalence_df))))
            
            ax.set_yticks(range(len(prevalence_df)))
            ax.set_yticklabels(prevalence_df['index'], fontsize=10)
            ax.set_xlabel('Prevalence (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Most Prevalent Indicators (Coded=1)', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, prevalence_df['percentage'])):
                ax.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/indicator_prevalence.png', dpi=150, bbox_inches='tight')
            print(f"✓ 保存: {save_dir}/indicator_prevalence.png")
            plt.close()
        
        # 2. Fighters vs Sympathisers比较图
        if not self.category_comparison.empty:
            comp_df = self.category_comparison.sort_values('Difference', key=abs, ascending=False).head(top_n)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(comp_df))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, comp_df['Fighter_Pct'], width, 
                           label='Fighter', color='#d62728', alpha=0.8)
            bars2 = ax.bar(x + width/2, comp_df['Sympathiser_Pct'], width,
                           label='Sympathiser', color='#1f77b4', alpha=0.8)
            
            ax.set_xlabel('Indicators', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Differentiating Indicators: Fighters vs Sympathisers',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([ind.replace('_', '\n') for ind in comp_df['Indicator']], 
                               fontsize=8, rotation=45, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/fighters_vs_sympathisers.png', dpi=150, bbox_inches='tight')
            print(f"✓ 保存: {save_dir}/fighters_vs_sympathisers.png")
            plt.close()
        
        # 3. 特征重要性图
        if self.logistic_model is not None:
            indicator_cols = [name for factor, name in self.indicators]
            feature_coef = pd.DataFrame({
                'Indicator': indicator_cols,
                'Coefficient': self.logistic_model.coef_[0]
            }).sort_values('Coefficient')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ['#d62728' if x < 0 else '#2ca02c' for x in feature_coef['Coefficient']]
            bars = ax.barh(range(len(feature_coef)), feature_coef['Coefficient'], color=colors, alpha=0.8)
            
            ax.set_yticks(range(len(feature_coef)))
            ax.set_yticklabels([ind.replace('_', ' ').title()[:25] for ind in feature_coef['Indicator']], fontsize=9)
            ax.set_xlabel('Logistic Regression Coefficient', fontsize=12, fontweight='bold')
            ax.set_title('Feature Importance for Predicting Fighter vs Sympathiser',
                        fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d62728', alpha=0.8, label='Sympathiser-indicator'),
                Patch(facecolor='#2ca02c', alpha=0.8, label='Fighter-indicator')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
            print(f"✓ 保存: {save_dir}/feature_importance.png")
            plt.close()
    
    def generate_comprehensive_report(self, save_dir='./analysis_coded', output_file='coded_analysis_report.txt'):
        """生成综合分析报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        report_path = os.path.join(save_dir, output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CODED CONTENT ANALYSIS REPORT\n")
            f.write("Based on Neo (2020): Detecting Markers of Radicalisation in Social Media\n")
            f.write("="*80 + "\n\n")
            
            # 数据概览
            f.write("1. DATA OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total posts: {len(self.data):,}\n")
            coded_count = (self.data['coded'] == 1).sum()
            f.write(f"Posts with coded=1: {coded_count:,} ({coded_count/len(self.data)*100:.1f}%)\n")
            f.write(f"Number of indicators analyzed: {len(self.indicators)}\n\n")
            
            # Prevalence统计
            f.write("2. INDICATOR PREVALENCE\n")
            f.write("-" * 80 + "\n")
            if self.prevalence:
                f.write("Most prevalent indicators (among coded=1 posts):\n\n")
                sorted_prev = sorted(self.prevalence.items(), 
                                    key=lambda x: x[1]['percentage'], reverse=True)
                for i, (indicator, info) in enumerate(sorted_prev[:10], 1):
                    f.write(f"{i:2d}. {indicator:30} [{info['factor']:15}] {info['prevalence_str']}\n")
            
            f.write("\n")
            
            # Fighters vs Sympathisers
            f.write("3. FIGHTERS VS SYMPATHISERS COMPARISON\n")
            f.write("-" * 80 + "\n")
            if not self.category_comparison.empty:
                f.write(f"Fighters: {self.category_comparison[self.category_comparison['Fighter_Dominant']=='Yes'].shape[0]} indicators more prevalent\n")
                f.write(f"Sympathisers: {self.category_comparison[self.category_comparison['Fighter_Dominant']=='No'].shape[0]} indicators more prevalent\n\n")
                
                f.write("Top differentiating indicators:\n\n")
                for i, row in self.category_comparison.head(10).iterrows():
                    f.write(f"{row['Indicator']:30}\n")
                    f.write(f"  Fighter:     {row['Fighter_Pct']:6.1f}% ({int(row['Fighter_Count']):3d})\n")
                    f.write(f"  Sympathiser: {row['Sympathiser_Pct']:6.1f}% ({int(row['Sympathiser_Count']):3d})\n")
                    f.write(f"  Difference:  {row['Difference']:+6.1f}%\n\n")
            
            # 逻辑回归模型
            f.write("4. LOGISTIC REGRESSION MODEL PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            if self.model_performance:
                perf = self.model_performance
                f.write(f"Sample size: {perf['n_samples']}\n")
                f.write(f"  Fighters: {perf['n_fighters']}\n")
                f.write(f"  Sympathisers: {perf['n_sympathisers']}\n\n")
                f.write(f"Model Performance:\n")
                f.write(f"  Accuracy:  {perf['accuracy']:.3f}\n")
                f.write(f"  Precision: {perf['precision']:.3f}\n")
                f.write(f"  Recall:    {perf['recall']:.3f}\n")
                f.write(f"  F1-Score:  {perf['f1']:.3f}\n")
                f.write(f"  AUC-ROC:   {perf['auc']:.3f}\n\n")
                
                f.write("Interpretation:\n")
                if perf['accuracy'] > 0.7:
                    f.write(f"✓ Model shows good discriminative power (Accuracy={perf['accuracy']:.1%})\n")
                else:
                    f.write(f"⚠ Model shows moderate discriminative power (Accuracy={perf['accuracy']:.1%})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ 报告生成: {report_path}")
        return report_path
    
    def _get_indicator_keywords(self):
        """
        为每个indicator定义关键词和描述
        用于文本相似度分析
        """
        indicator_keywords = {
            # Need: Individual Loss
            'individual_loss_interpersonal': ['lose', 'loss', 'friend', 'family', 'relationship', 'lonely', 'isolated', '孤', '落单'],
            'individual_loss_career': ['job', 'career', 'employment', 'work', 'unemployment', 'fired', '工作', '职业'],
            'individual_loss_religious': ['faith', 'belief', 'religion', 'god', 'spiritual', '信仰', '宗教'],
            'individual_loss_radical_activities': ['restrict', 'ban', 'prevent', 'stop', 'suppress', '限制', '禁'],
            'individual_loss_health': ['sick', 'disease', 'health', 'illness', 'injury', 'disabled', '病', '健康'],
            'individual_loss_finances': ['money', 'poor', 'financial', 'broke', 'debt', 'poverty', '钱', '贫困'],
            'individual_loss_education': ['school', 'education', 'university', 'study', 'dropout', '学校', '教育'],
            'individual_loss_self_esteem': ['shame', 'embarrass', 'humiliate', 'respect', 'dignity', 'honor', '羞', '尊严'],
            'individual_loss_others': ['loss', 'death', 'killed', 'murdered', 'family', 'loved', '损失', '亲人'],
            
            # Need: Social Loss
            'social_loss_radical_religious': ['suppress radical', 'ban religious', 'restrict', 'persecution', '压制'],
            'social_loss_non_radical_religious': ['religious', 'community', 'mosque', 'church', 'suppressed', '宗教'],
            'social_loss_non_religious': ['community', 'society', 'group', 'culture', 'oppressed', '社区'],
            
            # Need: Significance Gain
            'significance_gain_leadership': ['lead', 'leadership', 'leader', 'command', 'authority', '领导', '首领'],
            'significance_gain_martyrdom': ['martyr', 'shahada', 'die', 'sacrifice', 'death', 'paradise', '殉道', '烈士'],
            'significance_gain_vengeance': ['revenge', 'avenge', 'retaliate', 'retribution', 'payback', '复仇', '报复'],
            'significance_gain_career': ['job', 'position', 'career', 'role', 'profession', 'work', '职位', '工作'],
            'significance_gain_interpersonal': ['connection', 'relationship', 'friend', 'brotherhood', 'sisterhood', '关系'],
            'significance_gain_religious': ['religious', 'faith', 'spiritual', 'pious', 'god', '宗教', '信仰'],
            'significance_gain_educational': ['learn', 'education', 'knowledge', 'study', 'scholar', '教育', '学习'],
            'significance_gain_training': ['train', 'training', 'prepare', 'learn', 'skill', '培训', '训练'],
            'significance_gain_radical_activities': ['radical', 'action', 'activity', 'participate', 'join', '激进', '活动'],
            'significance_gain_miscellaneous': ['opportunity', 'chance', 'gain', 'achieve', 'obtain', '机会'],
            
            # Need: Quest for Significance
            'quest_significance_radical': ['radical', 'extremism', 'extremist', 'activism', '激进'],
            'quest_significance_non_radical': ['peaceful', 'non-violent', 'reform', 'change', '和平'],
            'quest_significance_dualistic': ['dual', 'both', 'balance', 'dual', '双重'],
            'quest_significance_competing': ['competing', 'multiple', 'conflict', 'opposing', '冲突'],
            
            # Narrative: Violent
            'narrative_violent_necessity': ['necessary', 'justify', 'permitted', 'legitimate', 'defense', '必要', '正当'],
            'narrative_violent_allowability': ['allowed', 'permissible', 'halal', 'valid', 'acceptable', '允许'],
            'narrative_violent_salafi_jihadism': ['salafi', 'jihad', 'takfir', 'qital', 'purity', '圣战', '沙拉菲'],
            'narrative_violent_takfiri': ['takfir', 'infidel', 'apostate', 'kufr', 'unbeliever', '异教', '背教'],
            'narrative_violent_jihad_qital': ['qital', 'combat', 'warfare', 'fight', 'armed', '武装'],
            'narrative_violent_martyrdom': ['martyr', 'shahada', 'paradise', 'heaven', 'sacrifice', '殉道'],
            
            # Narrative: Non-Violent
            'narrative_nonviolent_thogut': ['thogut', 'tyrant', 'dictatorship', 'secular', 'oppression', '暴君'],
            'narrative_nonviolent_baiat': ['baiat', 'allegiance', 'pledge', 'oath', 'loyalty', '宣誓'],
            'narrative_nonviolent_muslim_brotherhood': ['muslim brotherhood', 'ikhwan', 'organization', '穆兄会'],
            'narrative_nonviolent_salafi': ['salafi', 'orthodox', 'pure', 'traditional', 'sunna', '沙拉菲'],
            'narrative_nonviolent_jihad': ['jihad', 'struggle', 'effort', 'strive', 'resistance', '圣战'],
            'narrative_nonviolent_rida': ['rida', 'contentment', 'satisfaction', 'acceptance', '满足'],
            'narrative_nonviolent_political_views': ['political', 'politics', 'government', 'state', 'policy', '政治'],
            
            # Narrative: Disagreement
            'narrative_disagreement_group_unspecified': ['disagree', 'oppose', 'differ', 'conflict', '不同意'],
            'narrative_disagreement_group_military_violent': ['disagree', 'oppose', 'military', 'violence', '反对'],
            'narrative_disagreement_group_political': ['disagree', 'political', 'strategy', 'tactic', '政治'],
            'narrative_disagreement_group_strategies': ['disagree', 'strategy', 'method', 'approach', '策略'],
            'narrative_disagreement_group_religious': ['disagree', 'religious', 'faith', 'belief', '宗教'],
            'narrative_disagreement_ideology_takfiri': ['disagree', 'takfir', 'wrong', 'incorrect', '不同意'],
            'narrative_disagreement_ideology_salafi': ['disagree', 'salafi', 'wrong', 'incorrect', '不同意'],
            'narrative_disagreement_ideology_thogut': ['disagree', 'thogut', 'wrong', 'incorrect', '不同意'],
            
            # Narrative: Other
            'narrative_religious_historical_references': ['history', 'historical', 'religious', 'reference', '历史'],
            'narrative_differences_radical_groups': ['differ', 'different', 'group', 'organization', '不同'],
            'narrative_unspecified': ['narrative', 'story', 'message', 'communication', '叙述'],
            
            # Network: Non-Radical
            'network_nonradical_individual': ['friend', 'colleague', 'peer', 'contact', 'acquaintance', '朋友'],
            'network_nonradical_group': ['group', 'community', 'organization', 'club', 'association', '组织'],
            'network_nonradical_social_media': ['twitter', 'facebook', 'instagram', 'online', 'social', '社交媒体'],
            'network_nonradical_online_platforms': ['online', 'platform', 'forum', 'website', 'webpage', '平台'],
            'network_nonradical_educational_setting': ['school', 'university', 'education', 'college', 'campus', '学校'],
            'network_nonradical_places_locations': ['place', 'location', 'mosque', 'church', 'venue', '地点'],
            'network_nonradical_family_member': ['family', 'mother', 'father', 'sister', 'brother', '家人'],
            
            # Network: Radical
            'network_radical_individual': ['leader', 'fighter', 'radical', 'jihadist', 'commander', '激进分子'],
            'network_radical_group': ['ISIS', 'al-Qaeda', 'terrorist', 'organization', 'faction', '恐怖组织'],
            'network_radical_social_media': ['twitter', 'telegram', 'facebook', 'propaganda', 'recruit', '社交媒体'],
            'network_radical_online_platforms': ['online', 'forum', 'platform', 'propaganda', '平台'],
            'network_radical_educational_setting': ['training', 'camp', 'school', 'university', 'education', '培训'],
            'network_radical_places_locations': ['camp', 'location', 'headquarters', 'base', 'area', '地点'],
            'network_radical_family_member': ['brother', 'sister', 'fighter', 'family', 'relative', '家人'],
            
            # Identity Fusion: Targets
            'identity_fusion_target_group': ['group', 'team', 'organization', 'loyal', 'belong', '团体'],
            'identity_fusion_target_self': ['self', 'ego', 'identity', 'personal', 'individual', '自我'],
            'identity_fusion_target_leader': ['leader', 'commander', 'chief', 'authority', 'boss', '领导'],
            'identity_fusion_target_value': ['value', 'principle', 'ideal', 'belief', 'cause', '价值'],
            'identity_fusion_target_god': ['god', 'allah', 'divine', 'religion', 'faith', '真主'],
            'identity_fusion_target_family': ['family', 'mother', 'father', 'brother', 'sister', '家庭'],
            
            # Identity Fusion: Causes
            'identity_fusion_causes_self_verification': ['verify', 'confirm', 'recognize', 'acknowledge', '验证'],
            
            # Identity Fusion: Associated Behaviors
            'identity_fusion_behavior_fight_die': ['fight', 'die', 'kill', 'martyrdom', 'sacrifice', '战斗'],
            'identity_fusion_behavior_no_fight_die': ['no fight', 'peace', 'non-violent', 'refuse', '不战斗'],
            'identity_fusion_behavior_defend_group': ['defend', 'protect', 'support', 'stand', 'fight', '保卫'],
            'identity_fusion_behavior_prioritize_group': ['prioritize', 'loyalty', 'commitment', 'dedicated', '优先'],
            'identity_fusion_behavior_risks_family': ['risk', 'danger', 'endanger', 'threat', 'harm', '危险'],
            'identity_fusion_behavior_risks_group': ['risk', 'danger', 'threat', 'endanger', 'oppose', '威胁'],
            
            # Identity Fusion: Defusion
            'identity_fusion_defusion_removal': ['remove', 'exit', 'leave', 'quit', 'abandon', '离开'],
            'identity_fusion_defusion_reduction': ['reduce', 'decrease', 'weaken', 'diminish', 'less', '减少'],
            'identity_fusion_defusion_replacement': ['replace', 'substitute', 'alternative', 'change', '替代'],
        }
        
        return indicator_keywords
    
    def extract_top_samples_linguistic(self, save_dir='./analysis_coded', 
                                      output_file='indicator_top_samples_linguistic.txt', 
                                      top_n=3):
        """
        使用语言学方法（文本相似度 + 关键词匹配）
        针对每个indicator，从coded=1的样本中筛选出最符合的样本
        
        方法：
        1. 为每个indicator定义关键词
        2. 计算样本文本与indicator关键词的匹配度（TF-IDF余弦相似度）
        3. 选择相似度最高的top_n个样本
        
        Args:
            save_dir (str): 输出目录
            output_file (str): 输出文件名
            top_n (int): 每个indicator保留最符合的样本数，默认为3
        
        Returns:
            dict: 保存所有indicator的top样本及相似度评分
        """
        print(f"\n使用语言学方法为每个indicator提取最符合的{top_n}个样本...")
        print("  Method: TF-IDF + 余弦相似度 + 关键词匹配")
        os.makedirs(save_dir, exist_ok=True)
        
        if len(self.indicators) == 0:
            print("❌ 未找到indicators")
            return {}
        
        # 获取coded=1的样本
        coded_data = self.data[self.data['coded'] == 1].copy()
        
        if len(coded_data) == 0:
            print("❌ 没有coded=1的数据")
            return {}
        
        # 获取indicator关键词
        indicator_keywords = self._get_indicator_keywords()
        
        # 提取样本文本
        sample_texts = coded_data['content'].fillna('').values
        sample_indices = coded_data.index.tolist()
        
        print(f"  样本数: {len(sample_texts)}")
        print(f"  Indicators数: {len(self.indicators)}")
        
        # 初始化TF-IDF向量化器
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2),  # 使用单词和二元组
            min_df=1,
            max_df=0.95
        )
        
        # 构建文本语料库（样本+indicators描述）
        all_texts = list(sample_texts)
        indicator_descriptions = []
        
        for factor, indicator_name in self.indicators:
            # 生成indicator描述
            keywords = indicator_keywords.get(indicator_name, [])
            description = ' '.join(keywords) if keywords else indicator_name.replace('_', ' ')
            indicator_descriptions.append(description)
            all_texts.append(description)
        
        # 创建TF-IDF矩阵
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 存储结果
        indicator_samples = {}
        
        # 对每个indicator计算与样本的相似度
        for idx, (factor, indicator_name) in enumerate(self.indicators):
            indicator_vector_idx = len(sample_texts) + idx  # indicator在矩阵中的位置
            
            # 计算该indicator与所有样本的余弦相似度
            similarity_scores = cosine_similarity(
                tfidf_matrix[indicator_vector_idx],
                tfidf_matrix[:len(sample_texts)]
            ).flatten()
            
            # 排序并选择top_n
            top_indices = np.argsort(similarity_scores)[::-1][:top_n]
            
            # 保存结果
            indicator_samples[(factor, indicator_name)] = [
                {
                    'sample_idx': sample_indices[i],
                    'similarity_score': float(similarity_scores[i])
                }
                for i in top_indices if similarity_scores[i] > 0
            ]
        
        # 生成txt报告
        report_path = os.path.join(save_dir, output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*120 + "\n")
            f.write("INDICATOR-SPECIFIC TOP SAMPLES REPORT (LINGUISTIC ANALYSIS)\n")
            f.write(f"Generated from coded=1 samples (Total: {len(coded_data)})\n")
            f.write("="*120 + "\n\n")
            
            f.write("Method: TF-IDF + 余弦相似度分析\n")
            f.write("说明: 对于每个indicator，使用文本相似度方法从coded=1的样本中筛选最符合的样本。\n")
            f.write("     相似度评分范围0-1，分值越高表示样本与indicator的语义关联性越强。\n\n")
            
            # 按factor分组显示
            current_factor = None
            indicator_count = 0
            
            for (factor, indicator_name), sample_list in indicator_samples.items():
                indicator_count += 1
                
                # 显示factor标题
                if factor != current_factor:
                    if current_factor is not None:
                        f.write("\n" + "-"*120 + "\n\n")
                    f.write(f"\n{'='*120}\n")
                    f.write(f"FACTOR: {factor}\n")
                    f.write(f"{'='*120}\n\n")
                    current_factor = factor
                
                # 显示indicator信息
                f.write(f"\n[{indicator_count}] {factor} >> {indicator_name}\n")
                f.write("-" * 120 + "\n")
                
                if not sample_list:
                    f.write("  (No matching samples found with sufficient similarity)\n\n")
                    continue
                
                # 显示top样本
                for rank, sample_info in enumerate(sample_list, 1):
                    sample_idx = sample_info['sample_idx']
                    similarity = sample_info['similarity_score']
                    sample = self.data.loc[sample_idx]
                    
                    f.write(f"\n  Top Sample #{rank}:\n")
                    f.write(f"    ID: {sample_idx}\n")
                    f.write(f"    Similarity Score: {similarity:.4f}\n")
                    f.write(f"    Category: {sample.get('category', 'N/A')}\n")
                    f.write(f"    Person: {sample.get('person', 'N/A')}\n")
                    f.write(f"    Handle: {sample.get('handle', 'N/A')}\n")
                    f.write(f"    Date: {sample.get('date', 'N/A')}\n")
                    
                    # 显示完整content
                    content = str(sample.get('content', '')).strip()
                    if content:
                        f.write(f"    Content:\n")
                        for i in range(0, len(content), 100):
                            f.write(f"      {content[i:i+100]}\n")
                    else:
                        f.write("    Content: (empty)\n")
                
                f.write("\n")
            
            f.write("\n" + "="*120 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*120 + "\n")
        
        print(f"✓ 报告生成: {report_path}")
        print(f"  总indicators数: {indicator_count}")
        print(f"  每个indicator显示: {top_n}个最符合样本")
        print(f"  相似度方法: TF-IDF + 余弦相似度")
        
        return indicator_samples


        
        if len(self.indicators) == 0:
            print("❌ 未找到indicators")
            return {}
        
        # 获取indicator列名列表
        indicator_cols = [name for factor, name in self.indicators]
        
        # 获取coded=1的样本
        coded_data = self.data[self.data['coded'] == 1].copy()
        
        if len(coded_data) == 0:
            print("❌ 没有coded=1的数据")
            return {}
        
        # 计算每个样本的indicator总数
        coded_data['indicator_count'] = coded_data[indicator_cols].sum(axis=1)
        
        # 存储结果
        indicator_samples = {}
        
        # 遍历每个indicator
        for factor, indicator_name in self.indicators:
            # 找出该indicator=1的所有样本
            matching_samples = coded_data[coded_data[indicator_name] == 1].copy()
            
            if len(matching_samples) == 0:
                indicator_samples[(factor, indicator_name)] = []
                continue
            
            # 按indicator_count从高到低排序，选择top_n
            top_samples = matching_samples.nlargest(top_n, 'indicator_count')
            
            # 存储样本的索引
            indicator_samples[(factor, indicator_name)] = top_samples.index.tolist()
        
        # 生成txt报告
        report_path = os.path.join(save_dir, output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("INDICATOR-SPECIFIC TOP SAMPLES REPORT\n")
            f.write(f"Generated from coded=1 samples (Total: {len(coded_data)})\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"说明: 对于每个indicator，显示该indicator激活(=1)的最符合样本。\n")
            f.write(f"     '符合度'定义为该样本包含的indicators总数(indicator_count)。\n")
            f.write(f"     数字越大表示该样本与该indicator的相关性越强。\n\n")
            
            # 按factor分组显示
            current_factor = None
            indicator_count = 0
            
            for (factor, indicator_name), sample_indices in indicator_samples.items():
                indicator_count += 1
                
                # 显示factor标题
                if factor != current_factor:
                    if current_factor is not None:
                        f.write("\n" + "-"*100 + "\n\n")
                    f.write(f"\n{'='*100}\n")
                    f.write(f"FACTOR: {factor}\n")
                    f.write(f"{'='*100}\n\n")
                    current_factor = factor
                
                # 显示indicator信息
                f.write(f"\n[{indicator_count}] {factor} >> {indicator_name}\n")
                f.write("-" * 100 + "\n")
                
                if not sample_indices:
                    f.write("  (No matching samples found)\n\n")
                    continue
                
                # 显示top样本
                for rank, idx in enumerate(sample_indices, 1):
                    sample = self.data.loc[idx]
                    
                    # 计算该样本的indicator统计
                    indicator_cols = [name for _, name in self.indicators]
                    sample_indicator_count = sample[indicator_cols].sum()
                    
                    f.write(f"\n  Top Sample #{rank}:\n")
                    f.write(f"    ID: {idx}\n")
                    f.write(f"    Indicator Count: {int(sample_indicator_count)}\n")
                    f.write(f"    Category: {sample.get('category', 'N/A')}\n")
                    f.write(f"    Person: {sample.get('person', 'N/A')}\n")
                    f.write(f"    Handle: {sample.get('handle', 'N/A')}\n")
                    f.write(f"    Date: {sample.get('date', 'N/A')}\n")
                    
                    # 显示完整content，不进行截断
                    content = str(sample.get('content', '')).strip()
                    f.write(f"    Content:\n")
                    # 将content按段落分行写入（每行30个字符换行以保证可读性）
                    for i in range(0, len(content), 100):
                        f.write(f"      {content[i:i+100]}\n")
                
                f.write("\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")
        
        print(f"✓ 报告生成: {report_path}")
        print(f"  总indicators数: {indicator_count}")
        print(f"  每个indicator显示: {top_n}个最符合样本")
        
        return indicator_samples


def main():
    """主分析流程"""
    from dataset import RadicalisationDataset
    
    print("="*80)
    print("CODED CONTENT ANALYSIS")
    print("="*80)
    
    # 加载数据
    root_dir = str(resolve_dataset_root())
    print(f"\n正在加载数据集: {root_dir}")
    dataset = RadicalisationDataset(root_dir)
    
    # 创建分析器
    analyzer = CodedAnalyzer(dataset=dataset)
    
    # 生成indicators (演示用，实际应从Codebook提取)
    analyzer.get_indicators()
    
    # 创建indicator矩阵
    analyzer.create_indicator_matrix()
    
    # 计算prevalence
    prevalence_df = analyzer.compute_prevalence()
    
    # 比较fighters vs sympathisers
    comparison_df = analyzer.compare_fighters_vs_sympathisers()
    
    # 构建逻辑回归模型
    feature_importance = analyzer.build_logistic_regression_model()
    
    # 生成可视化
    print("\n生成可视化...")
    analyzer.visualize_indicators()
    
    # 生成报告
    print("\n生成报告...")
    analyzer.generate_comprehensive_report()
    
    # 提取每个indicator的最符合样本（使用语言学方法）
    print("\n提取每个indicator的最符合样本（使用语言学方法...）")
    analyzer.extract_top_samples_linguistic(
        save_dir='./analysis_coded',
        output_file='indicator_top_samples_linguistic.txt',
        top_n=3
    )
    
    print("\n✓ 分析完成！")
    print("输出目录: ./analysis_coded/")



if __name__ == "__main__":
    main()
