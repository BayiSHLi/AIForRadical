"""
Fine-tune Text Embedding Models for Radicalization Detection Classification
基于text embeddings的激进化内容分类器finetune
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Import sentence-transformers and related libraries
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


class CodedDataset:
    """
    处理Radicalization数据集的类
    为embedding finetune准备数据
    """
    
    def __init__(self, dataset=None, data=None):
        """
        初始化数据集
        
        Args:
            dataset: RadicalisationDataset对象
            data: pandas DataFrame
        """
        self.data = None
        self.label_names = ['non-radicalisation', 'radicalization']
        
        if dataset is not None:
            self._load_from_dataset(dataset)
        elif data is not None:
            self.data = data.copy()
        
        self._prepare_data()
    
    def _load_from_dataset(self, dataset):
        """从RadicalisationDataset加载数据"""
        print("正在加载数据集...")
        
        rows = []
        for idx, row in enumerate(dataset.data):
            content = str(row.get('content', '')).strip()
            if len(content) > 5:  # 只保留有效的文本
                rows.append({
                    'text': content,
                    'coded': row.get('coded', 0),
                    'category': row.get('category', 'unknown'),
                    'person': row.get('person', 'unknown'),
                })
        
        self.data = pd.DataFrame(rows)
        print(f"✓ 加载完成: {len(self.data)} 条posts")
    
    def _prepare_data(self):
        """数据准备"""
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded")
        
        # 移除空值
        self.data = self.data[self.data['text'].notna()].copy()
        self.data['text'] = self.data['text'].str.strip()
        self.data = self.data[self.data['text'].str.len() > 5].copy()
        
        print(f"✓ 数据准备完成: {len(self.data)} 条有效posts")
    
    def get_statistics(self):
        """获取数据集统计"""
        stats = {
            'total': len(self.data),
            'coded_0': (self.data['coded'] == 0).sum(),
            'coded_1': (self.data['coded'] == 1).sum(),
            'avg_text_length': self.data['text'].str.len().mean(),
            'fighters': (self.data['category'].str.contains('fighter', case=False, na=False)).sum(),
            'sympathisers': (~self.data['category'].str.contains('fighter', case=False, na=False)).sum(),
        }
        return stats
    
    def split_data(self, val_size=0.1, test_size=0.1, random_state=42, stratify_by='coded'):
        """
        分割训练、验证、测试集
        
        Args:
            val_size: 验证集比例
            test_size: 测试集比例
            random_state: 随机种子
            stratify_by: 按哪个列进行分层抽样
        """
        # 第一次split：分离test集
        stratify_col = self.data[stratify_by] if stratify_by in self.data.columns else None
        
        train_val, test = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        # 第二次split：从train_val分离val集
        stratify_col_tv = train_val[stratify_by] if stratify_by in train_val.columns else None
        adjusted_val_size = val_size / (1 - test_size)
        
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=stratify_col_tv
        )
        
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class EmbeddingClassifier(nn.Module):
    """
    基于embedding的分类器
    使用预训练embedding + 轻量级分类头
    """
    
    def __init__(self, embedding_dim=384, num_classes=2, dropout=0.2):
        """
        初始化分类器
        
        Args:
            embedding_dim: embedding维度
            num_classes: 分类数（2: binary classification）
            dropout: dropout率
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # 分类头：embedding → hidden → output
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, embeddings):
        """前向传播"""
        return self.classifier(embeddings)


class EmbeddingFineTuner:
    """
    Text Embedding模型的finetune管理器
    """
    
    def __init__(self, 
                 model_name='all-MiniLM-L6-v2',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir='./embeddings_finetuned'):
        """
        初始化finetune器
        
        Args:
            model_name: SentenceTransformers模型名称
            device: 训练设备
            output_dir: 输出目录
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        
        print(f"正在加载embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"✓ Model loaded: {model_name}")
        print(f"  Device: {device}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.classifier = None
        self.optimizer = None
        self.scheduler = None
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
        }
    
    def get_embeddings(self, texts: List[str], batch_size=32):
        """
        获取文本的embeddings
        
        Args:
            texts: 文本列表
            batch_size: 批大小
            
        Returns:
            embeddings数组
        """
        print(f"正在计算embeddings ({len(texts)} texts)...")
        embeddings = self.embedding_model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings
    
    def create_classifier(self, num_classes=2, dropout=0.2):
        """创建分类器"""
        self.classifier = EmbeddingClassifier(
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        print(f"✓ Classifier created")
        print(f"  Parameters: {sum(p.numel() for p in self.classifier.parameters()):,}")
    
    def setup_optimizer(self, learning_rate=1e-3, weight_decay=0.01):
        """设置优化器"""
        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        print(f"✓ Optimizer configured: Adam (lr={learning_rate}, wd={weight_decay})")
    
    def train_epoch(self, train_embeddings, train_labels, batch_size=32):
        """训练一个epoch"""
        self.classifier.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 创建mini-batches
        indices = np.arange(len(train_embeddings))
        np.random.shuffle(indices)
        
        for i in tqdm(range(0, len(indices), batch_size), desc="Training"):
            batch_indices = indices[i:i+batch_size]
            
            batch_embeddings = torch.FloatTensor(train_embeddings[batch_indices]).to(self.device)
            batch_labels = torch.LongTensor(train_labels[batch_indices]).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.classifier(batch_embeddings)
            
            # Loss
            loss = nn.CrossEntropyLoss()(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * len(batch_indices)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += len(batch_indices)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def evaluate(self, val_embeddings, val_labels, batch_size=32):
        """评估模型"""
        self.classifier.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(val_embeddings), batch_size):
                batch_embeddings = torch.FloatTensor(val_embeddings[i:i+batch_size]).to(self.device)
                batch_labels = val_labels[i:i+batch_size]
                
                logits = self.classifier(batch_embeddings)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch_labels)
        
        all_logits = np.vstack(all_logits)
        all_labels = np.hstack(all_labels)
        
        # 计算指标
        loss = nn.CrossEntropyLoss()(
            torch.FloatTensor(all_logits),
            torch.LongTensor(all_labels)
        ).item()
        
        predictions = all_logits.argmax(axis=1)
        accuracy = accuracy_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions)
        
        return loss, accuracy, f1, predictions, all_logits
    
    def train(self, train_embeddings, train_labels, val_embeddings, val_labels,
              epochs=100, batch_size=32, patience=15):
        """
        训练循环
        
        Args:
            train_embeddings: 训练embedding数据
            train_labels: 训练标签
            val_embeddings: 验证embedding数据
            val_labels: 验证标签
            epochs: epoch数
            batch_size: 批大小
            patience: early stopping patience
        """
        print("\n" + "="*80)
        print("开始训练")
        print("="*80 + "\n")
        
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_embeddings, train_labels, batch_size)
            
            # 验证
            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_embeddings, val_labels, batch_size)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                self.save_checkpoint(epoch)
                print(f"  ✓ 模型已保存 (F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered (patience={patience})")
                    break
        
        print("\n✓ 训练完成！")
    
    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save(self.classifier.state_dict(), 
                  os.path.join(checkpoint_dir, 'classifier.pt'))
        
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'epoch': epoch,
        }
        with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_checkpoint(self, checkpoint_dir):
        """加载检查点"""
        self.classifier.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'classifier.pt'))
        )
        print(f"✓ 模型已加载: {checkpoint_dir}")
    
    def evaluate_on_test(self, test_embeddings, test_labels, test_texts):
        """在测试集上评估"""
        print("\n" + "="*80)
        print("测试集评估")
        print("="*80 + "\n")
        
        test_loss, test_acc, test_f1, predictions, logits = self.evaluate(
            test_embeddings, test_labels
        )
        
        # 计算更多指标
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        auc_score = roc_auc_score(test_labels, logits[:, 1])
        
        # 分类报告
        print("分类报告:")
        print(classification_report(test_labels, predictions, 
                                   target_names=['Non-Radicalization', 'Radicalization']))
        
        print("\n综合指标:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {test_f1:.4f}")
        print(f"  ROC-AUC:   {auc_score:.4f}")
        
        results = {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': test_f1,
            'auc': auc_score,
            'loss': test_loss,
            'predictions': predictions,
            'probabilities': logits[:, 1],
            'confusion_matrix': confusion_matrix(test_labels, predictions),
            'roc_curve': roc_curve(test_labels, logits[:, 1]),
        }
        
        return results
    
    def visualize_results(self, results, save_dir='./embeddings_finetuned'):
        """可视化训练结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 训练曲线
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Training Loss', fontweight='bold', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', marker='s')
        axes[1].plot(self.history['val_f1'], label='Val F1', marker='^')
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Score', fontweight='bold')
        axes[1].set_title('Training Metrics', fontweight='bold', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=150, bbox_inches='tight')
        print(f"✓ 保存: training_history.png")
        plt.close()
        
        # 2. 混淆矩阵
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Non-Radical', 'Radical'],
                   yticklabels=['Non-Radical', 'Radical'],
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('True', fontweight='bold')
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f"✓ 保存: confusion_matrix.png")
        plt.close()
        
        # 3. ROC曲线
        fpr, tpr, _ = results['roc_curve']
        roc_auc = results['auc']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curve', fontweight='bold', fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
        print(f"✓ 保存: roc_curve.png")
        plt.close()
    
    def save_results(self, results, test_data, save_dir='./embeddings_finetuned'):
        """保存结果报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        report_path = os.path.join(save_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EMBEDDING CLASSIFIER - TEST RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. MODEL CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Base Model: {self.model_name}\n")
            f.write(f"Embedding Dimension: {self.embedding_dim}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("2. PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall:    {results['recall']:.4f}\n")
            f.write(f"F1-Score:  {results['f1']:.4f}\n")
            f.write(f"ROC-AUC:   {results['auc']:.4f}\n\n")
            
            f.write("3. CONFUSION MATRIX\n")
            f.write("-"*80 + "\n")
            cm = results['confusion_matrix']
            f.write(f"TN: {cm[0,0]}, FP: {cm[0,1]}\n")
            f.write(f"FN: {cm[1,0]}, TP: {cm[1,1]}\n\n")
            
            f.write("4. PREDICTIONS SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Samples: {len(test_data)}\n")
            f.write(f"Predicted as Non-Radical: {(results['predictions']==0).sum()}\n")
            f.write(f"Predicted as Radical: {(results['predictions']==1).sum()}\n")
        
        print(f"✓ 报告已保存: {report_path}")


def main():
    """主训练流程"""
    from dataset import RadicalisationDataset
    
    print("="*80)
    print("TEXT EMBEDDING CLASSIFIER - FINE-TUNING")
    print("="*80 + "\n")
    
    # 1. 加载数据
    print("STEP 1: 数据加载")
    print("-"*80)
    root_dir = r"c:\Users\shanghong.li\Desktop\AI for radicalisation\Fighter and sympathiser"
    dataset = RadicalisationDataset(root_dir)
    
    coded_dataset = CodedDataset(dataset=dataset)
    stats = coded_dataset.get_statistics()
    
    print(f"总posts: {stats['total']:,}")
    print(f"  Coded=0 (Non-Radical): {stats['coded_0']:,} ({stats['coded_0']/stats['total']*100:.1f}%)")
    print(f"  Coded=1 (Radical):     {stats['coded_1']:,} ({stats['coded_1']/stats['total']*100:.1f}%)")
    print(f"平均文本长度: {stats['avg_text_length']:.0f} 字符")
    print(f"Fighters: {stats['fighters']:,}, Sympathisers: {stats['sympathisers']:,}\n")
    
    # 2. 分割数据
    print("STEP 2: 数据分割")
    print("-"*80)
    train_data, val_data, test_data = coded_dataset.split_data(
        val_size=0.1, test_size=0.1, stratify_by='coded'
    )
    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}\n")
    
    # 3. 初始化finetune器
    print("STEP 3: 初始化Embedding Model")
    print("-"*80)
    finetuner = EmbeddingFineTuner(
        model_name='all-MiniLM-L6-v2',  # 轻量级且高效的模型
        output_dir='./embeddings_finetuned'
    )
    print()
    
    # 4. 计算embeddings
    print("STEP 4: 计算Text Embeddings")
    print("-"*80)
    train_embeddings = finetuner.get_embeddings(train_data['text'].tolist())
    val_embeddings = finetuner.get_embeddings(val_data['text'].tolist())
    test_embeddings = finetuner.get_embeddings(test_data['text'].tolist())
    
    train_labels = train_data['coded'].values
    val_labels = val_data['coded'].values
    test_labels = test_data['coded'].values
    print()
    
    # 5. 创建和训练分类器
    print("STEP 5: 创建分类器")
    print("-"*80)
    finetuner.create_classifier(num_classes=2, dropout=0.2)
    finetuner.setup_optimizer(learning_rate=1e-3, weight_decay=0.01)
    print()
    
    # 6. 训练
    print("STEP 6: 训练")
    print("-"*80)
    finetuner.train(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        epochs=200,
        batch_size=32,
        patience=150
    )
    
    # 7. 加载最佳模型
    checkpoint_dir = os.path.join(finetuner.output_dir, 'checkpoint_epoch_0')
    if os.path.exists(checkpoint_dir):
        finetuner.load_checkpoint(checkpoint_dir)
    
    # 8. 测试评估
    results = finetuner.evaluate_on_test(test_embeddings, test_labels, test_data)
    
    # 9. 可视化
    print("\nSTEP 7: 可视化结果")
    print("-"*80)
    finetuner.visualize_results(results)
    
    # 10. 保存结果
    print("\nSTEP 8: 保存结果")
    print("-"*80)
    finetuner.save_results(results, test_data)
    
    print("\n" + "="*80)
    print("✓ 训练完成！")
    print(f"输出目录: {finetuner.output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
