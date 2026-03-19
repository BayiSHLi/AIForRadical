"""
Data Generator (Float16 版本) - 不使用量化以避免数值不稳定
"""

import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import jsonlines

from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_builder import build_fp16_instruction_prompt
from simulator_config import (
    MODEL_CONFIG, 
    GENERATION_PARAMS, 
    RADICALITY_LEVELS, 
    INDICATORS,
    OUTPUT_DIR,
    SAMPLE_FILE,
)

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataGeneratorFP16:
    """LLM 数据生成器 (Float16 版本，不使用量化)"""
    
    def __init__(self, model_name: str = None):
        """
        初始化生成器
        
        Args:
            model_name: HuggingFace 模型名称
        """
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🚀 初始化模型: {self.model_name}")
        logger.info(f"📊 设备: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"💾 GPU 0: {torch.cuda.get_device_name(0)}")
            logger.info(f"📈 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 加载 tokenizer
        logger.info("加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.tokenizer.padding_side = "left"
        logger.info("✓ Tokenizer 加载完成")
        
        # 加载模型 (float16, 不量化)
        logger.info("加载模型 (float16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        logger.info("✓ 模型加载完成")
    
    def build_prompt(self, indicator: str, radicality: str) -> str:
        """构建生成样本的 prompt"""
        if indicator not in INDICATORS:
            raise ValueError(f"Unknown indicator: {indicator}")
        if radicality not in RADICALITY_LEVELS:
            raise ValueError(f"Unknown radicality: {radicality}")
        
        ind_config = INDICATORS[indicator]
        rad_config = RADICALITY_LEVELS[radicality]

        return build_fp16_instruction_prompt(
            indicator=indicator,
            radicality=radicality,
            ind_config=ind_config,
            rad_config=rad_config,
        )
    
    def generate_sample(self, indicator: str, radicality: str, sample_id: int) -> Dict:
        """生成单个样本"""
        prompt = self.build_prompt(indicator, radicality)
        
        # 分词
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成（使用贪心解码 - 最稳定）
        try:
            logger.info(f"  生成样本 {sample_id}...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=50,  # 减少到 50
                    min_new_tokens=5,
                    do_sample=False,  # 贪心解码最稳定
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                )
            
            # 解码
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分
            content = generated_text[len(prompt):].strip()
            
            if len(content) < 10:
                content = "Sample content generated for testing purposes."
                logger.warning(f"  生成内容太短，使用占位文本")
            
            logger.info(f"  ✓ 样本 {sample_id}: {content[:50]}...")
            
            return {
                "ID": sample_id,
                "indicator": indicator,
                "Radicality": radicality,
                "Content": content,
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"  ✗ 样本 {sample_id} 生成失败: {str(e)}")
            raise
    
    def generate_batch(
        self, 
        indicator: str, 
        radicality: str, 
        count: int = 5,
        start_id: int = 1
    ) -> List[Dict]:
        """批量生成样本"""
        logger.info(f"📝 开始生成 {count} 个样本")
        logger.info(f"   Indicator: {indicator}")
        logger.info(f"   Radicality: {radicality}")
        
        samples = []
        for i in range(count):
            try:
                sample = self.generate_sample(indicator, radicality, start_id + i)
                samples.append(sample)
            except Exception as e:
                logger.error(f"样本 {i+1} 生成失败，跳过")
        
        logger.info(f"✓ 成功生成 {len(samples)}/{count} 个样本")
        return samples
    
    def save_samples(self, samples: List[Dict], output_path: Optional[str] = None):
        """保存样本到 JSONL 文件"""
        if output_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, SAMPLE_FILE)
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with jsonlines.open(output_path, mode='w') as writer:
            for sample in samples:
                writer.write(sample)
        
        logger.info(f"💾 {len(samples)} 个样本已保存到 {output_path}")


def main():
    """主函数：进行测试生成"""
    
    # 初始化生成器
    generator = DataGeneratorFP16()
    
    # 测试参数
    TEST_INDICATOR = "individual_loss_interpersonal"
    TEST_RADICALITY = "Low"
    SAMPLE_COUNT = 5
    
    logger.info(f"\n🧪 开始测试:")
    logger.info(f"   Indicator: {TEST_INDICATOR}")
    logger.info(f"   Radicality: {TEST_RADICALITY}")
    logger.info(f"   生成数量: {SAMPLE_COUNT}")
    logger.info("")
    
    # 生成样本
    samples = generator.generate_batch(
        indicator=TEST_INDICATOR,
        radicality=TEST_RADICALITY,
        count=SAMPLE_COUNT,
    )
    
    # 保存样本
    if samples:
        generator.save_samples(samples)
        
        # 输出样本
        logger.info("\n" + "="*80)
        logger.info("📋 生成的样本列表:")
        logger.info("="*80)
        for sample in samples:
            print(f"\nID: {sample['ID']}")
            print(f"Indicator: {sample['indicator']}")
            print(f"Radicality: {sample['Radicality']}")
            print(f"Content: {sample['Content']}")
        
        logger.info("\n✅ 测试完成!")
    else:
        logger.error("\n❌ 没有成功生成任何样本")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
