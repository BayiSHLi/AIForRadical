"""
Data Generator - 样本生成器
使用 HuggingFace 的 LLM 生成符合 indicator 和 radicality 的文本样本
"""

import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import jsonlines

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

class DataGenerator:
    """LLM 数据生成器"""
    
    def __init__(self, model_name: str = None, use_quantization: bool = True):
        """
        初始化生成器
        
        Args:
            model_name: HuggingFace 模型名称
            use_quantization: 是否使用 8-bit 量化以节省显存
        """
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.use_quantization = use_quantization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🚀 初始化模型: {self.model_name}")
        logger.info(f"📊 设备: {self.device}, 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,  # 使用慢速 tokenizer 避免某些问题
        )
        
        # 正确设置 pad_token（对于 Mistral 模型很重要）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 确保 padding 在左侧（对于生成任务）
        self.tokenizer.padding_side = "left"
        
        if self.use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=200.0,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            ).to(self.device)
        
        self.model.eval()
        logger.info("✓ 模型加载完成")
    
    def build_prompt(self, indicator: str, radicality: str) -> str:
        """
        构建生成样本的 prompt
        
        Args:
            indicator: indicator 名称
            radicality: radicality 等级
            
        Returns:
            构建的 prompt
        """
        if indicator not in INDICATORS:
            raise ValueError(f"Unknown indicator: {indicator}")
        if radicality not in RADICALITY_LEVELS:
            raise ValueError(f"Unknown radicality: {radicality}")
        
        ind_config = INDICATORS[indicator]
        rad_config = RADICALITY_LEVELS[radicality]
        
        # 使用更简洁的 prompt 模板（适配 Mistral）
        prompt = f"""[INST] You are a social media content generator. Generate a realistic social media post (tweet, Facebook comment, etc.) based on the following:

Topic: {ind_config['description']}
Intensity Level: {radicality} - {rad_config['description'][:100]}

Requirements:
- Write 1-2 sentences (30-150 characters)
- Sound natural and authentic
- Match the topic and intensity level

Example style: {ind_config['example_content'][:80]}...

Generate the post: [/INST]

"""
        return prompt
    
    def generate_sample(self, indicator: str, radicality: str, sample_id: int) -> Dict:
        """
        生成单个样本
        
        Args:
            indicator: indicator 名称
            radicality: radicality 等级
            sample_id: 样本 ID
            
        Returns:
            包含生成内容的字典
        """
        prompt = self.build_prompt(indicator, radicality)
        
        # 分词 - 使用更安全的方式
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # 限制输入长度
        )
        
        # 将输入移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=min(GENERATION_PARAMS["max_length"], 100),
                    min_new_tokens=10,  # 确保至少生成一些内容
                    temperature=max(GENERATION_PARAMS["temperature"], 0.1),  # 避免温度过低
                    top_p=GENERATION_PARAMS["top_p"],
                    top_k=GENERATION_PARAMS["top_k"],
                    do_sample=True,
                    repetition_penalty=GENERATION_PARAMS.get("repetition_penalty", 1.1),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=None,  # 避免某些词导致的问题
                    num_return_sequences=1,
                )
        except RuntimeError as e:
            logger.error(f"生成时发生错误: {str(e)}")
            logger.info("尝试使用贪心解码（不采样）...")
            # 使用贪心解码作为后备方案
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=80,
                        min_new_tokens=10,
                        do_sample=False,  # 贪心解码
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_beams=1,
                    )
            except RuntimeError as e2:
                logger.error(f"贪心解码也失败: {str(e2)}")
                raise
        
        # 解码
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的部分（去掉prompt）
        content = generated_text[len(prompt):].strip()
        
        # 如果内容为空或太短，使用备用方案
        if len(content) < 10:
            content = f"[Generated sample for {indicator} at {radicality} level]"
            logger.warning(f"生成的内容太短，使用占位文本")
        
        return {
            "ID": sample_id,
            "indicator": indicator,
            "Radicality": radicality,
            "Content": content,
            "timestamp": datetime.now().isoformat(),
        }
    
    def generate_batch(
        self, 
        indicator: str, 
        radicality: str, 
        count: int = 5,
        start_id: int = 1
    ) -> List[Dict]:
        """
        批量生成样本
        
        Args:
            indicator: indicator 名称
            radicality: radicality 等级
            count: 生成数量
            start_id: 起始 ID
            
        Returns:
            生成的样本列表
        """
        logger.info(f"📝 开始生成 {count} 个样本 (indicator={indicator}, radicality={radicality})")
        
        samples = []
        for i in range(count):
            try:
                sample = self.generate_sample(indicator, radicality, start_id + i)
                samples.append(sample)
                logger.info(f"  ✓ 样本 {i+1}/{count}: {sample['Content'][:60]}...")
            except Exception as e:
                logger.error(f"  ✗ 样本 {i+1} 生成失败: {str(e)}")
        
        return samples
    
    def save_samples(self, samples: List[Dict], output_path: Optional[str] = None):
        """
        保存生成的样本到 JSONL 文件
        
        Args:
            samples: 样本列表
            output_path: 输出文件路径
        """
        if output_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, SAMPLE_FILE)
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with jsonlines.open(output_path, mode='a') as writer:
            for sample in samples:
                writer.write(sample)
        
        logger.info(f"✓ {len(samples)} 个样本已保存到 {output_path}")


def main():
    """主函数：进行测试生成"""
    
    # 初始化生成器
    generator = DataGenerator()
    
    # 测试参数：选择一个 indicator 和 radicality
    TEST_INDICATOR = "individual_loss_interpersonal"
    TEST_RADICALITY = "Low"
    SAMPLE_COUNT = 5
    
    logger.info(f"🧪 开始测试:")
    logger.info(f"   Indicator: {TEST_INDICATOR}")
    logger.info(f"   Radicality: {TEST_RADICALITY}")
    logger.info(f"   生成数量: {SAMPLE_COUNT}")
    
    # 生成样本
    samples = generator.generate_batch(
        indicator=TEST_INDICATOR,
        radicality=TEST_RADICALITY,
        count=SAMPLE_COUNT,
    )
    
    # 保存样本
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


if __name__ == "__main__":
    main()
