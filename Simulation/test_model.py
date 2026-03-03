"""
简单测试脚本 - 使用贪心解码测试模型是否正常工作
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    """测试模型基本功能"""
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    logger.info(f"加载模型: {model_name}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # 加载模型（8-bit 量化）
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=200.0,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    logger.info("✓ 模型加载完成")
    
    # 测试 prompt
    prompt = "[INST] Write a short social media post about losing friends due to disagreements. [/INST]\n\n"
    
    logger.info(f"测试 prompt: {prompt[:50]}...")
    
    # 分词
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    logger.info("✓ 分词完成")
    
    # 测试 1: 贪心解码（最安全）
    logger.info("\n测试 1: 贪心解码（do_sample=False）")
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✓ 生成成功!")
        logger.info(f"结果: {generated_text[len(prompt):]}")
        
    except Exception as e:
        logger.error(f"✗ 贪心解码失败: {str(e)}")
        return False
    
    # 测试 2: 低温度采样
    logger.info("\n测试 2: 低温度采样（temperature=0.3）")
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=50,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✓ 生成成功!")
        logger.info(f"结果: {generated_text[len(prompt):]}")
        
    except Exception as e:
        logger.error(f"✗ 低温度采样失败: {str(e)}")
        logger.warning("如果这步失败，说明采样存在问题")
        return False
    
    # 测试 3: 正常温度采样
    logger.info("\n测试 3: 正常温度采样（temperature=0.8）")
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✓ 生成成功!")
        logger.info(f"结果: {generated_text[len(prompt):]}")
        
    except Exception as e:
        logger.error(f"✗ 正常温度采样失败: {str(e)}")
        return False
    
    logger.info("\n✅ 所有测试通过!")
    return True


if __name__ == "__main__":
    success = test_model()
    exit(0 if success else 1)
