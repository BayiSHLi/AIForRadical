"""
Data Generator (Ollama with llama_index)
使用 llama_index 的 Ollama 集成加载 qwen2.5:7b
"""

import os
import logging
import random
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import jsonlines

from llama_index.llms.ollama import Ollama
from full_indicators import FULL_INDICATORS
from prompt_builder import (
    RADICALITY_PROGRESS_PRIOR,
    build_ollama_generation_prompt,
    sample_diversity_profile,
)
try:
    from external_retrieval import ExternalDataRetriever
except ImportError:
    from Simulation.external_retrieval import ExternalDataRetriever
from simulator_config import (
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

class DataGeneratorOllama:
    """使用 llama_index Ollama 的数据生成器"""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        temperature: float = 0.8,
        use_external_retrieval: bool = True,
        external_index_dir: Optional[str] = None,
        external_top_k: int = 3,
    ):
        """
        初始化生成器
        
        Args:
            model_name: Ollama 模型名称
            temperature: 生成温度
        """
        self.model_name = model_name
        self.temperature = temperature
        self.indicator_catalog = self._build_indicator_catalog()
        self.radicality_order = ["Neutral", "Low", "Medium", "High"]
        self.use_external_retrieval = use_external_retrieval
        self.external_top_k = external_top_k
        self.external_retriever: Optional[ExternalDataRetriever] = None
        
        logger.info(f"🚀 初始化 Ollama 生成器")
        logger.info(f"📊 模型: {self.model_name}")
        logger.info(f"🌡️  温度: {self.temperature}")
        
        # 使用 llama_index 的 Ollama 集成
        try:
            self.llm = Ollama(model=self.model_name, temperature=self.temperature)
            logger.info("✓ LLM 初始化成功")
        except Exception as e:
            logger.error(f"❌ 无法初始化 LLM: {e}")
            logger.error("请确保 Ollama 正在运行并且模型已下载")
            raise

        if self.use_external_retrieval:
            try:
                self.external_retriever = ExternalDataRetriever(
                    index_dir=external_index_dir,
                    similarity_top_k=self.external_top_k,
                )
                if self.external_retriever.is_available:
                    logger.info("✓ 外部数据检索接口已启用")
                else:
                    logger.info("ℹ 外部数据索引未就绪，生成将不使用外部检索上下文")
            except Exception as e:
                logger.warning(f"⚠ 初始化外部数据检索失败: {e}")
                self.external_retriever = None

    def _build_indicator_catalog(self) -> Dict[str, Dict]:
        """合并 full_indicators 与 simulator_config 中的扩展字段。"""
        catalog = {k: dict(v) for k, v in FULL_INDICATORS.items()}
        for key, value in INDICATORS.items():
            if key not in catalog:
                catalog[key] = dict(value)
            else:
                catalog[key].update(value)
        logger.info(f"✓ 已加载 indicator 数量: {len(catalog)}")
        return catalog
    
    def build_prompt(
        self,
        indicator: str,
        radicality: str,
        style_hint: str,
        banned_snippets: List[str],
        diversity_profile: Optional[Dict[str, str]] = None,
        external_context: str = "",
    ) -> str:
        """构建生成样本的 prompt"""
        if indicator not in self.indicator_catalog:
            raise ValueError(f"Unknown indicator: {indicator}")
        if radicality not in RADICALITY_LEVELS:
            raise ValueError(f"Unknown radicality: {radicality}")

        return build_ollama_generation_prompt(
            indicator=indicator,
            radicality=radicality,
            ind_config=self.indicator_catalog[indicator],
            rad_config=RADICALITY_LEVELS[radicality],
            style_hint=style_hint,
            banned_snippets=banned_snippets,
            diversity_profile=diversity_profile,
            external_context=external_context,
        )

    def _retrieve_external_context(self, indicator: str, radicality: str) -> str:
        """Retrieve external dataset snippets for grounding prompt style."""
        if not self.use_external_retrieval:
            return ""
        if self.external_retriever is None or not self.external_retriever.is_available:
            return ""

        indicator_desc = self.indicator_catalog.get(indicator, {}).get("description", "")
        query_text = self.external_retriever.build_query(
            indicator=indicator,
            radicality=radicality,
            indicator_description=indicator_desc,
        )
        results = self.external_retriever.retrieve(query_text=query_text, top_k=self.external_top_k)
        return self.external_retriever.format_for_prompt(results=results, max_items=self.external_top_k)
    
    def _normalize_for_dedup(self, text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"\s+", " ", lowered)
        lowered = re.sub(r"[^\w\s]", "", lowered)
        return lowered

    def _get_default_dimension_scores(self, radicality: str) -> Dict[str, float]:
        """Use configured priors as fallback schema scores."""
        prior = RADICALITY_PROGRESS_PRIOR.get(radicality, RADICALITY_PROGRESS_PRIOR["Low"])
        return {
            "opinion": float(prior.get("opinion", 0.0)),
            "radicalization": float(prior.get("radicalization", 0.0)),
            "mobilization": float(prior.get("mobilization", 0.0)),
        }

    def _clamp_score(self, value: object, fallback: float) -> float:
        """Convert score to float and clamp into [0,1]."""
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = fallback
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    def _extract_json_payload(self, raw_text: str) -> Optional[Dict]:
        """Try multiple strategies to parse one JSON object from model output."""
        candidates = [raw_text.strip()]

        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw_text, flags=re.IGNORECASE)
        candidates.extend(chunk.strip() for chunk in fenced if chunk.strip())

        brace_match = re.search(r"\{[\s\S]*\}", raw_text)
        if brace_match:
            candidates.append(brace_match.group(0).strip())

        for candidate in candidates:
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue
        return None

    def _parse_generated_payload(
        self,
        raw_output: str,
        indicator: str,
        radicality: str,
        sample_id: int,
    ) -> Tuple[str, Dict]:
        """Parse structured JSON output; fallback to plain-text extraction."""
        default_scores = self._get_default_dimension_scores(radicality)
        payload = self._extract_json_payload(raw_output)

        if payload:
            text_candidate = payload.get("text") or payload.get("content") or payload.get("post")
            content = self._clean_content(str(text_candidate or ""))

            raw_scores = payload.get("dimension_scores", {})
            if not isinstance(raw_scores, dict):
                raw_scores = {}
            dimension_scores = {
                "opinion": self._clamp_score(raw_scores.get("opinion"), default_scores["opinion"]),
                "radicalization": self._clamp_score(
                    raw_scores.get("radicalization"), default_scores["radicalization"]
                ),
                "mobilization": self._clamp_score(
                    raw_scores.get("mobilization"), default_scores["mobilization"]
                ),
            }

            progression_meta = payload.get("progression_meta", {})
            if not isinstance(progression_meta, dict):
                progression_meta = {}
            progression_meta.setdefault("target_radicality", radicality)
            progression_meta.setdefault("target_prior", default_scores)
            progression_meta.setdefault("schema_version", "prmm_v1")

            indicator_vector = payload.get("indicator_vector_79", {})
            if not isinstance(indicator_vector, dict):
                indicator_vector = {}
            normalized_vector: Dict[str, float] = {}
            for key, value in list(indicator_vector.items())[:20]:
                key_text = str(key).strip()
                if not key_text:
                    continue
                base = 1.0 if key_text == indicator else 0.0
                normalized_vector[key_text] = self._clamp_score(value, base)
            if indicator not in normalized_vector:
                normalized_vector[indicator] = 1.0

            reasoning = str(payload.get("reasoning", "")).strip()
            if not reasoning:
                reasoning = f"Aligned with {radicality} progression target for this indicator."

            parsed_sample_id = payload.get("sample_id")
            if not isinstance(parsed_sample_id, int):
                parsed_sample_id = sample_id

            source = str(payload.get("source", "simulation_ollama")).strip() or "simulation_ollama"

            return content, {
                "sample_id": parsed_sample_id,
                "text": content,
                "dimension_scores": dimension_scores,
                "progression_meta": progression_meta,
                "indicator_vector_79": normalized_vector,
                "reasoning": reasoning[:240],
                "source": source[:80],
            }

        content = self._clean_content(raw_output)
        return content, {
            "sample_id": sample_id,
            "text": content,
            "dimension_scores": default_scores,
            "progression_meta": {
                "target_radicality": radicality,
                "target_prior": default_scores,
                "schema_version": "prmm_v1",
                "parse_mode": "plain_text_fallback",
            },
            "indicator_vector_79": {indicator: 1.0},
            "reasoning": "Fallback extraction from plain text response.",
            "source": "simulation_ollama",
        }

    def generate_sample(
        self,
        indicator: str,
        radicality: str,
        sample_id: int,
        existing_contents: List[str],
    ) -> Dict:
        """生成单个样本"""
        logger.info(f"  生成样本 {sample_id}...")
        seen = {self._normalize_for_dedup(x) for x in existing_contents}
        external_context = self._retrieve_external_context(indicator=indicator, radicality=radicality)
        
        for attempt in range(1, 6):
            diversity_profile = sample_diversity_profile()
            style_hint = diversity_profile["style"]
            banned = random.sample(existing_contents, k=min(len(existing_contents), 6)) if existing_contents else []
            prompt = self.build_prompt(
                indicator,
                radicality,
                style_hint,
                banned,
                diversity_profile=diversity_profile,
                external_context=external_context,
            )

            try:
                temperature = min(1.1, self.temperature + (attempt - 1) * 0.05)
                llm = Ollama(model=self.model_name, temperature=temperature)
                response = llm.complete(prompt)

                if hasattr(response, "text"):
                    raw_output = response.text.strip()
                else:
                    raw_output = str(response).strip()

                content, schema_fields = self._parse_generated_payload(
                    raw_output=raw_output,
                    indicator=indicator,
                    radicality=radicality,
                    sample_id=sample_id,
                )
                normalized = self._normalize_for_dedup(content)

                if len(content) < 20:
                    logger.warning(f"  重试#{attempt}: 内容过短")
                    continue
                if normalized in seen:
                    logger.warning(f"  重试#{attempt}: 检测到重复")
                    continue

                logger.info(f"  ✓ 样本 {sample_id}: {content[:60]}...")
                timestamp = datetime.now().isoformat()
                return {
                    "ID": sample_id,
                    "indicator": indicator,
                    "Radicality": radicality,
                    "Content": content,
                    "timestamp": timestamp,
                    "sample_id": schema_fields["sample_id"],
                    "text": schema_fields["text"],
                    "dimension_scores": schema_fields["dimension_scores"],
                    "progression_meta": schema_fields["progression_meta"],
                    "indicator_vector_79": schema_fields["indicator_vector_79"],
                    "reasoning": schema_fields["reasoning"],
                    "source": schema_fields["source"],
                }
            except Exception as e:
                logger.warning(f"  重试#{attempt} 失败: {str(e)}")

        content = f"Post about {indicator} at {radicality} level with varied phrasing."
        default_scores = self._get_default_dimension_scores(radicality)
        logger.warning("  使用后备内容（多次尝试后仍失败）")
        return {
            "ID": sample_id,
            "indicator": indicator,
            "Radicality": radicality,
            "Content": content,
            "timestamp": datetime.now().isoformat(),
            "sample_id": sample_id,
            "text": content,
            "dimension_scores": default_scores,
            "progression_meta": {
                "target_radicality": radicality,
                "target_prior": default_scores,
                "schema_version": "prmm_v1",
                "parse_mode": "generation_fallback",
            },
            "indicator_vector_79": {indicator: 1.0},
            "reasoning": "Fallback content after repeated generation failures.",
            "source": "simulation_ollama",
        }
    
    def _clean_content(self, content: str) -> str:
        """清理生成的内容"""
        # 移除可能的引号
        content = content.strip('"\'')
        
        # 移除 "Post:" 等前缀
        prefixes = ["Post:", "post:", "Content:", "content:", "Tweet:", "tweet:"]
        for prefix in prefixes:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        
        # 只取第一行或第一段（如果有多行）
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines:
            content = lines[0]

        # 去掉外层括号或无关前后缀
        content = content.strip("-• ")
        
        # 限制长度
        if len(content) > 300:
            content = content[:300].strip()
        
        return content
    
    def generate_batch(
        self, 
        indicator: str, 
        radicality: str, 
        count: int = 20,
        start_id: int = 1
    ) -> List[Dict]:
        """批量生成样本"""
        logger.info(f"\n📝 开始生成 {count} 个样本")
        logger.info(f"   Indicator: {indicator}")
        logger.info(f"   Radicality: {radicality}\n")
        
        samples = []
        for i in range(count):
            try:
                existing_contents = [x["Content"] for x in samples]
                sample = self.generate_sample(
                    indicator=indicator,
                    radicality=radicality,
                    sample_id=start_id + i,
                    existing_contents=existing_contents,
                )
                samples.append(sample)
            except Exception as e:
                logger.error(f"  样本 {i+1} 生成失败: {str(e)}")
                logger.info("  跳过并继续...")
        
        logger.info(f"\n✓ 成功生成 {len(samples)}/{count} 个样本")
        return samples
    
    def save_samples(self, samples: List[Dict], output_path: Optional[str] = None, mode: str = "w"):
        """保存样本到 JSONL 文件"""
        if output_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, SAMPLE_FILE)
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with jsonlines.open(output_path, mode=mode) as writer:
            for sample in samples:
                writer.write(sample)
        
        logger.info(f"💾 {len(samples)} 个样本已保存到 {output_path}")

    def generate_full_matrix(
        self,
        count_per_pair: int = 20,
        output_file: str = "samples_79x4x20.jsonl",
    ) -> Tuple[int, int]:
        """生成全部 79 indicators × 4 radicality × count_per_pair。"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, output_file)
        if os.path.exists(output_path):
            os.remove(output_path)

        all_indicator_ids = list(self.indicator_catalog.keys())
        total_pairs = len(all_indicator_ids) * len(self.radicality_order)
        total_expected = total_pairs * count_per_pair
        logger.info(f"\n🚀 开始全量生成: indicators={len(all_indicator_ids)}, radicality=4, 每组={count_per_pair}")
        logger.info(f"🎯 目标样本总数: {total_expected}")

        global_id = 1
        generated_total = 0
        pair_idx = 0

        for indicator in all_indicator_ids:
            for radicality in self.radicality_order:
                pair_idx += 1
                logger.info(f"\n[{pair_idx}/{total_pairs}] {indicator} | {radicality}")
                batch = self.generate_batch(
                    indicator=indicator,
                    radicality=radicality,
                    count=count_per_pair,
                    start_id=global_id,
                )
                global_id += count_per_pair
                generated_total += len(batch)
                self.save_samples(batch, output_path=output_path, mode="a")

                logger.info(
                    f"📈 累计进度: {generated_total}/{total_expected} "
                    f"({(generated_total / total_expected) * 100:.2f}%)"
                )

        logger.info("\n✅ 全量生成完成")
        logger.info(f"📁 输出文件: {output_path}")
        logger.info(f"📊 最终样本数: {generated_total}/{total_expected}")
        return generated_total, total_expected


def main():
    """主函数：全量小规模测试（79x4x20）"""
    
    # 初始化生成器
    try:
        generator = DataGeneratorOllama(
            model_name="qwen2.5:7b",
            temperature=0.8
        )
    except Exception as e:
        logger.error("\n无法初始化生成器")
        logger.error("请确保:")
        logger.error("  1. Ollama 正在运行 (ollama serve)")
        logger.error("  2. 模型已下载 (ollama pull qwen2.5:7b)")
        return 1
    
    # 全量小规模测试配置：79 x 4 x 20 = 6320
    COUNT_PER_PAIR = 20
    OUTPUT_FILE = "samples_79x4x20.jsonl"

    logger.info(f"\n{'='*80}")
    logger.info("🧪 小规模全量测试配置")
    logger.info(f"   Indicators: {len(generator.indicator_catalog)}")
    logger.info(f"   Radicality levels: {len(generator.radicality_order)}")
    logger.info(f"   每组样本: {COUNT_PER_PAIR}")
    logger.info(f"   目标总数: {len(generator.indicator_catalog) * len(generator.radicality_order) * COUNT_PER_PAIR}")
    logger.info(f"{'='*80}")

    generated, expected = generator.generate_full_matrix(
        count_per_pair=COUNT_PER_PAIR,
        output_file=OUTPUT_FILE,
    )

    if generated > 0:
        logger.info("\n✅ 测试完成")
        logger.info(f"📁 样本文件: {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
        logger.info(f"📊 完成率: {(generated / expected) * 100:.2f}%")
        return 0 if generated == expected else 2

    logger.error("\n❌ 没有成功生成任何样本")
    return 1


if __name__ == "__main__":
    exit(main())
