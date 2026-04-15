"""
Multi-GPU Data Generator with O/R/M Independent Dimensions
使用更强的LLM（qwen2.5:14b）和独立的O/R/M维度生成79*4*4*4*5个样本
支持双GPU并行生成以加速流程
"""

import os
import logging
import random
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
import jsonlines

from llama_index.llms.ollama import Ollama
from full_indicators import FULL_INDICATORS
from prompt_builder import (
    RADICALITY_PROGRESS_PRIOR,
    build_ollama_generation_prompt,
    sample_diversity_profile,
    _choose_indicator_anchors,
    _format_rule_block,
    _format_output_schema_block,
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
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ O/R/M 维度定义 ============
DIMENSION_LEVELS = {
    "opinion": ["Neutral", "Low", "Medium", "High"],
    "radicalization": ["Neutral", "Low", "Medium", "High"],
    "mobilization": ["Neutral", "Low", "Medium", "High"],
}


SAMPLES_PER_COMBINATION = 5
SAMPLES_PER_INDICATOR = (
    len(DIMENSION_LEVELS["opinion"])
    * len(DIMENSION_LEVELS["radicalization"])
    * len(DIMENSION_LEVELS["mobilization"])
    * SAMPLES_PER_COMBINATION
)


class MultiDimensionOllamaGenerator:
    """支持O/R/M独立维度的多GPU数据生成器"""

    def __init__(
        self,
        model_name: str = "qwen2.5:14b",
        temperature: float = 0.8,
        gpu_id: int = 0,
        base_url: str = "http://127.0.0.1:11434",
        use_external_retrieval: bool = True,
        external_index_dir: Optional[str] = None,
        external_top_k: int = 3,
    ):
        """
        初始化生成器
        
        Args:
            model_name: Ollama 模型名称 (推荐使用 qwen2.5:14b 代替 7b)
            temperature: 生成温度
            gpu_id: GPU编号标识（用于日志和资源提示）
            base_url: Ollama 服务地址（用于绑定不同GPU实例）
            use_external_retrieval: 是否使用外部检索
            external_index_dir: 外部索引目录
            external_top_k: 外部检索top-k
        """
        self.model_name = model_name
        self.temperature = temperature
        self.gpu_id = gpu_id
        self.base_url = base_url
        self.indicator_catalog = self._build_indicator_catalog()
        self.use_external_retrieval = use_external_retrieval
        self.external_top_k = external_top_k
        self.external_retriever: Optional[ExternalDataRetriever] = None
        
        logger.info(f"🚀 初始化 Ollama 生成器 (GPU{gpu_id})")
        logger.info(f"📊 模型: {self.model_name}")
        logger.info(f"🌐 Ollama: {self.base_url}")
        logger.info(f"🌡️  温度: {self.temperature}")
        
        try:
            self.llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
            )
            logger.info(f"✓ LLM 初始化成功 (GPU{gpu_id})")
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
                    logger.info(f"✓ 外部数据检索接口已启用 (GPU{gpu_id})")
                else:
                    logger.info(f"ℹ 外部数据索引未就绪 (GPU{gpu_id})")
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
    
    def _normalize_for_dedup(self, text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"\s+", " ", lowered)
        lowered = re.sub(r"[^\w\s]", "", lowered)
        return lowered

    def _get_default_dimension_scores(
        self,
        opinion: str,
        radicalization: str, 
        mobilization: str
    ) -> Dict[str, float]:
        """根据三个维度的强度级别生成目标分数"""
        scores = {}
        
        # 从RADICALITY_PROGRESS_PRIOR中获取先验值
        opinion_prior = RADICALITY_PROGRESS_PRIOR.get(opinion, {}).get("opinion", 0.0)
        rad_prior = RADICALITY_PROGRESS_PRIOR.get(radicalization, {}).get("radicalization", 0.0)
        mob_prior = RADICALITY_PROGRESS_PRIOR.get(mobilization, {}).get("mobilization", 0.0)
        
        return {
            "opinion": float(opinion_prior),
            "radicalization": float(rad_prior),
            "mobilization": float(mob_prior),
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
        opinion: str,
        radicalization: str,
        mobilization: str,
        sample_id: int,
    ) -> Tuple[str, Dict]:
        """Parse structured JSON output; fallback to plain-text extraction."""
        default_scores = self._get_default_dimension_scores(opinion, radicalization, mobilization)
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
            progression_meta.setdefault("target_opinion", opinion)
            progression_meta.setdefault("target_radicalization", radicalization)
            progression_meta.setdefault("target_mobilization", mobilization)
            progression_meta.setdefault("target_prior", default_scores)
            progression_meta.setdefault("schema_version", "prmm_v1_multidim")

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
                reasoning = f"O={opinion} R={radicalization} M={mobilization}"

            parsed_sample_id = payload.get("sample_id")
            if not isinstance(parsed_sample_id, int):
                parsed_sample_id = sample_id

            source = str(payload.get("source", "simulation_ollama_multidim")).strip() or "simulation_ollama_multidim"

            return content, {
                "sample_id": parsed_sample_id,
                "text": content,
                "dimension_scores": dimension_scores,
                "progression_meta": progression_meta,
                "indicator_vector_79": normalized_vector,
                "reasoning": reasoning[:240],
                "source": source,
            }

        content = self._clean_content(raw_output)
        return content, {
            "sample_id": sample_id,
            "text": content,
            "dimension_scores": default_scores,
            "progression_meta": {
                "target_opinion": opinion,
                "target_radicalization": radicalization,
                "target_mobilization": mobilization,
                "target_prior": default_scores,
                "schema_version": "prmm_v1_multidim",
                "parse_mode": "plain_text_fallback",
            },
            "indicator_vector_79": {indicator: 1.0},
            "reasoning": "Fallback extraction from plain text response.",
            "source": "simulation_ollama_multidim",
        }

    def _build_dimension_prompt(
        self,
        indicator: str,
        opinion: str,
        radicalization: str,
        mobilization: str,
    ) -> str:
        """构建考虑O/R/M三维度的prompt"""
        if indicator not in self.indicator_catalog:
            raise ValueError(f"Unknown indicator: {indicator}")
        
        ind_config = self.indicator_catalog[indicator]
        
        # 使用radicalization作为主维度来获取基础配置
        rad_config = RADICALITY_LEVELS.get(radicalization, RADICALITY_LEVELS["Low"])
        
        indicator_anchors = _choose_indicator_anchors(ind_config)
        anchor_text = ", ".join(indicator_anchors) if indicator_anchors else "N/A"
        
        # 为三个维度分别构建规则块（虽然当前还是用了统一的，但结构已为独立维度准备好）
        default_prior = RADICALITY_PROGRESS_PRIOR.get(radicalization, RADICALITY_PROGRESS_PRIOR["Low"])
        progression_target = (
            f"O~{default_prior['opinion']:.2f} "
            f"R~{default_prior['radicalization']:.2f} "
            f"M~{default_prior['mobilization']:.2f}"
        )
        
        # 为O/R/M分别构建约束
        opinion_rules = _format_rule_block(opinion) if opinion else ""
        radicalization_rules = _format_rule_block(radicalization) if radicalization else ""
        mobilization_rules = _format_rule_block(mobilization) if mobilization else ""
        
        output_schema_block = _format_output_schema_block(indicator=indicator, radicality=radicalization)
        
        example_content = ind_config.get("example_content", "")
        keywords = rad_config.get("keywords", [])
        keyword_text = ", ".join(keywords[:4]) if keywords else "N/A"

        external_context = ""
        if self.use_external_retrieval and self.external_retriever and self.external_retriever.is_available:
            try:
                indicator_desc = ind_config.get("description", "")
                query_text = self.external_retriever.build_query(
                    indicator=indicator,
                    radicality=radicalization,
                    indicator_description=indicator_desc,
                )
                results = self.external_retriever.retrieve(query_text=query_text, top_k=self.external_top_k)
                external_context = self.external_retriever.format_for_prompt(results=results, max_items=self.external_top_k)
            except Exception:
                external_context = ""

        diversity_profile = sample_diversity_profile()
        length_instruction = diversity_profile.get(
            "length_instruction", "1-2 sentences, 35-180 characters"
        )
        voice_hint = diversity_profile.get("voice", "first-person singular (I/me)")
        tone_hint = diversity_profile.get("tone", "calm and restrained")
        structure_hint = diversity_profile.get("structure", "single concise statement")
        punctuation_hint = diversity_profile.get("punctuation", "minimal punctuation, plain style")
        lexicon_hint = diversity_profile.get("lexicon", "mixed colloquial and formal words")
        opening_hint = diversity_profile.get("opening", "begin directly with a personal observation")

        return f"""You generate one synthetic social media post for multi-dimensional radicalization research.

Indicator ID: {indicator}
Indicator Factor: {ind_config.get('factor', 'N/A')}
Indicator Meaning: {ind_config.get('description', 'N/A')}
Indicator anchors: {anchor_text}

TARGET DIMENSIONS (each 0-1 scale):
- Opinion level: {opinion}
- Radicalization level: {radicalization}
- Mobilization level: {mobilization}
Progression prior (internal): {progression_target}

DIMENSIONAL CONSTRAINTS:

{opinion_rules}

{radicalization_rules}

{mobilization_rules}

Style requirement: {diversity_profile.get("style", "first_person_reflection")}
Narrative voice: {voice_hint}
Tone target: {tone_hint}
Sentence structure: {structure_hint}
Punctuation style: {punctuation_hint}
Lexical style: {lexicon_hint}
Opening strategy: {opening_hint}
Language policy: mainly English, allow occasional short internet slang, no hashtags unless natural.
Length target: {length_instruction}.

Diversity constraints:
1) Use a fresh wording pattern and sentence structure.
2) Keep it natural, specific, and varied.
3) Avoid repetitive openers.
4) Keep exactly one coherent post, no lists, no role-play tags.

Reference style sample (not to copy): {example_content[:140] if example_content else 'N/A'}

External dataset references (for realism and grounding):
{external_context if external_context else '(No external dataset references retrieved)'}

{output_schema_block}"""

    def generate_sample(
        self,
        indicator: str,
        opinion: str,
        radicalization: str,
        mobilization: str,
        sample_id: int,
        existing_contents: List[str],
    ) -> Dict:
        """生成单个样本"""
        seen = {self._normalize_for_dedup(x) for x in existing_contents}
        
        for attempt in range(1, 6):
            prompt = self._build_dimension_prompt(
                indicator=indicator,
                opinion=opinion,
                radicalization=radicalization,
                mobilization=mobilization,
            )

            try:
                temperature = min(1.1, self.temperature + (attempt - 1) * 0.05)
                llm = Ollama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=temperature,
                )
                response = llm.complete(prompt)

                if hasattr(response, "text"):
                    raw_output = response.text.strip()
                else:
                    raw_output = str(response).strip()

                content, schema_fields = self._parse_generated_payload(
                    raw_output=raw_output,
                    indicator=indicator,
                    opinion=opinion,
                    radicalization=radicalization,
                    mobilization=mobilization,
                    sample_id=sample_id,
                )
                normalized = self._normalize_for_dedup(content)

                if len(content) < 20:
                    continue
                if normalized in seen:
                    continue

                timestamp = datetime.now().isoformat()
                return {
                    "ID": sample_id,
                    "indicator": indicator,
                    "Opinion": opinion,
                    "Radicalization": radicalization,
                    "Mobilization": mobilization,
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
                logger.warning(f"  GPU{self.gpu_id} 重试#{attempt} 失败: {str(e)}")

        default_scores = self._get_default_dimension_scores(opinion, radicalization, mobilization)
        content = f"Post about {indicator} at O={opinion} R={radicalization} M={mobilization}."
        return {
            "ID": sample_id,
            "indicator": indicator,
            "Opinion": opinion,
            "Radicalization": radicalization,
            "Mobilization": mobilization,
            "Content": content,
            "timestamp": datetime.now().isoformat(),
            "sample_id": sample_id,
            "text": content,
            "dimension_scores": default_scores,
            "progression_meta": {
                "target_opinion": opinion,
                "target_radicalization": radicalization,
                "target_mobilization": mobilization,
                "target_prior": default_scores,
                "schema_version": "prmm_v1_multidim",
                "parse_mode": "generation_fallback",
            },
            "indicator_vector_79": {indicator: 1.0},
            "reasoning": "Fallback content after repeated generation failures.",
            "source": "simulation_ollama_multidim",
        }

    def _clean_content(self, content: str) -> str:
        """清理生成的内容"""
        content = content.strip('"\'')
        prefixes = ["Post:", "post:", "Content:", "content:", "Tweet:", "tweet:"]
        for prefix in prefixes:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines:
            content = lines[0]

        content = content.strip("-• ")
        
        if len(content) > 300:
            content = content[:300].strip()
        
        return content


def worker_generate_batch(
    gpu_id: int,
    indicator_list: List[Tuple[int, str]],
    sample_queue: Queue,
    model_name: str = "qwen2.5:14b",
    temperature: float = 0.8,
    base_url: str = "http://127.0.0.1:11434",
):
    """Worker process for GPU-specific generation"""
    logger.info(f"🎯 GPU{gpu_id} Worker started with {len(indicator_list)} indicators ({base_url})")
    
    generator = MultiDimensionOllamaGenerator(
        model_name=model_name,
        temperature=temperature,
        gpu_id=gpu_id,
        base_url=base_url,
    )
    
    for indicator_index, indicator in indicator_list:
        combo_index = 0
        for opinion in DIMENSION_LEVELS["opinion"]:
            for radicalization in DIMENSION_LEVELS["radicalization"]:
                for mobilization in DIMENSION_LEVELS["mobilization"]:
                    for _ in range(SAMPLES_PER_COMBINATION):
                        sample_id = indicator_index * SAMPLES_PER_INDICATOR + combo_index + 1
                        try:
                            sample = generator.generate_sample(
                                indicator=indicator,
                                opinion=opinion,
                                radicalization=radicalization,
                                mobilization=mobilization,
                                sample_id=sample_id,
                                existing_contents=[],
                            )
                            sample_queue.put(sample)
                            combo_index += 1
                        except Exception as e:
                            logger.error(f"GPU{gpu_id} 生成失败: {e}")
    
    logger.info(f"✓ GPU{gpu_id} Worker completed")


def main():
    """主函数：完整的多GPU并行生成"""
    model_name = "qwen2.5:14b"
    temperature = 0.8
    base_urls = [
        value.strip()
        for value in os.environ.get(
            "OLLAMA_BASE_URLS",
            "http://127.0.0.1:11434,http://127.0.0.1:11435",
        ).split(",")
        if value.strip()
    ]
    if not base_urls:
        base_urls = ["http://127.0.0.1:11434"]

    num_gpus = int(os.environ.get("NUM_GPUS", str(len(base_urls))))
    num_gpus = max(1, min(num_gpus, len(base_urls)))
    base_urls = base_urls[:num_gpus]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 启动多维度多GPU数据生成")
    logger.info(f"   模型: {model_name}")
    logger.info(f"   GPUs: {num_gpus}")
    logger.info(f"   Ollama endpoints: {', '.join(base_urls)}")
    logger.info(f"   生成规模: 79 indicators × 4^3 dimensions × 5 samples = 25,280 样本")
    logger.info(f"{'='*80}\n")
    
    # 获取所有indicator并分配到GPU
    all_indicators = list(FULL_INDICATORS.keys())
    indexed_indicators = list(enumerate(all_indicators))
    indicators_per_gpu = [indexed_indicators[i::num_gpus] for i in range(num_gpus)]
    
    logger.info(f"📊 Indicator分配：")
    for gpu_id, inds in enumerate(indicators_per_gpu):
        logger.info(f"   GPU{gpu_id}: {len(inds)} indicators -> {base_urls[gpu_id]}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "samples_multidim_79x64x5.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 启动多GPU worker
    sample_queue: Queue = Queue(maxsize=1000)
    processes: List[Process] = []
    
    for gpu_id in range(num_gpus):
        p = Process(
            target=worker_generate_batch,
            args=(
                gpu_id,
                indicators_per_gpu[gpu_id],
                sample_queue,
                model_name,
                temperature,
                base_urls[gpu_id],
            ),
            name=f"GPU{gpu_id}-Worker",
        )
        p.start()
        processes.append(p)
        logger.info(f"✓ 启动 GPU{gpu_id} Worker进程")
    
    # 收集结果并写入JSONL
    samples_written = 0
    samples_total = len(all_indicators) * SAMPLES_PER_INDICATOR
    
    with jsonlines.open(output_file, mode="w") as writer:
        while samples_written < samples_total or any(p.is_alive() for p in processes):
            try:
                sample = sample_queue.get(timeout=5)
                writer.write(sample)
                samples_written += 1
                
                if samples_written % 100 == 0:
                    logger.info(f"📈 进度: {samples_written}/{samples_total} ({(samples_written/samples_total)*100:.1f}%)")
            except Exception:
                if all(not p.is_alive() for p in processes):
                    break
                continue
    
    # 等待所有工作进程完成
    for p in processes:
        p.join(timeout=300)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 生成完成")
    logger.info(f"   总样本数: {samples_written}")
    logger.info(f"   输出文件: {output_file}")
    logger.info(f"{'='*80}\n")
    
    return 0 if samples_written == samples_total else 1


if __name__ == "__main__":
    exit(main())
