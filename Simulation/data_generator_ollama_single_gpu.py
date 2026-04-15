"""
Single-GPU Data Generator with O/R/M Independent Dimensions
使用qwen2.5:14b和独立的O/R/M维度生成79*4*4*4*5个样本
单GPU模式：简化设计，确保GPU使用和ID顺序正确
"""

import os
import logging
import random
import re
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import jsonlines

from llama_index.llms.ollama import Ollama
from full_indicators import FULL_INDICATORS
from prompt_builder import (
    RADICALITY_PROGRESS_PRIOR,
    RADICALITY_BEHAVIOR_RULES,
    OPINION_BEHAVIOR_RULES,
    MOBILIZATION_BEHAVIOR_RULES,
    build_ollama_generation_prompt,
    sample_diversity_profile,
    _choose_indicator_anchors,
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ O/R/M 维度定义 ============
DIMENSION_LEVELS = {
    "opinion": ["Neutral", "Low", "Medium", "High"],
    "radicalization": ["Neutral", "Low", "Medium", "High"],
    "mobilization": ["Neutral", "Low", "Medium", "High"],
}


# ============ 单GPU生成器类 ============
class SingleGPUOllamaGenerator:
    """单GPU模式下的O/R/M多维度数据生成器"""
    
    def __init__(
        self,
        model_name: str = "qwen2.5:14b",
        temperature: float = 0.8,
        external_index_dir: str = "../data/external_data_index",
        use_external_retrieval: bool = True,
        external_top_k: int = 3,
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.use_external_retrieval = use_external_retrieval
        self.external_top_k = external_top_k
        self.external_index_dir = external_index_dir
        
        # 初始化Ollama LLM，base_url可由环境变量覆盖以支持多实例
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )
        
        # 加载indicator类目
        self.indicator_catalog = self._build_indicator_catalog()
        
        # 初始化外部数据检索
        self.external_retriever = None
        if self.use_external_retrieval:
            try:
                self.external_retriever = ExternalDataRetriever(
                    index_dir=external_index_dir,
                    similarity_top_k=self.external_top_k,
                )
                if self.external_retriever.is_available:
                    logger.info(f"✓ 外部数据检索接口已启用")
                else:
                    logger.info(f"ℹ 外部数据索引未就绪")
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
                pass

        return None

    def _parse_generated_payload(self, response_text: str) -> Optional[Dict]:
        """Parse JSON response from LLM."""
        json_payload = self._extract_json_payload(response_text)
        if json_payload is None:
            return None

        return json_payload

    def _format_dimension_rule_block(self, title: str, level: str, rule_map: Dict[str, Dict[str, List[str]]]) -> str:
        """Format one dimension's rule block for the target level."""
        rules = rule_map.get(level, {})
        must_items = rules.get("must", [])
        avoid_items = rules.get("avoid", [])
        must_text = "\n".join(f"- {item}" for item in must_items) or "- Follow level semantics."
        avoid_text = "\n".join(f"- {item}" for item in avoid_items) or "- Avoid semantic drift."
        return f"{title}\nTarget level: {level}\nMust do:\n{must_text}\nMust avoid:\n{avoid_text}"

    def _build_dimension_prompt(self, indicator: str, opinion: str, radicalization: str, mobilization: str) -> str:
        """为指定的O/R/M组合构建提示"""
        ind_config = self.indicator_catalog.get(indicator, {})
        rad_config = RADICALITY_LEVELS.get(radicalization, {})
        
        # _choose_indicator_anchors expects indicator config dict, not indicator id string.
        indicator_anchors = _choose_indicator_anchors(ind_config)
        anchor_text = ", ".join(indicator_anchors[:3] or [""])
        progression_target = RADICALITY_PROGRESS_PRIOR.get(radicalization, {})
        
        # 为 O/R/M 分别构建独立约束，避免互相冲突。
        opinion_rules = self._format_dimension_rule_block(
            "Opinion dimension rules:", opinion, OPINION_BEHAVIOR_RULES
        )
        radicalization_rules = self._format_dimension_rule_block(
            "Radicalization dimension rules:", radicalization, RADICALITY_BEHAVIOR_RULES
        )
        mobilization_rules = self._format_dimension_rule_block(
            "Mobilization dimension rules:", mobilization, MOBILIZATION_BEHAVIOR_RULES
        )
        
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
                response = self.llm.complete(prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)
                
                json_payload = self._parse_generated_payload(response_text)
                
                if json_payload:
                    dimension_scores = json_payload.get("dimension_scores", {})
                    if not isinstance(dimension_scores, dict):
                        dimension_scores = {}

                    progression_meta = json_payload.get("progression_meta", {})
                    if not isinstance(progression_meta, dict):
                        progression_meta = {}

                    indicator_vector = json_payload.get("indicator_vector_79", {indicator: 1.0})
                    if not isinstance(indicator_vector, dict):
                        indicator_vector = {indicator: 1.0}

                    raw_text = (
                        json_payload.get("text")
                        or json_payload.get("content")
                        or json_payload.get("post")
                        or ""
                    )
                    content = self._clean_content(str(raw_text))
                    if not content:
                        continue

                    schema_fields = {
                        "sample_id": sample_id,
                        "text": content,
                        "dimension_scores": {
                            "opinion": self._clamp_score(dimension_scores.get("opinion"), 0.5),
                            "radicalization": self._clamp_score(dimension_scores.get("radicalization"), 0.5),
                            "mobilization": self._clamp_score(dimension_scores.get("mobilization"), 0.5),
                        },
                        "progression_meta": progression_meta,
                        "indicator_vector_79": indicator_vector,
                        "reasoning": json_payload.get("reasoning", "")[:240],
                        "source": "simulation_ollama_singlegpu",
                    }
                    
                    normalized_content = self._normalize_for_dedup(content)
                    if normalized_content not in seen:
                        seen.add(normalized_content)
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
                logger.warning(f"  重试#{attempt} 失败: {str(e)}")

        # 降级处理：使用默认值
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
            "source": "simulation_ollama_singlegpu",
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


def main():
    """主流程：单GPU生成79×64×5=25,280个样本。
    
    通过环境变量控制多实例并行：
      OLLAMA_BASE_URL  - Ollama 服务地址，默认 http://localhost:11434
      OUTPUT_FILENAME  - 输出文件名，默认 samples_multidim_79x64x5.jsonl
    """
    # 从环境变量读取实例配置，支持双卡并行
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    output_filename = os.environ.get("OUTPUT_FILENAME", "samples_multidim_79x64x5.jsonl")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 输出配置信息
    logger.info("=" * 70)
    logger.info("🚀 单GPU多维度数据生成器启动")
    logger.info("=" * 70)
    logger.info(f"📊 生成规模: 79 indicators × 4 Opinion × 4 Radicalization × 4 Mobilization × 5 reps = 25,280 samples")
    logger.info(f"📍 输出路径: {OUTPUT_DIR}/{output_filename}")
    logger.info(f"🤖 LLM模型: qwen2.5:14b | Ollama: {ollama_base_url}")
    logger.info("=" * 70)
    
    # 初始化生成器
    generator = SingleGPUOllamaGenerator(
        model_name="qwen2.5:14b",
        temperature=0.8,
        base_url=ollama_base_url,
    )
    
    output_file = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 获取所有indicators
    all_indicators = list(FULL_INDICATORS.keys())
    logger.info(f"✓ 已加载 {len(all_indicators)} 个 indicators")
    
    sample_id = 1
    samples_generated = 0
    existing_contents = []
    
    # 生成循环：indicator → opinion → radicalization → mobilization → 5 reps
    total_combinations = len(all_indicators) * 4 * 4 * 4 * 5
    
    try:
        with jsonlines.open(output_file, mode="w") as writer:
            for ind_idx, indicator in enumerate(all_indicators, 1):
                logger.info(f"\n📌 [Indicator {ind_idx}/{len(all_indicators)}] {indicator}")
                
                for opinion in DIMENSION_LEVELS["opinion"]:
                    for radicalization in DIMENSION_LEVELS["radicalization"]:
                        for mobilization in DIMENSION_LEVELS["mobilization"]:
                            for rep in range(5):
                                try:
                                    sample = generator.generate_sample(
                                        indicator=indicator,
                                        opinion=opinion,
                                        radicalization=radicalization,
                                        mobilization=mobilization,
                                        sample_id=sample_id,
                                        existing_contents=existing_contents,
                                    )
                                    
                                    writer.write(sample)
                                    # Flush each sample so progress is visible immediately on disk.
                                    if hasattr(writer, "_fp") and writer._fp:
                                        writer._fp.flush()
                                    existing_contents.append(sample.get("Content", ""))
                                    samples_generated += 1
                                    sample_id += 1
                                    
                                    # 每100个样本打印进度
                                    if samples_generated % 100 == 0:
                                        progress_pct = (samples_generated / total_combinations) * 100
                                        logger.info(f"   ✓ 已生成 {samples_generated}/{total_combinations} ({progress_pct:.1f}%)")
                                    
                                except Exception as e:
                                    logger.error(f"   ✗ 生成失败 (sample_id={sample_id}): {e}")
                                    sample_id += 1
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ 生成完成！")
        logger.info(f"   样本数: {samples_generated}")
        logger.info(f"   输出文件: {output_file}")
        logger.info(f"   预期: {total_combinations}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("⚠ 用户中断生成过程")
    except Exception as e:
        logger.error(f"❌ 生成过程出错: {e}")
        raise


if __name__ == "__main__":
    main()
