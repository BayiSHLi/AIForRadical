"""
Shared prompt builders for Simulation generators.

This module centralizes prompt templates so all generators keep consistent
semantics around indicator targeting, radicality control, and diversity.
"""

import random
from typing import Dict, List, Optional


STYLE_POOL = [
    "first_person_reflection",
    "short_status_update",
    "question_to_audience",
    "observational_commentary",
    "emotion_forward",
    "neutral_informational",
    "dialogue_snippet",
    "micro_story",
]


LENGTH_POOL = [
    {
        "label": "micro",
        "instruction": "exactly 1 sentence, 35-70 characters",
    },
    {
        "label": "short",
        "instruction": "1 sentence, 60-110 characters",
    },
    {
        "label": "standard",
        "instruction": "1-2 sentences, 90-150 characters",
    },
    {
        "label": "expanded",
        "instruction": "2 short sentences, 140-190 characters",
    },
]


VOICE_POOL = [
    "first-person singular (I/me)",
    "first-person plural (we/us)",
    "second-person address (you)",
    "impersonal observer voice",
]


TONE_POOL = [
    "calm and restrained",
    "frustrated but controlled",
    "anxious and uncertain",
    "assertive and declarative",
    "reflective and introspective",
]


STRUCTURE_POOL = [
    "single concise statement",
    "cause -> effect phrasing",
    "contrast phrasing (A but B)",
    "short setup + short conclusion",
    "question then self-answer",
]


PUNCTUATION_POOL = [
    "minimal punctuation, plain style",
    "one comma max, no semicolons",
    "allow one rhetorical question mark",
    "allow one dash for emphasis",
]


LEXICON_POOL = [
    "everyday vocabulary, very colloquial",
    "mixed colloquial and formal words",
    "more abstract wording, still natural",
]


OPENING_POOL = [
    "begin directly with a personal observation",
    "begin with a concrete situation",
    "begin with a short emotional state",
    "begin with a brief claim",
]


RADICALITY_PROGRESS_PRIOR = {
    # Expected O/R/M strength priors used as soft controls in prompt wording.
    "Neutral": {"opinion": 0.20, "radicalization": 0.05, "mobilization": 0.00},
    "Low": {"opinion": 0.45, "radicalization": 0.20, "mobilization": 0.05},
    "Medium": {"opinion": 0.65, "radicalization": 0.55, "mobilization": 0.20},
    "High": {"opinion": 0.80, "radicalization": 0.80, "mobilization": 0.65},
}


RADICALITY_BEHAVIOR_RULES = {
    "Neutral": {
        "must": [
            "Stay focused on ordinary life context.",
            "Keep the statement non-ideological and non-violent.",
        ],
        "avoid": [
            "Do not include extremist slogans or militant framing.",
            "Do not imply support for radical groups or actions.",
        ],
    },
    "Low": {
        "must": [
            "Show awareness/exposure only (observation, mention, concern).",
            "Use uncertain or distanced stance instead of commitment.",
        ],
        "avoid": [
            "Do not imply membership, recruitment, or operational intent.",
            "Do not endorse violence or celebrate attacks.",
        ],
    },
    "Medium": {
        "must": [
            "Convey non-violent involvement/support for radical narratives.",
            "Allow stronger ideological certainty than Low.",
        ],
        "avoid": [
            "Do not provide explicit violent intent or attack planning.",
            "Do not include tactical instructions.",
        ],
    },
    "High": {
        "must": [
            "Convey explicit endorsement or intent related to violent extremism.",
            "Maintain realistic social-media language, not manifesto style.",
        ],
        "avoid": [
            "Do not include logistical details, targets, or how-to guidance.",
            "Do not output multiple paragraphs or formal speeches.",
        ],
    },
}


OPINION_BEHAVIOR_RULES = {
    "Neutral": {
        "must": [
            "Express ordinary views without ideological commitment.",
            "Keep judgments mild and context-bound.",
        ],
        "avoid": [
            "Do not use absolutist in-group vs out-group framing.",
            "Do not normalize extremist worldviews.",
        ],
    },
    "Low": {
        "must": [
            "Show curiosity or concern, but keep stance uncertain.",
            "Allow mention of controversial narratives without endorsing them.",
        ],
        "avoid": [
            "Do not present ideological claims as unquestionable truth.",
            "Do not call for social exclusion of out-groups.",
        ],
    },
    "Medium": {
        "must": [
            "Express clearer alignment with radical narratives.",
            "Use confident wording while staying non-operational.",
        ],
        "avoid": [
            "Do not escalate to explicit violent approval.",
            "Do not include tactical or target-level details.",
        ],
    },
    "High": {
        "must": [
            "Show strongly polarized or dehumanizing ideological stance.",
            "Signal explicit endorsement of violent-extremist narratives.",
        ],
        "avoid": [
            "Do not provide practical attack guidance.",
            "Do not output manifesto-style long form rhetoric.",
        ],
    },
}


MOBILIZATION_BEHAVIOR_RULES = {
    "Neutral": {
        "must": [
            "Contain no call-to-action beyond normal daily activities.",
            "Keep behavior intent low-stakes and non-political.",
        ],
        "avoid": [
            "Do not imply joining any radical network.",
            "Do not encourage coordinated collective action.",
        ],
    },
    "Low": {
        "must": [
            "At most imply passive attention (reading, noticing, observing).",
            "Keep agency weak and non-committal.",
        ],
        "avoid": [
            "Do not express intention to join, train, or organize.",
            "Do not ask others to participate in radical action.",
        ],
    },
    "Medium": {
        "must": [
            "Allow mild non-violent mobilization cues (discussion, support circles).",
            "Show willingness to engage socially around the narrative.",
        ],
        "avoid": [
            "Do not include violent preparation or logistics.",
            "Do not make direct recruitment calls.",
        ],
    },
    "High": {
        "must": [
            "Convey strong intent to act or help mobilize others.",
            "Permit explicit commitment language, but keep it non-operational.",
        ],
        "avoid": [
            "Do not include targets, locations, timing, or methods.",
            "Do not provide weapons, bomb, or attack instructions.",
        ],
    },
}


def _choose_indicator_anchors(ind_config: Dict, max_items: int = 6) -> List[str]:
    """Select short anchor terms to keep indicator semantics grounded."""
    anchors = []
    sample_keywords = ind_config.get("sample_keywords") or []
    for item in sample_keywords:
        term = str(item).strip()
        if term and term not in anchors:
            anchors.append(term)

    factor = str(ind_config.get("factor", "")).strip()
    if factor and factor not in anchors:
        anchors.append(factor)

    return anchors[:max_items]


def _format_rule_block(radicality: str) -> str:
    """Format O/R/M level constraints aligned with the progressive theory."""

    def _format_dimension_rule_block(title: str, rule_map: Dict[str, Dict[str, List[str]]]) -> str:
        rules = rule_map.get(radicality, {})
        must_items = rules.get("must", [])
        avoid_items = rules.get("avoid", [])
        must_text = "\n".join(f"- {item}" for item in must_items) or "- Follow level semantics."
        avoid_text = "\n".join(f"- {item}" for item in avoid_items) or "- Avoid semantic drift."
        return f"{title}\nMust do:\n{must_text}\nMust avoid:\n{avoid_text}"

    radicalization_block = _format_dimension_rule_block(
        "Radicalization dimension rules:", RADICALITY_BEHAVIOR_RULES
    )
    opinion_block = _format_dimension_rule_block(
        "Opinion dimension rules:", OPINION_BEHAVIOR_RULES
    )
    mobilization_block = _format_dimension_rule_block(
        "Mobilization dimension rules:", MOBILIZATION_BEHAVIOR_RULES
    )

    return (
        "Level-specific rules (O/R/M aligned):\n"
        f"{opinion_block}\n\n"
        f"{radicalization_block}\n\n"
        f"{mobilization_block}"
    )


def get_progression_target_text(radicality: str) -> str:
    """Return a compact O/R/M prior string for the requested radicality."""
    prior = RADICALITY_PROGRESS_PRIOR.get(radicality, RADICALITY_PROGRESS_PRIOR["Low"])
    return (
        f"O~{prior['opinion']:.2f}, "
        f"R~{prior['radicalization']:.2f}, "
        f"M~{prior['mobilization']:.2f}"
    )


def _format_banned_snippets(banned_snippets: List[str]) -> str:
    if not banned_snippets:
        return "- None"

    cleaned = []
    for item in banned_snippets[:6]:
        text = " ".join(str(item).split())
        if text:
            cleaned.append(f"- {text[:90]}")
    return "\n".join(cleaned) if cleaned else "- None"


def _format_output_schema_block(indicator: str, radicality: str) -> str:
    """Unified schema aligned with README Phase-A protocol."""
    return f"""Output format (strict): return ONE valid JSON object only.
Do not wrap in markdown fences and do not add extra text.

{{
    "sample_id": null,
    "text": "<one social media post>",
    "timestamp": null,
    "dimension_scores": {{
        "opinion": 0.00,
        "radicalization": 0.00,
        "mobilization": 0.00
    }},
    "progression_meta": {{
        "target_radicality": "{radicality}",
        "consistency_note": "<short check of O/R/M progression>"
    }},
    "indicator_vector_79": {{
        "{indicator}": 1.0
    }},
    "reasoning": "<max 25 words>",
    "source": "simulation_llm"
}}

Schema rules:
- "text" must be exactly one coherent post.
- dimension_scores must be floats in [0,1].
- Keep monotonic tendency in most cases: opinion >= radicalization >= mobilization.
- "indicator_vector_79" must include "{indicator}" with a positive value.
- Keep all keys present even when value is null.
"""


def sample_diversity_profile() -> Dict[str, str]:
    """Sample a prompt diversity profile from shared pools."""
    length_item = random.choice(LENGTH_POOL)
    return {
        "style": random.choice(STYLE_POOL),
        "length_label": length_item["label"],
        "length_instruction": length_item["instruction"],
        "voice": random.choice(VOICE_POOL),
        "tone": random.choice(TONE_POOL),
        "structure": random.choice(STRUCTURE_POOL),
        "punctuation": random.choice(PUNCTUATION_POOL),
        "lexicon": random.choice(LEXICON_POOL),
        "opening": random.choice(OPENING_POOL),
    }


def build_ollama_generation_prompt(
    indicator: str,
    radicality: str,
    ind_config: Dict,
    rad_config: Dict,
    style_hint: str,
    banned_snippets: Optional[List[str]] = None,
    diversity_profile: Optional[Dict[str, str]] = None,
    external_context: str = "",
) -> str:
    """Prompt template used by data_generator_ollama.py."""
    banned_snippets = banned_snippets or []
    diversity_profile = diversity_profile or {}
    example_content = ind_config.get("example_content", "")
    keywords = rad_config.get("keywords", [])
    keyword_text = ", ".join(keywords[:4]) if keywords else "N/A"
    indicator_anchors = _choose_indicator_anchors(ind_config)
    anchor_text = ", ".join(indicator_anchors) if indicator_anchors else "N/A"
    progression_target = get_progression_target_text(radicality)
    level_rule_block = _format_rule_block(radicality)
    output_schema_block = _format_output_schema_block(indicator=indicator, radicality=radicality)

    length_instruction = diversity_profile.get(
        "length_instruction", "1-2 sentences, 35-180 characters"
    )
    voice_hint = diversity_profile.get("voice", "first-person singular (I/me)")
    tone_hint = diversity_profile.get("tone", "calm and restrained")
    structure_hint = diversity_profile.get("structure", "single concise statement")
    punctuation_hint = diversity_profile.get("punctuation", "minimal punctuation, plain style")
    lexicon_hint = diversity_profile.get("lexicon", "mixed colloquial and formal words")
    opening_hint = diversity_profile.get("opening", "begin directly with a personal observation")

    banned_block = _format_banned_snippets(banned_snippets)

    return f"""You generate one synthetic social media post for simulation data augmentation.

Indicator ID: {indicator}
Indicator Factor: {ind_config.get('factor', 'N/A')}
Indicator Meaning: {ind_config.get('description', 'N/A')}
Indicator anchors (use 1-2 naturally): {anchor_text}

Radicality level: {radicality}
Radicality definition: {rad_config.get('description', '')}
Signal words for this level: {keyword_text}
Progression prior (internal guide, do not print numbers): {progression_target}

{level_rule_block}

Style requirement: {style_hint}
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
2) Do NOT reuse these snippets:
{banned_block}
3) Avoid repetitive openers like "Feeling..." unless clearly necessary.
4) Keep it natural, specific, and varied.
5) Keep exactly one coherent post, no lists, no role-play tags.

Reference style sample (not to copy): {example_content[:140] if example_content else 'N/A'}

External dataset references (for realism and grounding, do not copy literally):
{external_context if external_context else '(No external dataset references retrieved)'}

{output_schema_block}"""


def build_hf_instruction_prompt(
    indicator: str,
    radicality: str,
    ind_config: Dict,
    rad_config: Dict,
) -> str:
    """Prompt template used by data_generator.py."""
    indicator_anchors = _choose_indicator_anchors(ind_config)
    anchor_text = ", ".join(indicator_anchors) if indicator_anchors else "N/A"
    progression_target = get_progression_target_text(radicality)
    level_rule_block = _format_rule_block(radicality)
    output_schema_block = _format_output_schema_block(indicator=indicator, radicality=radicality)

    return f"""[INST] You are a social media content generator. Generate a realistic social media post (tweet, Facebook comment, etc.) based on the following:

Indicator ID: {indicator}
Topic: {ind_config.get('description', '')}
Indicator anchors: {anchor_text}
Intensity Level: {radicality} - {rad_config.get('description', '')[:120]}
Progression prior (internal): {progression_target}

{level_rule_block}

Requirements:
- Write 1-2 sentences (35-170 characters)
- Sound natural and authentic
- Match the topic and intensity level
- Use 1-2 anchor terms naturally
- Keep "text" in the requested length range and realistic social style

Example style: {str(ind_config.get('example_content', ''))[:80]}...

{output_schema_block}

Generate the post: [/INST]

"""


def build_fp16_instruction_prompt(
    indicator: str,
    radicality: str,
    ind_config: Dict,
    rad_config: Dict,
) -> str:
    """Prompt template used by data_generator_fp16.py."""
    _ = rad_config
    indicator_anchors = _choose_indicator_anchors(ind_config)
    anchor_text = ", ".join(indicator_anchors) if indicator_anchors else "N/A"
    progression_target = get_progression_target_text(radicality)
    level_rule_block = _format_rule_block(radicality)
    output_schema_block = _format_output_schema_block(indicator=indicator, radicality=radicality)

    return f"""[INST] Generate one short social media post.

Indicator ID: {indicator}
Topic: {ind_config.get('description', '')}
Indicator anchors: {anchor_text}
Intensity: {radicality} level
Progression prior (internal): {progression_target}
Style: natural, authentic social media language
Length: 1-2 sentences, 35-150 characters

{level_rule_block}

Example style: {str(ind_config.get('example_content', ''))[:70]}

{output_schema_block}

Write the post: [/INST]

"""
