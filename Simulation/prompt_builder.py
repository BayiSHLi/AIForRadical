"""
Shared prompt builders for Simulation generators.

Keep all generation prompt templates in one place so different simulator
implementations can share and evolve prompt logic consistently.
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
) -> str:
    """Prompt template used by data_generator_ollama.py."""
    banned_snippets = banned_snippets or []
    diversity_profile = diversity_profile or {}
    example_content = ind_config.get("example_content", "")
    keywords = rad_config.get("keywords", [])
    keyword_text = ", ".join(keywords[:4]) if keywords else "N/A"

    length_instruction = diversity_profile.get(
        "length_instruction", "1-2 sentences, 35-180 characters"
    )
    voice_hint = diversity_profile.get("voice", "first-person singular (I/me)")
    tone_hint = diversity_profile.get("tone", "calm and restrained")
    structure_hint = diversity_profile.get("structure", "single concise statement")
    punctuation_hint = diversity_profile.get("punctuation", "minimal punctuation, plain style")
    lexicon_hint = diversity_profile.get("lexicon", "mixed colloquial and formal words")
    opening_hint = diversity_profile.get("opening", "begin directly with a personal observation")

    banned_block = (
        "\n".join([f"- {s[:80]}" for s in banned_snippets[:6]])
        if banned_snippets
        else "- None"
    )

    return f"""You generate one synthetic social media post for research data augmentation.

Indicator ID: {indicator}
Indicator Factor: {ind_config.get('factor', 'N/A')}
Indicator Meaning: {ind_config.get('description', 'N/A')}

Radicality level: {radicality}
Radicality definition: {rad_config.get('description', '')}
Signal words for this level: {keyword_text}

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

Reference style sample (not to copy): {example_content[:140] if example_content else 'N/A'}

Output format: return ONLY the post text, no quotes, no explanation, no JSON."""


def build_hf_instruction_prompt(
    indicator: str,
    radicality: str,
    ind_config: Dict,
    rad_config: Dict,
) -> str:
    """Prompt template used by data_generator.py."""
    return f"""[INST] You are a social media content generator. Generate a realistic social media post (tweet, Facebook comment, etc.) based on the following:

Topic: {ind_config.get('description', '')}
Intensity Level: {radicality} - {rad_config.get('description', '')[:100]}

Requirements:
- Write 1-2 sentences (30-150 characters)
- Sound natural and authentic
- Match the topic and intensity level

Example style: {str(ind_config.get('example_content', ''))[:80]}...

Generate the post: [/INST]

"""


def build_fp16_instruction_prompt(
    indicator: str,
    radicality: str,
    ind_config: Dict,
    rad_config: Dict,
) -> str:
    """Prompt template used by data_generator_fp16.py."""
    _ = indicator, rad_config
    return f"""[INST] Generate a short social media post (1-2 sentences) about: {ind_config.get('description', '')}

Intensity: {radicality} level
Style: Natural, authentic social media language
Length: 30-120 characters

Example: {str(ind_config.get('example_content', ''))[:60]}

Write the post: [/INST]

"""
