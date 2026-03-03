"""
所有 79 个 Indicator 的完整配置
从 codebook.txt 中提取
"""

FULL_INDICATORS = {
    # ============ Need: Individual Loss (9) ============
    "individual_loss_interpersonal": {
        "factor": "Need: Individual Loss",
        "description": "Struggling with interpersonal relationships, social isolation, or loss of close connections",
    },
    "individual_loss_career": {
        "factor": "Need: Individual Loss",
        "description": "Lack of career prospects, unemployment, or dissatisfaction with work",
    },
    "individual_loss_religious": {
        "factor": "Need: Individual Loss",
        "description": "Loss or crisis of religious faith, questioning religious identity",
    },
    "individual_loss_radical_activities": {
        "factor": "Need: Individual Loss",
        "description": "Loss of radical activities or inability to participate in radical causes",
    },
    "individual_loss_health": {
        "factor": "Need: Individual Loss",
        "description": "Health problems, physical or mental suffering",
    },
    "individual_loss_finances": {
        "factor": "Need: Individual Loss",
        "description": "Financial difficulties, poverty, or economic hardship",
    },
    "individual_loss_education": {
        "factor": "Need: Individual Loss",
        "description": "Lack of educational opportunities or academic struggles",
    },
    "individual_loss_self_esteem": {
        "factor": "Need: Individual Loss",
        "description": "Low self-esteem, shame, or loss of personal dignity",
    },
    "individual_loss_others": {
        "factor": "Need: Individual Loss",
        "description": "Loss of others due to death, separation, or tragic events",
    },
    
    # ============ Need: Social Loss (3) ============
    "social_loss_radical_religious": {
        "factor": "Need: Social Loss",
        "description": "Loss of radical religious community or religious persecution",
    },
    "social_loss_non_radical_religious": {
        "factor": "Need: Social Loss",
        "description": "Loss of non-radical religious community or religious changes",
    },
    "social_loss_non_religious": {
        "factor": "Need: Social Loss",
        "description": "Loss of non-religious social groups, community, or social status",
    },
    
    # ============ Need: Significance Gain (10) ============
    "significance_gain_leadership": {
        "factor": "Need: Significance Gain",
        "description": "Seeking leadership role or authority in a group",
    },
    "significance_gain_martyrdom": {
        "factor": "Need: Significance Gain",
        "description": "Seeking significance through martyrdom or sacrifice",
    },
    "significance_gain_vengeance": {
        "factor": "Need: Significance Gain",
        "description": "Seeking significance through vengeance or retribution",
    },
    "significance_gain_career": {
        "factor": "Need: Significance Gain",
        "description": "Seeking career advancement or professional recognition",
    },
    "significance_gain_interpersonal": {
        "factor": "Need: Significance Gain",
        "description": "Seeking recognition in interpersonal relationships",
    },
    "significance_gain_religious": {
        "factor": "Need: Significance Gain",
        "description": "Seeking religious significance or spiritual recognition",
    },
    "significance_gain_educational": {
        "factor": "Need: Significance Gain",
        "description": "Seeking educational recognition or intellectual significance",
    },
    "significance_gain_training": {
        "factor": "Need: Significance Gain",
        "description": "Seeking recognition through training or skill development",
    },
    "significance_gain_radical_activities": {
        "factor": "Need: Significance Gain",
        "description": "Seeking significance through radical or violent activities",
    },
    "significance_gain_miscellaneous": {
        "factor": "Need: Significance Gain",
        "description": "Other forms of seeking significance",
    },
    
    # ============ Need: Quest for Significance (4) ============
    "quest_significance_radical": {
        "factor": "Need: Quest for Significance",
        "description": "Quest for significance through radical means",
    },
    "quest_significance_non_radical": {
        "factor": "Need: Quest for Significance",
        "description": "Quest for significance through non-radical means",
    },
    "quest_significance_dualistic": {
        "factor": "Need: Quest for Significance",
        "description": "Dualistic quest for significance (both radical and non-radical)",
    },
    "quest_significance_competing": {
        "factor": "Need: Quest for Significance",
        "description": "Competing quests for significance",
    },
    
    # ============ Narrative: Violent (6) ============
    "narrative_violent_necessity": {
        "factor": "Narrative: Violent",
        "description": "Narrative that violence is necessary",
    },
    "narrative_violent_allowability": {
        "factor": "Narrative: Violent",
        "description": "Narrative that violence is religiously or morally allowable",
    },
    "narrative_violent_salafi_jihadism": {
        "factor": "Narrative: Violent",
        "description": "Salafi jihadist narrative promoting violent jihad",
    },
    "narrative_violent_takfiri": {
        "factor": "Narrative: Violent",
        "description": "Takfiri narrative declaring others as apostates/disbelievers",
    },
    "narrative_violent_jihad_qital": {
        "factor": "Narrative: Violent",
        "description": "Narrative promoting violent jihad or armed struggle",
    },
    "narrative_violent_martyrdom": {
        "factor": "Narrative: Violent",
        "description": "Narrative glorifying martyrdom and sacrificial death",
    },
    
    # ============ Narrative: Non-Violent (7) ============
    "narrative_nonviolent_thogut": {
        "factor": "Narrative: Non-Violent",
        "description": "Non-violent discourse on Tawheed and opposing Taghut",
    },
    "narrative_nonviolent_baiat": {
        "factor": "Narrative: Non-Violent",
        "description": "Non-violent discourse on pledging allegiance (Baia)",
    },
    "narrative_nonviolent_muslim_brotherhood": {
        "factor": "Narrative: Non-Violent",
        "description": "Non-violent narrative on Muslim brotherhood and solidarity",
    },
    "narrative_nonviolent_salafi": {
        "factor": "Narrative: Non-Violent",
        "description": "Non-violent Salafi narrative",
    },
    "narrative_nonviolent_jihad": {
        "factor": "Narrative: Non-Violent",
        "description": "Non-violent discourse on spiritual jihad",
    },
    "narrative_nonviolent_rida": {
        "factor": "Narrative: Non-Violent",
        "description": "Non-violent narrative on Islamic governance (Rida)",
    },
    "narrative_nonviolent_political_views": {
        "factor": "Narrative: Non-Violent",
        "description": "Non-violent political ideology and views",
    },
    
    # ============ Narrative: Disagreement (8) ============
    "narrative_disagreement_group_unspecified": {
        "factor": "Narrative: Disagreement",
        "description": "Disagreement with unspecified groups",
    },
    "narrative_disagreement_group_military_violent": {
        "factor": "Narrative: Disagreement",
        "description": "Disagreement with military or violent groups",
    },
    "narrative_disagreement_group_political": {
        "factor": "Narrative: Disagreement",
        "description": "Disagreement with political groups or parties",
    },
    "narrative_disagreement_group_strategies": {
        "factor": "Narrative: Disagreement",
        "description": "Disagreement over group strategies or tactics",
    },
    "narrative_disagreement_group_religious": {
        "factor": "Narrative: Disagreement",
        "description": "Disagreement over religious interpretations or practices",
    },
    "narrative_disagreement_ideology_takfiri": {
        "factor": "Narrative: Disagreement",
        "description": "Ideological disagreement on takfirism",
    },
    "narrative_disagreement_ideology_salafi": {
        "factor": "Narrative: Disagreement",
        "description": "Ideological disagreement on Salafism",
    },
    "narrative_disagreement_ideology_thogut": {
        "factor": "Narrative: Disagreement",
        "description": "Ideological disagreement on opposing Taghut",
    },
    
    # ============ Narrative: Other (3) ============
    "narrative_religious_historical_references": {
        "factor": "Narrative: Other",
        "description": "Religious or historical references",
    },
    "narrative_differences_radical_groups": {
        "factor": "Narrative: Other",
        "description": "References to differences between radical groups",
    },
    "narrative_unspecified": {
        "factor": "Narrative: Other",
        "description": "Unspecified narrative content",
    },
    
    # ============ Network: Non-Radical (7) ============
    "network_nonradical_individual": {
        "factor": "Network: Non-Radical",
        "description": "Non-radical individual relationships",
    },
    "network_nonradical_group": {
        "factor": "Network: Non-Radical",
        "description": "Non-radical group affiliations",
    },
    "network_nonradical_social_media": {
        "factor": "Network: Non-Radical",
        "description": "Non-radical social media engagement",
    },
    "network_nonradical_online_platforms": {
        "factor": "Network: Non-Radical",
        "description": "Non-radical online platform usage",
    },
    "network_nonradical_educational_setting": {
        "factor": "Network: Non-Radical",
        "description": "Non-radical educational networks",
    },
    "network_nonradical_places_locations": {
        "factor": "Network: Non-Radical",
        "description": "Non-radical places and locations",
    },
    "network_nonradical_family_member": {
        "factor": "Network: Non-Radical",
        "description": "Non-radical family relationships",
    },
    
    # ============ Network: Radical (7) ============
    "network_radical_individual": {
        "factor": "Network: Radical",
        "description": "Radical individual relationships and influences",
    },
    "network_radical_group": {
        "factor": "Network: Radical",
        "description": "Radical group affiliations",
    },
    "network_radical_social_media": {
        "factor": "Network: Radical",
        "description": "Radical social media engagement and networks",
    },
    "network_radical_online_platforms": {
        "factor": "Network: Radical",
        "description": "Radical online platform usage and communities",
    },
    "network_radical_educational_setting": {
        "factor": "Network: Radical",
        "description": "Radical educational networks and indoctrination",
    },
    "network_radical_places_locations": {
        "factor": "Network: Radical",
        "description": "Radical locations and conflict zones",
    },
    "network_radical_family_member": {
        "factor": "Network: Radical",
        "description": "Radical family relationships and influence",
    },
    
    # ============ Identity Fusion: Targets (6) ============
    "identity_fusion_target_group": {
        "factor": "Identity Fusion: Targets",
        "description": "Identity fusion with a radical group",
    },
    "identity_fusion_target_self": {
        "factor": "Identity Fusion: Targets",
        "description": "Identity fusion with oneself/personal identity",
    },
    "identity_fusion_target_leader": {
        "factor": "Identity Fusion: Targets",
        "description": "Identity fusion with group leader",
    },
    "identity_fusion_target_value": {
        "factor": "Identity Fusion: Targets",
        "description": "Identity fusion with ideological values",
    },
    "identity_fusion_target_god": {
        "factor": "Identity Fusion: Targets",
        "description": "Identity fusion with religious/spiritual beliefs",
    },
    "identity_fusion_target_family": {
        "factor": "Identity Fusion: Targets",
        "description": "Identity fusion with family identity",
    },
    
    # ============ Identity Fusion: Behavior (6) ============
    "identity_fusion_behavior_fight_die": {
        "factor": "Identity Fusion: Behavior",
        "description": "Willingness to fight or die for group/ideology",
    },
    "identity_fusion_behavior_no_fight_die": {
        "factor": "Identity Fusion: Behavior",
        "description": "Willingness to refuse to fight but maintain allegiance",
    },
    "identity_fusion_behavior_defend_group": {
        "factor": "Identity Fusion: Behavior",
        "description": "Behavior defending group or cause",
    },
    "identity_fusion_behavior_prioritize_group": {
        "factor": "Identity Fusion: Behavior",
        "description": "Prioritizing group interests over personal interests",
    },
    "identity_fusion_behavior_risks_family": {
        "factor": "Identity Fusion: Behavior",
        "description": "Actions that risk family members for group/cause",
    },
    "identity_fusion_behavior_risks_group": {
        "factor": "Identity Fusion: Behavior",
        "description": "Actions that risk group members for cause",
    },
    
    # ============ Identity Fusion: Defusion (3) ============
    "identity_fusion_defusion_removal": {
        "factor": "Identity Fusion: Defusion",
        "description": "Signs of identity defusion or removal from group",
    },
    "identity_fusion_defusion_reduction": {
        "factor": "Identity Fusion: Defusion",
        "description": "Reduction of identity fusion with group",
    },
    "identity_fusion_defusion_replacement": {
        "factor": "Identity Fusion: Defusion",
        "description": "Replacement of group identity with alternative identity",
    },
}

# total: 79 indicators
assert len(FULL_INDICATORS) == 79, f"Expected 79 indicators, got {len(FULL_INDICATORS)}"

print(f"✓ 已加载 {len(FULL_INDICATORS)} 个 indicator")
