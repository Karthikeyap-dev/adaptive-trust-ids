"""
Analyst archetype feedback models.

Each archetype's feedback (confirms or rejects the AI's prediction) is a
mixture of three behaviors:
  - genuine judgment: noisy, at the archetype's base accuracy
  - automation bias: rubber-stamps high-confidence AI outputs regardless
    of whether they're actually correct (dangerous - can poison trust)
  - distrust bias: rejects low-confidence AI outputs regardless of truth
    (over-cautious, wastes escalation budget but not dangerous)

Response time is modeled separately (log-normal, archetype-specific mean)
as a descriptive workload statistic - it does not feed into the trust
math, only into the reported "cognitive load" characterization.
"""
import numpy as np

EXPERT_KNOWLEDGE_CATEGORIES = {
    "nsl_kdd": {"r2l", "u2r"},
    "unsw_nb15": {"Analysis", "Backdoor", "DoS"},
    "cicids2017": {"botnet"},
}

ARCHETYPES = {
    "novice": {
        "base_accuracy": 0.55,
        "automation_bias": 0.35,
        "distrust_bias": 0.20,
        "confidence_high_threshold": 0.80,
        "confidence_low_threshold": 0.50,
        "response_time_mean_s": 45.0,
        "response_time_sigma": 0.5,
    },
    "expert": {
        "base_accuracy": 0.92,
        "category_boost_accuracy": 0.97,
        "automation_bias": 0.05,
        "distrust_bias": 0.05,
        "confidence_high_threshold": 0.85,
        "confidence_low_threshold": 0.40,
        "response_time_mean_s": 12.0,
        "response_time_sigma": 0.3,
    },
    "complacent": {
        "base_accuracy": 0.75,
        "automation_bias": 0.85,
        "distrust_bias": 0.02,
        "confidence_high_threshold": 0.70,
        "confidence_low_threshold": 0.30,
        "response_time_mean_s": 5.0,
        "response_time_sigma": 0.2,
    },
}


def archetype_feedback(archetype_name, is_correct, ai_confidence, category, dataset_key, rng):
    """Returns (feedback: bool, response_time_seconds: float) for one alert."""
    params = ARCHETYPES[archetype_name]
    r = rng.random()

    if r < params["automation_bias"] and ai_confidence >= params["confidence_high_threshold"]:
        feedback = True
    elif r < params["automation_bias"] + params["distrust_bias"] and ai_confidence <= params["confidence_low_threshold"]:
        feedback = False
    else:
        base_acc = params["base_accuracy"]
        if archetype_name == "expert" and category in EXPERT_KNOWLEDGE_CATEGORIES.get(dataset_key, set()):
            base_acc = params["category_boost_accuracy"]
        judged_correctly = rng.random() < base_acc
        feedback = is_correct if judged_correctly else (not is_correct)

    response_time = rng.lognormal(np.log(params["response_time_mean_s"]), params["response_time_sigma"])
    return bool(feedback), float(response_time)
