"""
Adaptive Trust Engine (ATE) - core novelty component.

Maintains a Beta(a, b) reputation distribution per (agent, category) pair.
Dataset-agnostic: operates purely on (predicted category, human feedback),
so it works unchanged across NSL-KDD, UNSW-NB15, or any future dataset.
"""
from collections import defaultdict


class AdaptiveTrustEngine:
    def __init__(self, decay=0.98, prior_a=1.0, prior_b=1.0):
        self.decay = decay
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.store = defaultdict(lambda: [prior_a, prior_b])
        self.history = []

    def update(self, agent_id, category, human_confirms_correct, step=None):
        a, b = self.store[(agent_id, category)]
        a *= self.decay
        b *= self.decay
        if human_confirms_correct:
            a += 1.0
        else:
            b += 1.0
        self.store[(agent_id, category)] = [a, b]
        self.history.append({
            "step": step if step is not None else len(self.history),
            "agent_id": agent_id, "category": category,
            "human_confirms_correct": int(human_confirms_correct),
            "trust": a / (a + b), "a": a, "b": b,
        })

    def trust(self, agent_id, category):
        a, b = self.store[(agent_id, category)]
        return a / (a + b)

    def uncertainty(self, agent_id, category):
        a, b = self.store[(agent_id, category)]
        total = a + b
        return (a * b) / (total ** 2 * (total + 1))

    def snapshot(self):
        rows = []
        for (agent_id, category), (a, b) in self.store.items():
            rows.append({
                "agent_id": agent_id, "category": category,
                "trust": a / (a + b), "uncertainty": self.uncertainty(agent_id, category),
                "evidence_count": a + b - self.prior_a - self.prior_b,
            })
        return rows
