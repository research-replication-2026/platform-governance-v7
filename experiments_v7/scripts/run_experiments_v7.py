from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces

    BaseEnv = gym.Env
    Discrete = spaces.Discrete
    Box = spaces.Box
    GYMNASIUM_INSTALLED = True
except ImportError:
    BaseEnv = object
    GYMNASIUM_INSTALLED = False

    class Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(self.n))

        def contains(self, x: Any) -> bool:
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    class Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype=np.float32) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def contains(self, x: Any) -> bool:
            arr = np.asarray(x)
            return arr.shape == self.shape and np.all(arr >= self.low) and np.all(arr <= self.high)


ROOT = Path(__file__).resolve().parents[2]
V7 = ROOT / "experiments_v7"
OUT = V7 / "outputs"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"

SEED = 20261008

ACTION_NAMES = {
    0: "accept",
    1: "accept_conditionally",
    2: "automated_checks",
    3: "human_review",
    4: "temporarily_restrict",
    5: "reject",
}

POLICIES = [
    "static_permissive",
    "static_restrictive",
    "security_rule",
    "heuristic_score_threshold",
    "contextual_bandit",
    "q_learning",
    "q_learning_fine",
    "linear_q_approx",
    "oracle_upper_bound",
]

V6_BASELINES = [
    "static_permissive",
    "static_restrictive",
    "security_rule",
    "heuristic_score_threshold",
    "contextual_bandit",
    "q_learning",
]

MAIN_TRAIN_SEEDS = [1101, 1201, 1301]
MAIN_EVAL_SEEDS = [5101 + i * 37 for i in range(10)]
ABLATION_TRAIN_SEEDS = [2101, 2201]
ABLATION_EVAL_SEEDS = [6101 + i * 41 for i in range(8)]
DIAGNOSTIC_TRAIN_SEEDS = [3101, 3201]
DIAGNOSTIC_EVAL_SEEDS = [7101 + i * 43 for i in range(6)]


@dataclass
class Submission:
    risk: float
    uncertainty: float
    harmful: bool
    developer_quality: float
    diversity_group: int
    age: int = 0


@dataclass
class ScenarioConfig:
    name: str = "scenario"
    horizon: int = 100
    base_harm_rate: float = 0.22
    risk_alpha_harmful: float = 6.0
    risk_beta_harmful: float = 2.2
    risk_alpha_legitimate: float = 2.0
    risk_beta_legitimate: float = 6.0
    initial_trust: float = 0.74
    initial_participation: float = 0.75
    review_capacity: int = 4
    max_review_queue: int = 16
    auto_check_delay: int = 1
    human_review_delay: int = 2
    incident_delay_min: int = 2
    incident_delay_max: int = 6
    security_penalty: float = 2.2
    incident_penalty: float = 3.0
    false_restriction_penalty: float = 1.6
    delay_penalty: float = 0.04
    human_review_cost: float = 0.16
    automated_check_cost: float = 0.06
    restriction_cost: float = 0.22
    rejection_cost: float = 0.12
    conditional_cost: float = 0.05
    trust_loss_false_restriction: float = 0.035
    trust_loss_incident: float = 0.055
    trust_gain_legit_accept: float = 0.006
    participation_sensitivity: float = 0.55
    auto_sensitivity: float = 0.72
    human_sensitivity: float = 0.92
    appeal_success_legit: float = 0.55
    appeal_delay: int = 4
    mandatory_restrict_risk: float = 0.93
    mandatory_review_uncertainty: float = 0.82
    terminal_liquidation: bool = True


@dataclass
class MechanismConfig:
    name: str
    queue_effects: bool = True
    delayed_incidents: bool = True
    trust_effects: bool = True
    participation_effects: bool = True
    delayed_appeals: bool = True
    sequentiality_scale: float = 1.0


@dataclass
class LearnedPolicy:
    kind: str
    values: Any
    state_visits: Counter
    state_action_visits: Counter
    training_curve: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    discretizer: str = "adjusted"


SCENARIOS = [
    ScenarioConfig("balanced_reference"),
    ScenarioConfig("high_security_pressure", security_penalty=3.4, incident_penalty=4.6, base_harm_rate=0.30),
    ScenarioConfig("high_review_capacity", review_capacity=8, max_review_queue=28, human_review_cost=0.12),
    ScenarioConfig("constrained_review_capacity", review_capacity=2, max_review_queue=8, human_review_cost=0.24),
    ScenarioConfig(
        "high_false_restriction_cost",
        false_restriction_penalty=2.8,
        trust_loss_false_restriction=0.07,
        participation_sensitivity=0.85,
    ),
    ScenarioConfig(
        "participation_sensitive_ecosystem",
        initial_participation=0.82,
        participation_sensitivity=1.0,
        trust_loss_false_restriction=0.06,
        delay_penalty=0.08,
    ),
]

STRESS_SCENARIOS = [
    replace(SCENARIOS[0], name="risk_high_base_harm", base_harm_rate=0.38),
    replace(
        SCENARIOS[0],
        name="review_queue_saturation",
        base_harm_rate=0.34,
        human_review_cost=0.04,
        review_capacity=1,
        max_review_queue=3,
        human_review_delay=6,
        mandatory_review_uncertainty=0.70,
    ),
    replace(
        SCENARIOS[0],
        name="long_horizon_trust_feedback",
        horizon=150,
        participation_sensitivity=1.0,
        trust_loss_false_restriction=0.07,
        trust_loss_incident=0.09,
    ),
]

SENSITIVITY_SCENARIOS = [
    replace(SCENARIOS[0], name="human_review_high_cost", human_review_cost=0.38),
    replace(SCENARIOS[0], name="human_review_low_cost", human_review_cost=0.06),
    replace(SCENARIOS[0], name="reward_security_dominant", security_penalty=3.8, incident_penalty=5.2),
    replace(SCENARIOS[0], name="reward_generativity_dominant", false_restriction_penalty=3.2, restriction_cost=0.34),
    replace(
        SCENARIOS[0],
        name="risk_less_separable",
        risk_alpha_harmful=4.2,
        risk_beta_harmful=3.0,
        risk_alpha_legitimate=2.8,
        risk_beta_legitimate=4.8,
    ),
]

FULL_SEQUENTIAL = MechanismConfig("full_sequential")
NONSEQUENTIAL = MechanismConfig(
    "nonsequential",
    queue_effects=False,
    delayed_incidents=False,
    trust_effects=False,
    participation_effects=False,
    delayed_appeals=False,
    sequentiality_scale=0.0,
)
ABLATIONS = [
    FULL_SEQUENTIAL,
    replace(FULL_SEQUENTIAL, name="without_queue", queue_effects=False),
    replace(FULL_SEQUENTIAL, name="without_delayed_incidents", delayed_incidents=False),
    replace(FULL_SEQUENTIAL, name="without_trust_effects", trust_effects=False),
    replace(FULL_SEQUENTIAL, name="without_participation_effects", participation_effects=False),
    replace(FULL_SEQUENTIAL, name="without_delayed_appeals", delayed_appeals=False),
]
SEQUENTIALITY_LEVELS = [
    MechanismConfig("seq_scale_0", sequentiality_scale=0.0),
    MechanismConfig("seq_scale_0_5", sequentiality_scale=0.5),
    MechanismConfig("seq_scale_1", sequentiality_scale=1.0),
    MechanismConfig("seq_scale_1_5", sequentiality_scale=1.5),
]

SUMMARY_METRICS = [
    "total_reward",
    "mean_reward",
    "current_action_reward_total",
    "delayed_past_reward_total",
    "collective_state_reward_total",
    "terminal_liquidation_reward_total",
    "legitimate_contributions_accepted",
    "diversity_success_entropy",
    "developer_retention",
    "developer_trust",
    "mean_time_to_publication_proxy",
    "rejected_legitimate_innovation",
    "harmful_submissions_released",
    "security_incidents",
    "undetected_high_risk_actions",
    "human_review_workload",
    "automated_check_workload",
    "governance_expenditure",
    "reversed_restrictions",
    "safety_guard_interventions",
    "safety_constraint_compliance_rate",
    "review_delay_overflow",
    "review_queue_saturation_events",
    "terminal_pending_items_liquidated",
    "pending_appeals_end",
    "unresolved_incidents_end",
    "unseen_eval_state_rate",
]

DIFF_METRICS = [
    "mean_reward",
    "legitimate_contributions_accepted",
    "rejected_legitimate_innovation",
    "harmful_submissions_released",
    "security_incidents",
    "undetected_high_risk_actions",
    "governance_expenditure",
    "human_review_workload",
    "review_queue_saturation_events",
    "safety_constraint_compliance_rate",
]


class GovernanceEnvV7(BaseEnv):
    """Instrumented platform-governance environment for V7.

    The environment separates reward components so that contextual bandits are
    trained only on current-action reward, while RL algorithms can use total
    transition reward including delayed consequences and terminal liquidation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ScenarioConfig,
        environment: str,
        mechanisms: MechanismConfig | None = None,
        seed: int = SEED,
    ) -> None:
        if environment not in {"nonsequential", "sequential"}:
            raise ValueError(f"Unknown environment: {environment}")
        self.config = config
        self.environment = environment
        self.mechanisms = mechanisms if mechanisms is not None else (
            FULL_SEQUENTIAL if environment == "sequential" else NONSEQUENTIAL
        )
        self.action_space = Discrete(6)
        self.observation_space = Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)
        cfg = self.config
        self.t = 0
        self.trust = cfg.initial_trust
        self.participation = cfg.initial_participation
        self.recent_legit_accepts = deque(maxlen=12)
        self.security_exposure = 0.0
        self.unresolved_incidents = 0
        self.pending_appeals = 0
        self.false_restrictions = 0
        self.review_queue: deque[tuple[Submission, int]] = deque()
        self.auto_queue: deque[tuple[Submission, int]] = deque()
        self.incident_queue: deque[tuple[int, float]] = deque()
        self.appeal_queue: deque[tuple[int, bool]] = deque()
        self.metrics = Counter()
        self.diversity_success = Counter()
        self.last_capacity_remaining = cfg.review_capacity
        self.current_submission = self._draw_submission()
        return self._state(), {"action_mask": self.action_mask()}

    def _scale(self) -> float:
        return self.mechanisms.sequentiality_scale if self.environment == "sequential" else 0.0

    def _effective_trust(self) -> float:
        if self.environment == "sequential" and self.mechanisms.trust_effects:
            return self.trust
        return self.config.initial_trust

    def _effective_participation(self) -> float:
        if self.environment == "sequential" and self.mechanisms.participation_effects:
            return self.participation
        return self.config.initial_participation

    def _queue_enabled(self) -> bool:
        return self.environment == "sequential" and self.mechanisms.queue_effects and self._scale() > 0

    def _incidents_delayed(self) -> bool:
        return self.environment == "sequential" and self.mechanisms.delayed_incidents and self._scale() > 0

    def _trust_enabled(self) -> bool:
        return self.environment == "sequential" and self.mechanisms.trust_effects and self._scale() > 0

    def _participation_enabled(self) -> bool:
        return self.environment == "sequential" and self.mechanisms.participation_effects and self._scale() > 0

    def _appeals_delayed(self) -> bool:
        return self.environment == "sequential" and self.mechanisms.delayed_appeals and self._scale() > 0

    def _draw_submission(self) -> Submission:
        cfg = self.config
        scale = self._scale()
        tau = self._effective_trust()
        p = self._effective_participation()
        harm_prob = np.clip(
            cfg.base_harm_rate + scale * (0.22 * (1.0 - tau) - 0.08 * (p - 0.5)),
            0.04,
            0.70,
        )
        harmful = bool(self.rng.random() < harm_prob)
        if harmful:
            risk = float(self.rng.beta(cfg.risk_alpha_harmful, cfg.risk_beta_harmful))
            developer_quality = float(self.rng.beta(2.0, 5.0))
        else:
            risk = float(self.rng.beta(cfg.risk_alpha_legitimate, cfg.risk_beta_legitimate))
            developer_quality = float(self.rng.beta(5.0, 2.0))
        uncertainty = float(np.clip(self.rng.beta(2.0, 3.0) + scale * 0.18 * (1.0 - tau), 0.02, 0.98))
        diversity_probs = np.array([0.40, 0.28, 0.20, 0.12])
        diversity_probs = diversity_probs * (0.65 + 0.35 * p)
        diversity_probs[0] += scale * (1.0 - p)
        diversity_probs = diversity_probs / diversity_probs.sum()
        diversity_group = int(self.rng.choice(np.arange(4), p=diversity_probs))
        return Submission(risk, uncertainty, harmful, developer_quality, diversity_group)

    def _state(self) -> np.ndarray:
        cfg = self.config
        s = self.current_submission
        recent = sum(self.recent_legit_accepts) / max(1, len(self.recent_legit_accepts))
        review_risks = [sub.risk for sub, _ in self.review_queue]
        review_delays = [delay for _, delay in self.review_queue]
        incident_severities = [severity for _, severity in self.incident_queue]
        state = np.array(
            [
                s.risk,
                s.uncertainty,
                self._effective_participation(),
                recent,
                min(len(self.review_queue) / max(1, cfg.max_review_queue), 1.0) if self._queue_enabled() else 0.0,
                min(self.last_capacity_remaining / max(1, cfg.review_capacity), 1.0) if self._queue_enabled() else 1.0,
                min(self.security_exposure / 10.0, 1.0) if self.environment == "sequential" else 0.0,
                min(self.unresolved_incidents / 10.0, 1.0) if self.environment == "sequential" else 0.0,
                min(self.pending_appeals / 10.0, 1.0) if self._appeals_delayed() else 0.0,
                min(self.false_restrictions / 10.0, 1.0) if self.environment == "sequential" else 0.0,
                self._effective_trust(),
                min(len(self.auto_queue) / max(1, cfg.max_review_queue), 1.0) if self._queue_enabled() else 0.0,
                float(np.mean(review_risks)) if review_risks else 0.0,
                min(float(np.mean(review_delays)) / max(1, cfg.human_review_delay), 1.0) if review_delays else 0.0,
                min(len(self.incident_queue) / 10.0, 1.0) if self._incidents_delayed() else 0.0,
                float(np.mean(incident_severities)) if incident_severities else 0.0,
                min(len(self.appeal_queue) / 10.0, 1.0) if self._appeals_delayed() else 0.0,
            ],
            dtype=np.float32,
        )
        return np.clip(state, 0.0, 1.0)

    def action_mask(self) -> np.ndarray:
        cfg = self.config
        s = self.current_submission
        mask = np.ones(6, dtype=bool)
        if self._queue_enabled() and len(self.review_queue) >= cfg.max_review_queue:
            mask[3] = False
        if s.risk >= cfg.mandatory_restrict_risk:
            mask[0] = False
            mask[1] = False
        if s.uncertainty >= cfg.mandatory_review_uncertainty and s.risk >= 0.55:
            mask[0] = False
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        pending_counts = {
            "auto": len(self.auto_queue),
            "review": len(self.review_queue),
            "incident": len(self.incident_queue),
            "appeal": len(self.appeal_queue),
        }
        components = {
            "current_action_reward": 0.0,
            "delayed_past_reward": 0.0,
            "collective_state_reward": 0.0,
            "terminal_liquidation_reward": 0.0,
            "guard_penalty": 0.0,
        }
        mask = self.action_mask()
        if self._queue_enabled() and len(self.review_queue) >= self.config.max_review_queue:
            self.metrics["review_queue_saturation_events"] += 1
        requested_action = int(action)
        guard_applied = False
        if not mask[requested_action]:
            action = self._fallback_action(mask)
            guard_applied = True
            self.metrics["safety_guard_interventions"] += 1
            components["guard_penalty"] -= 0.25
        action = int(action)
        components["current_action_reward"] += self._apply_action(action)
        components["delayed_past_reward"] += self._process_existing_delayed(pending_counts)
        components["collective_state_reward"] += self._update_collective_state()
        self.t += 1
        truncated = self.t >= self.config.horizon
        if truncated and self.config.terminal_liquidation:
            components["terminal_liquidation_reward"] += self._liquidate_terminal()
        if not truncated:
            self.current_submission = self._draw_submission()
        total_reward = float(sum(components.values()))
        info = {
            "requested_action": ACTION_NAMES[requested_action],
            "executed_action": ACTION_NAMES[action],
            "guard_applied": guard_applied,
            "action_mask": self.action_mask(),
            "metrics": dict(self.metrics),
            "reward_components": components,
        }
        return self._state(), total_reward, False, truncated, info

    def _fallback_action(self, mask: np.ndarray) -> int:
        for candidate in [2, 3, 4, 5, 1, 0]:
            if mask[candidate]:
                return candidate
        return 5

    def _apply_action(self, action: int) -> float:
        cfg = self.config
        s = self.current_submission
        reward = 0.0
        self.metrics[f"action_{ACTION_NAMES[action]}"] += 1
        if action == 0:
            reward += self._release_submission(s, conditional=False)
        elif action == 1:
            reward += self._release_submission(s, conditional=True)
            action_cost = cfg.conditional_cost + cfg.delay_penalty
            self.metrics["governance_expenditure"] += action_cost
            reward -= action_cost
        elif action == 2:
            self.metrics["automated_check_workload"] += 1
            action_cost = cfg.automated_check_cost + cfg.delay_penalty
            self.metrics["governance_expenditure"] += action_cost
            reward -= action_cost
            if self._queue_enabled():
                self.auto_queue.append((s, max(1, int(round(cfg.auto_check_delay * max(0.5, self._scale()))))))
            else:
                reward += self._resolve_automated_check(s)
        elif action == 3:
            self.metrics["human_review_workload"] += 1
            review_delay = max(1, int(round(cfg.human_review_delay * max(0.5, self._scale()))))
            action_cost = cfg.human_review_cost + cfg.delay_penalty * review_delay
            self.metrics["governance_expenditure"] += action_cost
            reward -= action_cost
            if self._queue_enabled():
                self.review_queue.append((s, review_delay))
            else:
                reward += self._resolve_human_review(s)
        elif action == 4:
            reward += self._restrict_submission(s, temporary=True)
        elif action == 5:
            reward += self._restrict_submission(s, temporary=False)
        return reward

    def _release_submission(self, s: Submission, conditional: bool) -> float:
        cfg = self.config
        reward = 0.0
        if s.harmful:
            self.metrics["harmful_released"] += 1
            if s.risk >= 0.70:
                self.metrics["undetected_high_risk_actions"] += 1
            exposure = s.risk * (0.55 if conditional else 1.0)
            if self.environment == "sequential":
                self.security_exposure += exposure
            incident_prob = np.clip(s.risk * (0.75 if not conditional else 0.42) + 0.12 * s.uncertainty, 0.0, 0.95)
            if self.rng.random() < incident_prob:
                if self._incidents_delayed():
                    delay = int(self.rng.integers(cfg.incident_delay_min, cfg.incident_delay_max + 1))
                    delay = max(1, int(round(delay * max(0.5, self._scale()))))
                    self.incident_queue.append((delay, s.risk))
                else:
                    reward += self._materialize_incident(s.risk)
            reward -= cfg.security_penalty * exposure
        else:
            self.metrics["legitimate_accepted"] += 1
            self.recent_legit_accepts.append(1)
            self.diversity_success[s.diversity_group] += 1
            if self._trust_enabled():
                self.trust = min(1.0, self.trust + cfg.trust_gain_legit_accept * self._scale())
            reward += s.developer_quality
            if conditional:
                reward -= 0.12 * cfg.false_restriction_penalty
        self.metrics["time_to_publication_total"] += s.age + (1 if conditional else 0)
        return reward

    def _materialize_incident(self, severity: float) -> float:
        cfg = self.config
        self.metrics["security_incidents"] += 1
        if self.environment == "sequential":
            self.unresolved_incidents += 1
        if self._trust_enabled():
            self.trust = max(0.0, self.trust - cfg.trust_loss_incident * severity * self._scale())
        return -cfg.incident_penalty * severity

    def _restrict_submission(self, s: Submission, temporary: bool) -> float:
        cfg = self.config
        cost = cfg.restriction_cost if temporary else cfg.rejection_cost
        self.metrics["governance_expenditure"] += cost
        reward = -cost
        if s.harmful:
            self.metrics["harmful_blocked"] += 1
            reward += cfg.security_penalty * 0.35 * s.risk
        else:
            self.metrics["rejected_legitimate_innovation"] += 1
            if self.environment == "sequential":
                self.false_restrictions += 1
            self.recent_legit_accepts.append(0)
            if self._trust_enabled():
                self.trust = max(0.0, self.trust - cfg.trust_loss_false_restriction * self._scale())
            if self._appeals_delayed():
                self.pending_appeals += 1
                delay = max(1, int(round(cfg.appeal_delay * max(0.5, self._scale()))))
                self.appeal_queue.append((delay, True))
            elif self.rng.random() < cfg.appeal_success_legit:
                self.metrics["reversed_restrictions"] += 1
                reward += 0.18
            reward -= cfg.false_restriction_penalty * s.developer_quality
        return reward

    def _resolve_automated_check(self, sub: Submission) -> float:
        cfg = self.config
        if sub.harmful and self.rng.random() < cfg.auto_sensitivity:
            return self._restrict_submission(sub, temporary=True)
        if sub.uncertainty > 0.62:
            self.metrics["human_review_workload"] += 1
            self.metrics["governance_expenditure"] += cfg.human_review_cost
            return -cfg.human_review_cost + self._resolve_human_review(sub)
        return self._release_submission(sub, conditional=True)

    def _resolve_human_review(self, sub: Submission) -> float:
        cfg = self.config
        if sub.harmful and self.rng.random() < cfg.human_sensitivity:
            return self._restrict_submission(sub, temporary=False)
        false_restriction_prob = (1.0 - cfg.human_sensitivity) * (0.6 + sub.uncertainty)
        if (not sub.harmful) and self.rng.random() < false_restriction_prob:
            return self._restrict_submission(sub, temporary=False)
        return self._release_submission(sub, conditional=False)

    def _process_existing_delayed(self, pending_counts: dict[str, int]) -> float:
        if self.environment != "sequential":
            return 0.0
        reward = 0.0
        if self._incidents_delayed():
            reward += self._process_existing_incidents(pending_counts["incident"])
        if self._appeals_delayed():
            reward += self._process_existing_appeals(pending_counts["appeal"])
        if self._queue_enabled():
            reward += self._process_existing_auto_queue(pending_counts["auto"])
            reward += self._process_existing_review_queue(pending_counts["review"])
        else:
            self.last_capacity_remaining = self.config.review_capacity
        return reward

    def _process_existing_incidents(self, n_existing: int) -> float:
        reward = 0.0
        new_incidents: deque[tuple[int, float]] = deque()
        untouched = list(self.incident_queue)[n_existing:]
        for _ in range(n_existing):
            delay, severity = self.incident_queue.popleft()
            delay -= 1
            if delay <= 0:
                reward += self._materialize_incident(severity)
            else:
                new_incidents.append((delay, severity))
        new_incidents.extend(untouched)
        self.incident_queue = new_incidents
        return reward

    def _process_existing_appeals(self, n_existing: int) -> float:
        cfg = self.config
        reward = 0.0
        new_appeals: deque[tuple[int, bool]] = deque()
        untouched = list(self.appeal_queue)[n_existing:]
        for _ in range(n_existing):
            delay, legitimate = self.appeal_queue.popleft()
            delay -= 1
            if delay <= 0:
                self.pending_appeals = max(0, self.pending_appeals - 1)
                if legitimate and self.rng.random() < cfg.appeal_success_legit:
                    self.metrics["reversed_restrictions"] += 1
                    if self._trust_enabled():
                        self.trust = min(1.0, self.trust + 0.03 * self._scale())
                    reward += 0.18
            else:
                new_appeals.append((delay, legitimate))
        new_appeals.extend(untouched)
        self.appeal_queue = new_appeals
        return reward

    def _process_existing_auto_queue(self, n_existing: int) -> float:
        cfg = self.config
        reward = 0.0
        new_auto: deque[tuple[Submission, int]] = deque()
        untouched = list(self.auto_queue)[n_existing:]
        for _ in range(n_existing):
            sub, delay = self.auto_queue.popleft()
            sub.age += 1
            delay -= 1
            if delay > 0:
                new_auto.append((sub, delay))
                continue
            if sub.harmful and self.rng.random() < cfg.auto_sensitivity:
                reward += self._restrict_submission(sub, temporary=True)
            elif sub.uncertainty > 0.62 and len(self.review_queue) < cfg.max_review_queue:
                self.review_queue.append((sub, max(1, int(round(cfg.human_review_delay * max(0.5, self._scale()))))))
                self.metrics["human_review_workload"] += 1
                self.metrics["governance_expenditure"] += cfg.human_review_cost
                reward -= cfg.human_review_cost
            else:
                reward += self._release_submission(sub, conditional=True)
        new_auto.extend(untouched)
        self.auto_queue = new_auto
        return reward

    def _process_existing_review_queue(self, n_existing: int) -> float:
        cfg = self.config
        reward = 0.0
        processed = 0
        new_review: deque[tuple[Submission, int]] = deque()
        existing_items = [self.review_queue.popleft() for _ in range(min(n_existing, len(self.review_queue)))]
        untouched = list(self.review_queue)
        for sub, delay in existing_items:
            sub.age += 1
            delay -= 1
            if delay > 0:
                new_review.append((sub, delay))
                continue
            if processed >= cfg.review_capacity:
                new_review.append((sub, 1))
                self.metrics["review_delay_overflow"] += 1
                self.metrics["governance_expenditure"] += cfg.delay_penalty
                reward -= cfg.delay_penalty
                continue
            processed += 1
            reward += self._resolve_human_review(sub)
        new_review.extend(untouched)
        self.review_queue = new_review
        self.last_capacity_remaining = max(0, cfg.review_capacity - processed)
        return reward

    def _update_collective_state(self) -> float:
        cfg = self.config
        if self._participation_enabled():
            incident_pressure = min(self.unresolved_incidents / 10.0, 1.0)
            false_restriction_pressure = min(self.false_restrictions / 10.0, 1.0)
            target = np.clip(
                0.25
                + 0.65 * self._effective_trust()
                - self._scale() * 0.20 * false_restriction_pressure * cfg.participation_sensitivity
                - self._scale() * 0.10 * incident_pressure,
                0.05,
                1.0,
            )
            self.participation = float(0.88 * self.participation + 0.12 * target)
        else:
            self.participation = cfg.initial_participation
        if not self._trust_enabled():
            self.trust = cfg.initial_trust
        return 0.0

    def _liquidate_terminal(self) -> float:
        reward = 0.0
        pending_items = len(self.incident_queue) + len(self.appeal_queue) + len(self.auto_queue) + len(self.review_queue)
        self.metrics["terminal_pending_items_liquidated"] += pending_items
        while self.incident_queue:
            _, severity = self.incident_queue.popleft()
            reward += self._materialize_incident(severity)
        while self.auto_queue:
            sub, delay = self.auto_queue.popleft()
            sub.age += max(0, delay)
            delay_cost = self.config.delay_penalty * max(0, delay)
            self.metrics["governance_expenditure"] += delay_cost
            reward -= delay_cost
            reward += self._resolve_automated_check(sub)
        while self.review_queue:
            sub, delay = self.review_queue.popleft()
            sub.age += max(0, delay)
            delay_cost = self.config.delay_penalty * max(0, delay)
            self.metrics["governance_expenditure"] += delay_cost
            reward -= delay_cost
            reward += self._resolve_human_review(sub)
        while self.appeal_queue:
            _, legitimate = self.appeal_queue.popleft()
            self.pending_appeals = max(0, self.pending_appeals - 1)
            if legitimate and self.rng.random() < self.config.appeal_success_legit:
                self.metrics["reversed_restrictions"] += 1
                reward += 0.18
        return reward

    def final_metrics(self) -> dict[str, Any]:
        total_published = self.metrics["legitimate_accepted"] + self.metrics["harmful_released"]
        total_actions = sum(self.metrics[f"action_{name}"] for name in ACTION_NAMES.values())
        diversity_counts = np.array([self.diversity_success[i] for i in range(4)], dtype=float)
        if diversity_counts.sum() > 0:
            probs = diversity_counts / diversity_counts.sum()
            diversity = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum() / math.log(4))
        else:
            diversity = 0.0
        avg_time = self.metrics["time_to_publication_total"] / max(1, total_published)
        return {
            "legitimate_contributions_accepted": self.metrics["legitimate_accepted"],
            "diversity_success_entropy": diversity,
            "developer_retention": self._effective_participation(),
            "developer_trust": self._effective_trust(),
            "mean_time_to_publication_proxy": avg_time,
            "rejected_legitimate_innovation": self.metrics["rejected_legitimate_innovation"],
            "harmful_submissions_released": self.metrics["harmful_released"],
            "security_incidents": self.metrics["security_incidents"],
            "undetected_high_risk_actions": self.metrics["undetected_high_risk_actions"],
            "human_review_workload": self.metrics["human_review_workload"],
            "automated_check_workload": self.metrics["automated_check_workload"],
            "governance_expenditure": self.metrics["governance_expenditure"],
            "reversed_restrictions": self.metrics["reversed_restrictions"],
            "safety_guard_interventions": self.metrics["safety_guard_interventions"],
            "safety_constraint_compliance_rate": 1.0
            - self.metrics["safety_guard_interventions"] / max(1, total_actions),
            "review_delay_overflow": self.metrics["review_delay_overflow"],
            "review_queue_saturation_events": self.metrics["review_queue_saturation_events"],
            "terminal_pending_items_liquidated": self.metrics["terminal_pending_items_liquidated"],
            "pending_appeals_end": self.pending_appeals,
            "unresolved_incidents_end": self.unresolved_incidents,
        }


def discretize_state(obs: np.ndarray, mode: str = "adjusted") -> tuple[int, ...]:
    if mode == "fine":
        idx_bins = {
            0: [0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
            1: [0.20, 0.40, 0.60, 0.80],
            2: [0.35, 0.60, 0.80],
            4: [0.20, 0.50, 0.80],
            5: [0.20, 0.50, 0.80],
            6: [0.15, 0.35, 0.65],
            7: [0.10, 0.30, 0.60],
            8: [0.10, 0.30, 0.60],
            10: [0.35, 0.60, 0.80],
            11: [0.20, 0.50, 0.80],
            14: [0.10, 0.30, 0.60],
        }
    elif mode == "coarse":
        idx_bins = {i: [0.33, 0.66] for i in [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 14]}
    else:
        idx_bins = {
            0: [0.20, 0.40, 0.60, 0.80],
            1: [0.25, 0.50, 0.75],
            2: [0.33, 0.66],
            4: [0.25, 0.60],
            5: [0.25, 0.60],
            6: [0.20, 0.50],
            7: [0.10, 0.40],
            8: [0.10, 0.40],
            10: [0.33, 0.66],
            11: [0.20, 0.60],
            14: [0.10, 0.40],
        }
    return tuple(int(np.digitize(float(obs[i]), bins)) for i, bins in idx_bins.items())


def feature_vector(obs: np.ndarray) -> np.ndarray:
    return np.concatenate([np.array([1.0], dtype=float), np.asarray(obs, dtype=float)])


def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    return int(np.argmax(np.where(mask, q_values, -1e12)))


def choose_epsilon_greedy(q_values: np.ndarray, mask: np.ndarray, rng: np.random.Generator, epsilon: float) -> int:
    legal = np.flatnonzero(mask)
    if rng.random() < epsilon:
        return int(rng.choice(legal))
    return masked_argmax(q_values, mask)


def train_contextual_bandit(
    config: ScenarioConfig,
    environment: str,
    mechanisms: MechanismConfig,
    seed: int,
    episodes: int = 180,
    discretizer: str = "adjusted",
) -> LearnedPolicy:
    rng = np.random.default_rng(seed)
    q: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(6, dtype=float))
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(6, dtype=float))
    state_visits: Counter = Counter()
    state_action_visits: Counter = Counter()
    curve: list[dict[str, Any]] = []
    for ep in range(episodes):
        env = GovernanceEnvV7(config, environment, mechanisms, seed=seed + ep * 13)
        obs, info = env.reset(seed=seed + ep * 13)
        done = False
        total_reward = 0.0
        update_reward = 0.0
        epsilon = 0.04 + 0.24 * math.exp(-ep / max(1, episodes / 3))
        while not done:
            state = discretize_state(obs, discretizer)
            state_visits[state] += 1
            action = choose_epsilon_greedy(q[state], info["action_mask"], rng, epsilon)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            current_reward = float(next_info["reward_components"]["current_action_reward"])
            state_action_visits[(state, action)] += 1
            counts[state][action] += 1.0
            alpha = 1.0 / max(1.0, counts[state][action])
            q[state][action] += alpha * (current_reward - q[state][action])
            total_reward += reward
            update_reward += current_reward
            obs, info = next_obs, next_info
            done = terminated or truncated
        if ep % max(1, episodes // 12) == 0 or ep == episodes - 1:
            curve.append(
                {
                    "episode": ep,
                    "total_reward": total_reward,
                    "bandit_update_reward": update_reward,
                    "epsilon": epsilon,
                }
            )
    diagnostics = diagnostics_from_visits(q, state_visits, state_action_visits, curve, episodes)
    diagnostics["bandit_training_reward"] = "current_action_reward_only"
    return LearnedPolicy("contextual_bandit", q, state_visits, state_action_visits, curve, diagnostics, discretizer)


def train_q_learning(
    config: ScenarioConfig,
    environment: str,
    mechanisms: MechanismConfig,
    seed: int,
    episodes: int = 180,
    alpha: float = 0.14,
    gamma: float = 0.93,
    epsilon_start: float = 0.28,
    epsilon_end: float = 0.04,
    discretizer: str = "adjusted",
) -> LearnedPolicy:
    rng = np.random.default_rng(seed)
    q: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(6, dtype=float))
    state_visits: Counter = Counter()
    state_action_visits: Counter = Counter()
    curve: list[dict[str, Any]] = []
    for ep in range(episodes):
        env = GovernanceEnvV7(config, environment, mechanisms, seed=seed + ep * 17)
        obs, info = env.reset(seed=seed + ep * 17)
        done = False
        total_reward = 0.0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-ep / max(1, episodes / 3))
        while not done:
            state = discretize_state(obs, discretizer)
            state_visits[state] += 1
            action = choose_epsilon_greedy(q[state], info["action_mask"], rng, epsilon)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_state = discretize_state(next_obs, discretizer)
            future = np.max(np.where(next_info["action_mask"], q[next_state], -1e12))
            state_action_visits[(state, action)] += 1
            q[state][action] += alpha * (reward + gamma * future - q[state][action])
            total_reward += reward
            obs, info = next_obs, next_info
            done = terminated or truncated
        if ep % max(1, episodes // 12) == 0 or ep == episodes - 1:
            curve.append({"episode": ep, "total_reward": total_reward, "epsilon": epsilon})
    diagnostics = diagnostics_from_visits(q, state_visits, state_action_visits, curve, episodes)
    diagnostics.update({"alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start, "epsilon_end": epsilon_end})
    return LearnedPolicy("q_learning", q, state_visits, state_action_visits, curve, diagnostics, discretizer)


def train_linear_q(
    config: ScenarioConfig,
    environment: str,
    mechanisms: MechanismConfig,
    seed: int,
    episodes: int = 180,
    alpha: float = 0.035,
    gamma: float = 0.93,
    epsilon_start: float = 0.28,
    epsilon_end: float = 0.04,
) -> LearnedPolicy:
    rng = np.random.default_rng(seed)
    weights = np.zeros((6, 18), dtype=float)
    state_visits: Counter = Counter()
    state_action_visits: Counter = Counter()
    curve: list[dict[str, Any]] = []
    for ep in range(episodes):
        env = GovernanceEnvV7(config, environment, mechanisms, seed=seed + ep * 19)
        obs, info = env.reset(seed=seed + ep * 19)
        done = False
        total_reward = 0.0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-ep / max(1, episodes / 3))
        while not done:
            state = discretize_state(obs, "adjusted")
            state_visits[state] += 1
            x = feature_vector(obs)
            q_values = weights @ x
            action = choose_epsilon_greedy(q_values, info["action_mask"], rng, epsilon)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_x = feature_vector(next_obs)
            next_q = weights @ next_x
            future = np.max(np.where(next_info["action_mask"], next_q, -1e12))
            td_error = np.clip(reward + gamma * future - q_values[action], -5.0, 5.0)
            weights[action] += alpha * td_error * x
            state_action_visits[(state, action)] += 1
            total_reward += reward
            obs, info = next_obs, next_info
            done = terminated or truncated
        if ep % max(1, episodes // 12) == 0 or ep == episodes - 1:
            curve.append({"episode": ep, "total_reward": total_reward, "epsilon": epsilon})
    diagnostics = diagnostics_from_visits(None, state_visits, state_action_visits, curve, episodes)
    diagnostics.update({"alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start, "epsilon_end": epsilon_end})
    return LearnedPolicy("linear_q_approx", weights, state_visits, state_action_visits, curve, diagnostics)


def diagnostics_from_visits(
    q: dict[tuple[int, ...], np.ndarray] | None,
    state_visits: Counter,
    state_action_visits: Counter,
    curve: list[dict[str, Any]],
    episodes: int,
) -> dict[str, Any]:
    visited_states = len(state_visits)
    visited_state_actions = len(state_action_visits)
    possible_state_actions = max(1, visited_states * 6)
    last_rewards = [row["total_reward"] for row in curve[-3:]]
    first_rewards = [row["total_reward"] for row in curve[:3]]
    return {
        "episodes": episodes,
        "visited_states": visited_states,
        "visited_state_actions": visited_state_actions,
        "state_action_coverage_within_seen_states": visited_state_actions / possible_state_actions,
        "mean_first_curve_reward": float(np.mean(first_rewards)) if first_rewards else 0.0,
        "mean_last_curve_reward": float(np.mean(last_rewards)) if last_rewards else 0.0,
        "curve_reward_delta": (float(np.mean(last_rewards)) - float(np.mean(first_rewards))) if first_rewards else 0.0,
    }


def policy_action(
    policy: str,
    obs: np.ndarray,
    mask: np.ndarray,
    learned: LearnedPolicy | None = None,
) -> int:
    risk = float(obs[0])
    uncertainty = float(obs[1])
    queue_load = float(obs[4])
    capacity = float(obs[5])
    exposure = float(obs[6])
    incidents = float(obs[7])
    trust = float(obs[10])
    auto_load = float(obs[11])

    if policy == "static_permissive":
        preferred = [0, 1, 2, 3, 4, 5]
    elif policy == "static_restrictive":
        preferred = [5, 4, 3, 2, 1, 0]
    elif policy == "security_rule":
        if risk > 0.88 or incidents > 0.55:
            preferred = [4, 5, 3, 2, 1, 0]
        elif risk > 0.68:
            preferred = [3, 2, 4, 1, 5, 0]
        elif risk > 0.42 or uncertainty > 0.65:
            preferred = [2, 3, 1, 0, 4, 5]
        else:
            preferred = [0, 1, 2, 3, 4, 5]
    elif policy == "heuristic_score_threshold":
        score = 0.64 * risk + 0.18 * uncertainty + 0.08 * exposure + 0.05 * (1.0 - trust) + 0.03 * queue_load + 0.02 * auto_load
        if score > 0.78:
            preferred = [4, 5, 3, 2, 1, 0] if capacity < 0.25 else [3, 4, 5, 2, 1, 0]
        elif score > 0.58:
            preferred = [3, 2, 1, 4, 0, 5] if capacity > 0.20 else [2, 4, 1, 5, 0, 3]
        elif score > 0.36:
            preferred = [2, 1, 0, 3, 4, 5]
        else:
            preferred = [0, 1, 2, 3, 4, 5]
    elif policy in {"contextual_bandit", "q_learning", "q_learning_fine"}:
        if learned is None:
            raise ValueError(f"{policy} requires learned values")
        state = discretize_state(obs, learned.discretizer)
        return masked_argmax(learned.values[state], mask)
    elif policy == "linear_q_approx":
        if learned is None:
            raise ValueError("linear_q_approx requires learned values")
        return masked_argmax(learned.values @ feature_vector(obs), mask)
    else:
        raise ValueError(policy)

    for action in preferred:
        if mask[action]:
            return action
    return int(np.flatnonzero(mask)[0])


def oracle_action(env: GovernanceEnvV7, mask: np.ndarray) -> int:
    s = env.current_submission
    queue_load = len(env.review_queue) / max(1, env.config.max_review_queue)
    if s.harmful:
        if s.risk > 0.76:
            preferred = [4, 5, 3, 2, 1, 0]
        elif queue_load < 0.65:
            preferred = [3, 2, 4, 5, 1, 0]
        else:
            preferred = [2, 4, 5, 3, 1, 0]
    else:
        if s.risk < 0.45 and s.uncertainty < 0.75:
            preferred = [0, 1, 2, 3, 4, 5]
        elif s.uncertainty < 0.62:
            preferred = [1, 0, 2, 3, 4, 5]
        else:
            preferred = [2, 1, 3, 0, 4, 5]
    for action in preferred:
        if mask[action]:
            return action
    return int(np.flatnonzero(mask)[0])


def evaluate_policy(
    config: ScenarioConfig,
    environment: str,
    mechanisms: MechanismConfig,
    policy: str,
    seed: int,
    train_seed: int,
    learned: LearnedPolicy | None = None,
) -> dict[str, Any]:
    env = GovernanceEnvV7(config, environment, mechanisms, seed=seed)
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    action_counts = Counter()
    reward_components = Counter()
    unseen_states = 0
    observed_states = 0
    while not done:
        if learned is not None:
            state = discretize_state(obs, learned.discretizer)
            observed_states += 1
            if state not in learned.state_visits:
                unseen_states += 1
        if policy == "oracle_upper_bound":
            action = oracle_action(env, info["action_mask"])
        else:
            action = policy_action(policy, obs, info["action_mask"], learned)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action_counts[info["executed_action"]] += 1
        for key, value in info["reward_components"].items():
            reward_components[key] += value
        done = terminated or truncated
    row = env.final_metrics()
    row.update(
        {
            "environment": environment,
            "mechanism": mechanisms.name,
            "sequentiality_scale": mechanisms.sequentiality_scale if environment == "sequential" else 0.0,
            "scenario": config.name,
            "policy": policy,
            "train_seed": train_seed,
            "eval_seed": seed,
            "total_reward": total_reward,
            "mean_reward": total_reward / config.horizon,
            "current_action_reward_total": reward_components["current_action_reward"],
            "delayed_past_reward_total": reward_components["delayed_past_reward"],
            "collective_state_reward_total": reward_components["collective_state_reward"],
            "terminal_liquidation_reward_total": reward_components["terminal_liquidation_reward"],
            "guard_penalty_total": reward_components["guard_penalty"],
            "unseen_eval_state_rate": unseen_states / max(1, observed_states),
            **{f"action_{name}": action_counts[name] for name in ACTION_NAMES.values()},
        }
    )
    return row


def train_learners(
    config: ScenarioConfig,
    environment: str,
    mechanisms: MechanismConfig,
    train_seed: int,
    episodes: int,
) -> dict[str, LearnedPolicy]:
    base_seed = train_seed
    return {
        "contextual_bandit": train_contextual_bandit(config, environment, mechanisms, base_seed + 101, episodes=episodes),
        "q_learning": train_q_learning(config, environment, mechanisms, base_seed + 202, episodes=episodes),
        "q_learning_fine": train_q_learning(
            config, environment, mechanisms, base_seed + 303, episodes=episodes, alpha=0.10, gamma=0.95, discretizer="fine"
        ),
        "linear_q_approx": train_linear_q(config, environment, mechanisms, base_seed + 404, episodes=episodes),
    }


def run_policy_set(
    configs: list[ScenarioConfig],
    environment: str,
    mechanisms: MechanismConfig,
    train_seeds: list[int],
    eval_seeds: list[int],
    episodes: int,
    label: str,
    policies: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policies = policies or POLICIES
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    env_offset = 0 if environment == "nonsequential" else 500_000
    for scenario_idx, config in enumerate(configs):
        for train_seed in train_seeds:
            actual_train_seed = SEED + env_offset + scenario_idx * 10_000 + train_seed
            print(f"{label}: train {environment}/{mechanisms.name}/{config.name}/seed={train_seed}", flush=True)
            learned = train_learners(config, environment, mechanisms, actual_train_seed, episodes)
            for policy_name, learner in learned.items():
                diag_rows.append(
                    {
                        "label": label,
                        "environment": environment,
                        "mechanism": mechanisms.name,
                        "scenario": config.name,
                        "policy": policy_name,
                        "train_seed": train_seed,
                        **learner.diagnostics,
                    }
                )
                for curve_point in learner.training_curve:
                    curve_rows.append(
                        {
                            "label": label,
                            "environment": environment,
                            "mechanism": mechanisms.name,
                            "scenario": config.name,
                            "policy": policy_name,
                            "train_seed": train_seed,
                            **curve_point,
                        }
                    )
            for eval_seed in eval_seeds:
                actual_eval_seed = SEED + env_offset + scenario_idx * 10_000 + eval_seed
                for policy in policies:
                    learner = learned.get(policy)
                    rows.append(
                        evaluate_policy(
                            config,
                            environment,
                            mechanisms,
                            policy,
                            seed=actual_eval_seed,
                            train_seed=train_seed,
                            learned=learner,
                        )
                    )
    return pd.DataFrame(rows), pd.DataFrame(diag_rows), pd.DataFrame(curve_rows)


def summarize_results(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = dict(zip(group_cols, keys))
        row["n"] = int(len(g))
        for metric in SUMMARY_METRICS:
            mean = float(g[metric].mean())
            sd = float(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
            ci = 1.96 * sd / math.sqrt(max(1, len(g)))
            row[f"{metric}_mean"] = round(mean, 6)
            row[f"{metric}_sd"] = round(sd, 6)
            row[f"{metric}_ci95_low"] = round(mean - ci, 6)
            row[f"{metric}_ci95_high"] = round(mean + ci, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def action_distributions(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        denom = sum(float(g[f"action_{name}"].mean()) for name in ACTION_NAMES.values())
        for action in ACTION_NAMES.values():
            mean_count = float(g[f"action_{action}"].mean())
            rows.append(
                {
                    **base,
                    "action": action,
                    "mean_count": round(mean_count, 6),
                    "mean_share": round(mean_count / max(1.0, denom), 6),
                    "sd_count": round(float(g[f"action_{action}"].std(ddof=1)), 6),
                }
            )
    return pd.DataFrame(rows)


def paired_differences(
    df: pd.DataFrame,
    group_cols: list[str],
    focal_policy: str,
    comparisons: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pair_keys = ["train_seed", "eval_seed"]
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        focal = g[g["policy"].eq(focal_policy)][pair_keys + DIFF_METRICS]
        for baseline in comparisons:
            comp = g[g["policy"].eq(baseline)][pair_keys + DIFF_METRICS]
            merged = focal.merge(comp, on=pair_keys, suffixes=("_focal", "_baseline"))
            for metric in DIFF_METRICS:
                diff = merged[f"{metric}_focal"] - merged[f"{metric}_baseline"]
                n = int(len(diff))
                mean = float(diff.mean()) if n else float("nan")
                sd = float(diff.std(ddof=1)) if n > 1 else 0.0
                ci = 1.96 * sd / math.sqrt(max(1, n))
                rows.append(
                    {
                        **base,
                        "comparison": f"{focal_policy}_minus_{baseline}",
                        "metric": metric,
                        "n_pairs": n,
                        "mean_diff": round(mean, 6),
                        "sd_diff": round(sd, 6),
                        "ci95_low": round(mean - ci, 6),
                        "ci95_high": round(mean + ci, 6),
                    }
                )
    return pd.DataFrame(rows)


def run_q_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    diagnostic_scenarios = [SCENARIOS[0], SCENARIOS[1], STRESS_SCENARIOS[1]]
    budgets = [40, 120, 360]
    hparams = [
        {"alpha": 0.08, "gamma": 0.85, "epsilon_start": 0.20, "epsilon_end": 0.04},
        {"alpha": 0.14, "gamma": 0.93, "epsilon_start": 0.28, "epsilon_end": 0.04},
        {"alpha": 0.22, "gamma": 0.97, "epsilon_start": 0.34, "epsilon_end": 0.06},
    ]
    for scenario_idx, config in enumerate(diagnostic_scenarios):
        for budget in budgets:
            for hp_idx, hp in enumerate(hparams):
                for train_seed in DIAGNOSTIC_TRAIN_SEEDS:
                    actual_train_seed = SEED + 900_000 + scenario_idx * 20_000 + budget * 10 + hp_idx * 100 + train_seed
                    learner = train_q_learning(
                        config,
                        "sequential",
                        FULL_SEQUENTIAL,
                        actual_train_seed,
                        episodes=budget,
                        **hp,
                    )
                    eval_rows = []
                    for eval_seed in DIAGNOSTIC_EVAL_SEEDS:
                        actual_eval_seed = SEED + 950_000 + scenario_idx * 20_000 + eval_seed
                        eval_rows.append(
                            evaluate_policy(
                                config,
                                "sequential",
                                FULL_SEQUENTIAL,
                                "q_learning",
                                actual_eval_seed,
                                train_seed,
                                learner,
                            )
                        )
                    eval_df = pd.DataFrame(eval_rows)
                    rows.append(
                        {
                            "scenario": config.name,
                            "budget": budget,
                            "hparam_id": hp_idx,
                            "train_seed": train_seed,
                            **hp,
                            **learner.diagnostics,
                            "eval_mean_reward": float(eval_df["mean_reward"].mean()),
                            "eval_security_incidents": float(eval_df["security_incidents"].mean()),
                            "eval_unseen_state_rate": float(eval_df["unseen_eval_state_rate"].mean()),
                        }
                    )
                    for point in learner.training_curve:
                        curve_rows.append(
                            {
                                "scenario": config.name,
                                "budget": budget,
                                "hparam_id": hp_idx,
                                "train_seed": train_seed,
                                **point,
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def security_comparable_summary(summary: pd.DataFrame, focal_policy: str = "q_learning") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (environment, scenario), g in summary.groupby(["environment", "scenario"]):
        focal = g[g["policy"].eq(focal_policy)]
        if focal.empty:
            continue
        focal_security = float(focal["security_incidents_mean"].iloc[0])
        comparable = g[(g["security_incidents_mean"] - focal_security).abs() <= 0.5].copy()
        comparable = comparable.sort_values("mean_reward_mean", ascending=False)
        for rank, (_, row) in enumerate(comparable.iterrows(), start=1):
            rows.append(
                {
                    "environment": environment,
                    "scenario": scenario,
                    "focal_policy": focal_policy,
                    "focal_security_incidents_mean": focal_security,
                    "policy": row["policy"],
                    "security_incidents_mean": row["security_incidents_mean"],
                    "mean_reward_mean": row["mean_reward_mean"],
                    "governance_expenditure_mean": row["governance_expenditure_mean"],
                    "security_comparable_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def reward_attribution_summary(replications: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "current_action_reward_total",
        "delayed_past_reward_total",
        "terminal_liquidation_reward_total",
        "guard_penalty_total",
        "total_reward",
    ]
    rows = []
    for keys, g in replications.groupby(["environment", "scenario", "policy"]):
        row = dict(zip(["environment", "scenario", "policy"], keys))
        for col in cols:
            row[f"{col}_mean"] = round(float(g[col].mean()), 6)
        denom = max(1e-9, abs(float(g["total_reward"].mean())))
        row["abs_delayed_share_of_total_reward"] = round(abs(float(g["delayed_past_reward_total"].mean())) / denom, 6)
        row["abs_terminal_share_of_total_reward"] = round(
            abs(float(g["terminal_liquidation_reward_total"].mean())) / denom, 6
        )
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_reference() -> dict[str, Any]:
    events = pd.read_csv(ROOT / "data" / "processed" / "governance_events_final.csv")
    return {
        "events_total": int(len(events)),
        "google_play_events": int(events["platform"].eq("Android / Google Play").sum()),
        "platform_counts": events["platform"].value_counts().to_dict(),
        "direct_ml_events": int(events["round3_ml_evidence_code"].eq("DIRECT_ML_DECISION").sum()),
        "mechanism_counts": events["round3_governance_mechanism_code"].value_counts().to_dict(),
        "decision_locus_counts": events["round3_decision_locus_code"].value_counts().to_dict(),
    }


def markov_audit_table() -> pd.DataFrame:
    rows = [
        {
            "component": "current submission",
            "v7_observed": "risk, uncertainty",
            "still_hidden": "true harmfulness and developer quality",
            "assessment": "true harmfulness is intentionally hidden except from oracle_upper_bound",
        },
        {
            "component": "review queue",
            "v7_observed": "queue load, mean queued risk, mean remaining delay, capacity remaining",
            "still_hidden": "exact ordered item list and each submission quality",
            "assessment": "partially observed; improved over V6 but not perfectly Markov",
        },
        {
            "component": "automated-check queue",
            "v7_observed": "auto queue load",
            "still_hidden": "exact risk and delay composition",
            "assessment": "partially observed; possible residual POMDP structure",
        },
        {
            "component": "delayed incidents",
            "v7_observed": "incident queue load and mean severity",
            "still_hidden": "exact remaining delays and severities",
            "assessment": "partially observed; terminal liquidation reduces end-of-horizon bias",
        },
        {
            "component": "appeals",
            "v7_observed": "pending appeal count",
            "still_hidden": "exact appeal ages",
            "assessment": "partially observed; delayed appeals explicitly ablated",
        },
        {
            "component": "ecosystem feedback",
            "v7_observed": "trust, participation, false restrictions, unresolved incidents",
            "still_hidden": "individual developer identities",
            "assessment": "sufficient for aggregate simulation, not for real platform claims",
        },
    ]
    return pd.DataFrame(rows)


def bandit_validity_audit() -> pd.DataFrame:
    rows = [
        {
            "issue": "V6 contextual bandit reward attribution",
            "diagnosis": "The bandit updated on total step reward, which mixed current-action reward with consequences of earlier decisions.",
            "v7_correction": "Bandit updates use only reward_components.current_action_reward.",
            "implication": "The bandit can no longer learn from delayed consequences that it cannot assign to its own current action.",
        },
        {
            "issue": "Sequential evaluation",
            "diagnosis": "A bandit policy can still be evaluated in a sequential environment.",
            "v7_correction": "Evaluation reports total reward for all policies, but training remains non-sequential for the bandit.",
            "implication": "RL advantage is interpretable as value beyond contextual immediate-reward learning.",
        },
        {
            "issue": "Past-consequence reward",
            "diagnosis": "Delayed incidents, appeals and queue resolutions are not caused by the current decision.",
            "v7_correction": "Step info separates current_action_reward, delayed_past_reward and terminal_liquidation_reward.",
            "implication": "Reward components can be audited directly in output tables.",
        },
    ]
    return pd.DataFrame(rows)


def validity_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category": "simulation_assumption",
                "claim": "Harm rates, review accuracy, trust, participation, appeal and incident dynamics are simulated.",
                "status": "assumed",
                "manuscript_use": "Report as explicit model assumptions, not empirical estimates.",
            },
            {
                "category": "measured_result",
                "claim": "V7 measures reward components, convergence diagnostics, coverage, security, generativity and costs.",
                "status": "measured in V7",
                "manuscript_use": "Use as simulation evidence conditional on the model.",
            },
            {
                "category": "conditional_conclusion",
                "claim": "RL has value only if it beats corrected contextual bandit under delayed consequences and comparable safety.",
                "status": "tested in V7",
                "manuscript_use": "Use with paired differences, coverage diagnostics and ablations.",
            },
            {
                "category": "not_demonstrable",
                "claim": "RL improves real Google Play, Apple or Amazon governance.",
                "status": "not demonstrated",
                "manuscript_use": "Do not claim without logged state-action-outcome trajectories.",
            },
        ]
    )


def write_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(col) for col in df.columns]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            text = f"{value:.6f}".rstrip("0").rstrip(".")
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_winners(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return summary.sort_values(group_cols + ["mean_reward_mean"], ascending=[True] * len(group_cols) + [False]).groupby(
        group_cols
    ).head(1)


def top_line_answers(
    main_summary: pd.DataFrame,
    main_diffs: pd.DataFrame,
    q_diag: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    sequentiality_summary: pd.DataFrame,
) -> dict[str, str]:
    q_cb = main_diffs[
        main_diffs["metric"].eq("mean_reward")
        & main_diffs["comparison"].eq("q_learning_minus_contextual_bandit")
    ]
    seq = q_cb[q_cb["environment"].eq("sequential")]
    nonseq = q_cb[q_cb["environment"].eq("nonsequential")]
    seq_positive = int((seq["mean_diff"] > 0).sum())
    nonseq_positive = int((nonseq["mean_diff"] > 0).sum())
    seq_strict = int((seq["ci95_low"] > 0).sum())
    nonseq_strict = int((nonseq["ci95_low"] > 0).sum())
    winners = compact_winners(main_summary, ["environment", "scenario"])
    seq_rl_wins = int(winners[winners["environment"].eq("sequential")]["policy"].isin(["q_learning", "q_learning_fine", "linear_q_approx"]).sum())
    nonseq_rl_wins = int(winners[winners["environment"].eq("nonsequential")]["policy"].isin(["q_learning", "q_learning_fine", "linear_q_approx"]).sum())
    diag_best = q_diag.sort_values("eval_mean_reward", ascending=False).head(1)
    convergence_note = "No Q-learning diagnostic rows were produced."
    if not diag_best.empty:
        row = diag_best.iloc[0]
        convergence_note = (
            f"Best Q-learning diagnostic setting: {row['scenario']}, budget={int(row['budget'])}, "
            f"gamma={row['gamma']}, alpha={row['alpha']}, eval mean reward={row['eval_mean_reward']:.4f}, "
            f"unseen-state rate={row['eval_unseen_state_rate']:.4f}."
        )
    full_rl = ablation_summary[
        ablation_summary["mechanism"].eq("full_sequential") & ablation_summary["policy"].eq("q_learning")
    ][["scenario", "mean_reward_mean"]].rename(columns={"mean_reward_mean": "full_reward"})
    losses = []
    for mechanism in sorted(set(ablation_summary["mechanism"]) - {"full_sequential"}):
        sub = ablation_summary[ablation_summary["mechanism"].eq(mechanism) & ablation_summary["policy"].eq("q_learning")]
        merged = full_rl.merge(sub[["scenario", "mean_reward_mean"]], on="scenario", how="inner")
        if not merged.empty:
            merged["loss"] = merged["full_reward"] - merged["mean_reward_mean"]
            losses.append((mechanism, float(merged["loss"].mean())))
    losses = sorted(losses, key=lambda x: x[1], reverse=True)
    mechanism_note = (
        f"Largest average Q-learning reward loss after ablation: {losses[0][0]} ({losses[0][1]:.4f})."
        if losses
        else "No mechanism loss could be estimated."
    )
    seq_interaction = sequentiality_summary[
        sequentiality_summary["policy"].isin(["contextual_bandit", "q_learning", "q_learning_fine"])
    ][["mechanism", "policy", "mean_reward_mean"]]
    interaction_note = "Sequentiality interaction table is available in v7_sequentiality_summary.csv."
    if not seq_interaction.empty:
        pivot = seq_interaction.pivot_table(index="mechanism", columns="policy", values="mean_reward_mean", aggfunc="mean")
        if {"q_learning", "contextual_bandit"}.issubset(pivot.columns):
            pivot["q_minus_bandit"] = pivot["q_learning"] - pivot["contextual_bandit"]
            best = pivot["q_minus_bandit"].idxmax()
            interaction_note = f"The largest Q-learning minus bandit interaction appears at {best}: {pivot.loc[best, 'q_minus_bandit']:.4f}."
    return {
        "q1": (
            f"In the six main scenarios, Q-learning or another learned RL variant is the top policy in {seq_rl_wins}/6 "
            f"sequential scenarios. Q-learning beats corrected contextual bandit in {seq_positive}/6 sequential paired "
            f"mean differences, with strictly positive 95% CIs in {seq_strict}/6."
        ),
        "q1b": convergence_note,
        "q2": (
            f"In the non-sequential environment, learned RL variants are top in {nonseq_rl_wins}/6 scenarios. "
            f"Q-learning beats corrected contextual bandit in {nonseq_positive}/6 non-sequential paired differences, "
            f"with strictly positive 95% CIs in {nonseq_strict}/6."
        ),
        "q3": mechanism_note,
        "q4": interaction_note,
    }


def scientific_report_md(
    empirical: dict[str, Any],
    main_summary: pd.DataFrame,
    main_diffs: pd.DataFrame,
    q_diag: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    sequentiality_summary: pd.DataFrame,
    security_comparable: pd.DataFrame,
    reward_attr: pd.DataFrame,
) -> str:
    answers = top_line_answers(main_summary, main_diffs, q_diag, ablation_summary, sequentiality_summary)
    seq_winners = compact_winners(main_summary[main_summary["environment"].eq("sequential")], ["environment", "scenario"])
    nonseq_winners = compact_winners(main_summary[main_summary["environment"].eq("nonsequential")], ["environment", "scenario"])
    key_diffs = main_diffs[
        main_diffs["metric"].eq("mean_reward")
        & main_diffs["comparison"].isin(
            [
                "q_learning_minus_contextual_bandit",
                "q_learning_fine_minus_contextual_bandit",
                "linear_q_approx_minus_contextual_bandit",
                "q_learning_minus_security_rule",
                "q_learning_minus_heuristic_score_threshold",
            ]
        )
    ][["environment", "scenario", "comparison", "mean_diff", "ci95_low", "ci95_high", "n_pairs"]]
    q_diag_short = q_diag.sort_values("eval_mean_reward", ascending=False).head(12)[
        [
            "scenario",
            "budget",
            "hparam_id",
            "alpha",
            "gamma",
            "visited_states",
            "visited_state_actions",
            "state_action_coverage_within_seen_states",
            "eval_mean_reward",
            "eval_unseen_state_rate",
        ]
    ]
    seq_short = sequentiality_summary[
        sequentiality_summary["policy"].isin(["contextual_bandit", "q_learning", "q_learning_fine", "linear_q_approx"])
    ][["mechanism", "policy", "mean_reward_mean", "security_incidents_mean", "governance_expenditure_mean"]]
    attr_short = reward_attr[
        reward_attr["policy"].isin(["contextual_bandit", "q_learning", "q_learning_fine", "linear_q_approx"])
    ].head(24)
    security_short = security_comparable[security_comparable["focal_policy"].eq("q_learning")].head(36)
    lines = [
        "# V7 Scientific Validation of RL-Specific Value",
        "",
        "## Scientific status",
        "",
        f"The documentary governance-event register contains {empirical['events_total']} coded governance events, including {empirical['google_play_events']} Google Play events and {empirical['direct_ml_events']} direct ML-governance events. V7 still uses these events only to motivate mechanisms, not to estimate transition probabilities.",
        "",
        "V7 corrects the V6 reward-attribution problem by separating current-action reward from delayed consequences of past decisions and terminal liquidation of unresolved obligations.",
        "",
        "## Direct answers",
        "",
        "1. Does the V6 Q-learning disadvantage reflect learning failure or absence of sequential gain?",
        "",
        answers["q1"],
        answers["q1b"],
        "The result should be interpreted jointly with coverage diagnostics. If coverage is sparse and performance remains below the corrected bandit, V7 supports both a learning-capacity limitation and weak evidence of robust sequential gain in the current simulator.",
        "",
        "2. Does RL benefit more from delayed consequences than contextual bandit?",
        "",
        answers["q2"],
        "The corrected contextual bandit no longer trains on delayed-past reward. Any remaining Q-learning advantage is therefore attributable to total-return learning rather than reward leakage.",
        "",
        "3. Which mechanisms explain the differences?",
        "",
        answers["q3"],
        "Use the ablation tables for mechanism-specific interpretation; full rankings alone are insufficient.",
        "",
        "4. Is there a complexity or temporal-dependence threshold where sequential learning becomes useful?",
        "",
        answers["q4"],
        "The sequentiality-scale experiment estimates this interaction directly rather than inferring it from policy ranks.",
        "",
        "5. Are the results robust?",
        "",
        "Robustness is limited to the simulated parameter grid. The oracle upper bound is included only as a non-deployable theoretical ceiling with hidden information.",
        "",
        "## Sequential winners",
        "",
        df_to_markdown(
            seq_winners[
                ["scenario", "policy", "mean_reward_mean", "security_incidents_mean", "governance_expenditure_mean"]
            ]
        ),
        "",
        "## Non-sequential winners",
        "",
        df_to_markdown(
            nonseq_winners[
                ["scenario", "policy", "mean_reward_mean", "security_incidents_mean", "governance_expenditure_mean"]
            ]
        ),
        "",
        "## Key paired differences",
        "",
        df_to_markdown(key_diffs),
        "",
        "## Q-learning convergence diagnostics",
        "",
        df_to_markdown(q_diag_short),
        "",
        "## Sequentiality-scale interaction",
        "",
        df_to_markdown(seq_short),
        "",
        "## Reward attribution excerpt",
        "",
        df_to_markdown(
            attr_short[
                [
                    "environment",
                    "scenario",
                    "policy",
                    "current_action_reward_total_mean",
                    "delayed_past_reward_total_mean",
                    "terminal_liquidation_reward_total_mean",
                    "abs_delayed_share_of_total_reward",
                ]
            ]
        ),
        "",
        "## Security-comparable ranking excerpt",
        "",
        df_to_markdown(
            security_short[
                [
                    "environment",
                    "scenario",
                    "policy",
                    "security_incidents_mean",
                    "mean_reward_mean",
                    "security_comparable_rank",
                ]
            ]
        ),
        "",
        "## Interpretation limits",
        "",
        "- No result is presented as real-platform evidence.",
        "- The oracle policy has access to hidden harmfulness and is only a theoretical upper bound.",
        "- Linear function approximation is included as a lightweight RL approximation baseline; DQN is not used because the workspace does not require a deep-learning stack and the state/action space is small enough to audit tabular coverage first.",
        "- A negative result is preserved if Q-learning remains below the corrected contextual bandit after reward attribution, terminal liquidation and convergence checks.",
        "",
    ]
    return "\n".join(lines)


def methods_report_md() -> str:
    return "\n".join(
        [
            "# V7 Methodological Corrections",
            "",
            "## Reward attribution",
            "",
            "- `current_action_reward`: immediate reward and costs caused by the current governance action.",
            "- `delayed_past_reward`: incidents, review outcomes, automated checks, and appeals generated by earlier decisions and materializing during the current transition.",
            "- `collective_state_reward`: reserved for explicit aggregate state rewards; currently zero because trust and participation affect future submissions rather than direct reward.",
            "- `terminal_liquidation_reward`: end-of-episode settlement of pending incidents, queues, appeals, and review obligations.",
            "",
            "The contextual bandit is trained only on `current_action_reward`. Q-learning and linear Q approximation are trained on total transition reward.",
            "",
            "## Markov structure",
            "",
            "V7 expands the observation vector from 11 to 17 dimensions by adding automated-check load, review-queue risk/delay summaries, incident-queue load/severity, and appeal load. The environment remains an aggregate simulation and may still be partially observed because exact queue contents are summarized rather than fully exposed.",
            "",
            "## Terminal liquidation",
            "",
            "At truncation, all pending incidents, checks, review items, and appeals are liquidated. This avoids hiding deferred costs or benefits beyond the finite experimental horizon.",
            "",
            "## Algorithms",
            "",
            "- Six V6 baselines are preserved.",
            "- The contextual bandit is corrected for delayed-reward attribution.",
            "- Q-learning is evaluated with adjusted and fine discretizations.",
            "- A lightweight linear Q approximator is included to test whether tabular sparsity explains poor RL performance.",
            "- An oracle upper bound uses hidden harmfulness and is never treated as deployable governance.",
            "",
        ]
    )


def make_figures(
    main_summary: pd.DataFrame,
    main_diffs: pd.DataFrame,
    q_diag: pd.DataFrame,
    sequentiality_summary: pd.DataFrame,
) -> list[str]:
    made: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return made
    FIGURES.mkdir(parents=True, exist_ok=True)

    agg = main_summary.groupby(["environment", "policy"], as_index=False)["mean_reward_mean"].mean()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    policies = POLICIES
    x = np.arange(len(policies))
    width = 0.36
    for idx, environment in enumerate(["nonsequential", "sequential"]):
        values = [
            float(agg[(agg["environment"].eq(environment)) & (agg["policy"].eq(policy))]["mean_reward_mean"].mean())
            for policy in policies
        ]
        ax.bar(x + (idx - 0.5) * width, values, width, label=environment)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=35, ha="right")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("V7 policy performance by environment")
    ax.legend()
    fig.tight_layout()
    path = FIGURES / "v7_mean_reward_by_environment.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    diff = main_diffs[
        main_diffs["metric"].eq("mean_reward")
        & main_diffs["comparison"].isin(
            ["q_learning_minus_contextual_bandit", "q_learning_fine_minus_contextual_bandit"]
        )
    ].copy()
    diff["label"] = diff["comparison"].str.replace("_minus_contextual_bandit", "", regex=False) + "\n" + diff["environment"] + "\n" + diff["scenario"]
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(diff))
    err_low = diff["mean_diff"] - diff["ci95_low"]
    err_high = diff["ci95_high"] - diff["mean_diff"]
    ax.errorbar(diff["mean_diff"], y, xerr=[err_low, err_high], fmt="o", color="#2f5d8c", ecolor="#8899aa")
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(diff["label"], fontsize=7)
    ax.set_xlabel("Paired mean reward difference versus corrected contextual bandit")
    ax.set_title("V7 RL versus corrected bandit")
    fig.tight_layout()
    path = FIGURES / "v7_rl_vs_corrected_bandit.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(9, 5))
    diag = q_diag.groupby(["budget"], as_index=False)["eval_mean_reward"].mean()
    ax.plot(diag["budget"], diag["eval_mean_reward"], marker="o")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean validation reward")
    ax.set_title("Q-learning budget sensitivity")
    fig.tight_layout()
    path = FIGURES / "v7_q_learning_budget_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    seq = sequentiality_summary[
        sequentiality_summary["policy"].isin(["contextual_bandit", "q_learning", "q_learning_fine", "linear_q_approx"])
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for policy, g in seq.groupby("policy"):
        ordered = g.sort_values("sequentiality_scale")
        ax.plot(ordered["sequentiality_scale"], ordered["mean_reward_mean"], marker="o", label=policy)
    ax.set_xlabel("Sequentiality scale")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("Policy interaction with sequentiality")
    ax.legend()
    fig.tight_layout()
    path = FIGURES / "v7_sequentiality_interaction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))
    return made


def main() -> None:
    start = datetime.now(timezone.utc)
    for path in [TABLES, REPORTS, FIGURES, LOGS]:
        path.mkdir(parents=True, exist_ok=True)

    empirical = empirical_reference()
    write_csv(TABLES / "v7_markov_audit.csv", markov_audit_table())
    write_csv(TABLES / "v7_bandit_validity_audit.csv", bandit_validity_audit())
    write_csv(TABLES / "v7_validity_matrix.csv", validity_matrix())
    (REPORTS / "methodological_corrections.md").write_text(methods_report_md(), encoding="utf-8")

    main_frames = []
    diag_frames = []
    curve_frames = []
    for environment, mechanisms in [("nonsequential", NONSEQUENTIAL), ("sequential", FULL_SEQUENTIAL)]:
        reps, diags, curves = run_policy_set(
            SCENARIOS,
            environment,
            mechanisms,
            MAIN_TRAIN_SEEDS,
            MAIN_EVAL_SEEDS,
            episodes=180,
            label="main",
        )
        main_frames.append(reps)
        diag_frames.append(diags)
        curve_frames.append(curves)
    main_replications = pd.concat(main_frames, ignore_index=True)
    training_diagnostics = pd.concat(diag_frames, ignore_index=True)
    training_curves = pd.concat(curve_frames, ignore_index=True)
    main_summary = summarize_results(main_replications, ["environment", "scenario", "policy"])
    main_actions = action_distributions(main_replications, ["environment", "scenario", "policy"])
    main_diffs = pd.concat(
        [
            paired_differences(
                main_replications,
                ["environment", "scenario"],
                "q_learning",
                ["contextual_bandit", "security_rule", "heuristic_score_threshold"],
            ),
            paired_differences(
                main_replications,
                ["environment", "scenario"],
                "q_learning_fine",
                ["contextual_bandit", "security_rule", "heuristic_score_threshold"],
            ),
            paired_differences(
                main_replications,
                ["environment", "scenario"],
                "linear_q_approx",
                ["contextual_bandit", "security_rule", "heuristic_score_threshold"],
            ),
        ],
        ignore_index=True,
    )
    reward_attr = reward_attribution_summary(main_replications)
    security_comp = security_comparable_summary(main_summary, "q_learning")

    write_csv(TABLES / "v7_main_replications.csv", main_replications)
    write_csv(TABLES / "v7_main_summary.csv", main_summary)
    write_csv(TABLES / "v7_main_action_distributions.csv", main_actions)
    write_csv(TABLES / "v7_paired_differences.csv", main_diffs)
    write_csv(TABLES / "v7_reward_attribution_summary.csv", reward_attr)
    write_csv(TABLES / "v7_training_diagnostics.csv", training_diagnostics)
    write_csv(TABLES / "v7_training_curves.csv", training_curves)
    write_csv(TABLES / "v7_security_comparable_summary.csv", security_comp)

    ablation_frames = []
    for ablation in ABLATIONS:
        reps, _, _ = run_policy_set(
            [SCENARIOS[0], SCENARIOS[1], SCENARIOS[5], STRESS_SCENARIOS[1]],
            "sequential",
            ablation,
            ABLATION_TRAIN_SEEDS,
            ABLATION_EVAL_SEEDS,
            episodes=140,
            label="ablation",
            policies=["contextual_bandit", "q_learning", "q_learning_fine", "linear_q_approx", "heuristic_score_threshold"],
        )
        ablation_frames.append(reps)
    ablation_replications = pd.concat(ablation_frames, ignore_index=True)
    ablation_summary = summarize_results(ablation_replications, ["mechanism", "scenario", "policy"])
    ablation_diffs = pd.concat(
        [
            paired_differences(
                ablation_replications,
                ["mechanism", "scenario"],
                "q_learning",
                ["contextual_bandit", "heuristic_score_threshold"],
            ),
            paired_differences(
                ablation_replications,
                ["mechanism", "scenario"],
                "q_learning_fine",
                ["contextual_bandit", "heuristic_score_threshold"],
            ),
        ],
        ignore_index=True,
    )
    write_csv(TABLES / "v7_ablation_replications.csv", ablation_replications)
    write_csv(TABLES / "v7_ablation_summary.csv", ablation_summary)
    write_csv(TABLES / "v7_ablation_paired_differences.csv", ablation_diffs)

    seq_frames = []
    for mechanism in SEQUENTIALITY_LEVELS:
        env_name = "nonsequential" if mechanism.sequentiality_scale == 0 else "sequential"
        reps, _, _ = run_policy_set(
            [SCENARIOS[0], STRESS_SCENARIOS[1], STRESS_SCENARIOS[2]],
            env_name,
            mechanism if env_name == "sequential" else NONSEQUENTIAL,
            ABLATION_TRAIN_SEEDS,
            ABLATION_EVAL_SEEDS,
            episodes=140,
            label="sequentiality",
            policies=["contextual_bandit", "q_learning", "q_learning_fine", "linear_q_approx", "heuristic_score_threshold"],
        )
        reps["mechanism"] = mechanism.name
        reps["sequentiality_scale"] = mechanism.sequentiality_scale
        seq_frames.append(reps)
    sequentiality_replications = pd.concat(seq_frames, ignore_index=True)
    sequentiality_summary = summarize_results(sequentiality_replications, ["mechanism", "sequentiality_scale", "policy"])
    write_csv(TABLES / "v7_sequentiality_replications.csv", sequentiality_replications)
    write_csv(TABLES / "v7_sequentiality_summary.csv", sequentiality_summary)

    stress_reps, _, _ = run_policy_set(
        STRESS_SCENARIOS + SENSITIVITY_SCENARIOS,
        "sequential",
        FULL_SEQUENTIAL,
        DIAGNOSTIC_TRAIN_SEEDS,
        DIAGNOSTIC_EVAL_SEEDS,
        episodes=160,
        label="stress_sensitivity",
    )
    stress_summary = summarize_results(stress_reps, ["environment", "scenario", "policy"])
    write_csv(TABLES / "v7_stress_sensitivity_replications.csv", stress_reps)
    write_csv(TABLES / "v7_stress_sensitivity_summary.csv", stress_summary)

    q_diag, q_curves = run_q_diagnostics()
    write_csv(TABLES / "v7_q_convergence_diagnostics.csv", q_diag)
    write_csv(TABLES / "v7_q_convergence_curves.csv", q_curves)

    figures = make_figures(main_summary, main_diffs, q_diag, sequentiality_summary)
    (REPORTS / "scientific_report.md").write_text(
        scientific_report_md(
            empirical,
            main_summary,
            main_diffs,
            q_diag,
            ablation_summary,
            sequentiality_summary,
            security_comp,
            reward_attr,
        ),
        encoding="utf-8",
    )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python": sys.version,
        "gymnasium_installed": GYMNASIUM_INSTALLED,
        "policies": POLICIES,
        "v6_baselines_preserved": V6_BASELINES,
        "train_seeds": {
            "main": MAIN_TRAIN_SEEDS,
            "ablation": ABLATION_TRAIN_SEEDS,
            "diagnostic": DIAGNOSTIC_TRAIN_SEEDS,
        },
        "eval_seeds": {
            "main": MAIN_EVAL_SEEDS,
            "ablation": ABLATION_EVAL_SEEDS,
            "diagnostic": DIAGNOSTIC_EVAL_SEEDS,
        },
        "scenarios": [config.__dict__ for config in SCENARIOS],
        "stress_scenarios": [config.__dict__ for config in STRESS_SCENARIOS],
        "sensitivity_scenarios": [config.__dict__ for config in SENSITIVITY_SCENARIOS],
        "ablations": [ablation.__dict__ for ablation in ABLATIONS],
        "empirical_reference": empirical,
        "figures": figures,
        "input_hashes": {
            "governance_events_final.csv": sha256_file(ROOT / "data" / "processed" / "governance_events_final.csv"),
            "v6_script": sha256_file(ROOT / "experiments_v6" / "scripts" / "run_experiments_v6.py"),
            "v6_report": sha256_file(ROOT / "experiments_v6" / "outputs" / "reports" / "scientific_report.md"),
        },
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    end = datetime.now(timezone.utc)
    (LOGS / "run_log.txt").write_text(
        "\n".join(
            [
                "experiments_v7 run log",
                f"start_utc={start.isoformat()}",
                f"end_utc={end.isoformat()}",
                f"main_replications={len(main_replications)}",
                f"ablation_replications={len(ablation_replications)}",
                f"sequentiality_replications={len(sequentiality_replications)}",
                f"stress_sensitivity_replications={len(stress_reps)}",
                f"q_diagnostic_rows={len(q_diag)}",
                "outputs:",
                "- outputs/tables/v7_main_summary.csv",
                "- outputs/tables/v7_paired_differences.csv",
                "- outputs/tables/v7_q_convergence_diagnostics.csv",
                "- outputs/tables/v7_reward_attribution_summary.csv",
                "- outputs/reports/scientific_report.md",
                "- outputs/reports/methodological_corrections.md",
                "- outputs/figures/*.png",
                "- outputs/run_manifest.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("experiments_v7 complete")
    print(REPORTS / "scientific_report.md")


if __name__ == "__main__":
    main()
