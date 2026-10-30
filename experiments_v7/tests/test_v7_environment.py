from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments_v7" / "scripts"))

from run_experiments_v7 import (  # noqa: E402
    FULL_SEQUENTIAL,
    NONSEQUENTIAL,
    GovernanceEnvV7,
    ScenarioConfig,
    Submission,
    evaluate_policy,
    train_contextual_bandit,
    train_q_learning,
)


class V7EnvironmentTests(unittest.TestCase):
    def test_reward_components_separate_current_and_delayed_rewards(self) -> None:
        cfg = ScenarioConfig(name="test", horizon=4, incident_delay_min=1, incident_delay_max=1)
        env = GovernanceEnvV7(cfg, "sequential", FULL_SEQUENTIAL, seed=11)
        env.incident_queue.append((1, 0.9))
        env.current_submission = Submission(
            risk=0.2,
            uncertainty=0.2,
            harmful=False,
            developer_quality=0.8,
            diversity_group=0,
        )
        _, reward, _, _, info = env.step(0)
        components = info["reward_components"]
        self.assertGreater(components["current_action_reward"], 0.0)
        self.assertLess(components["delayed_past_reward"], 0.0)
        self.assertAlmostEqual(reward, sum(components.values()))

    def test_bandit_training_updates_on_current_action_reward_only(self) -> None:
        cfg = ScenarioConfig(name="test", horizon=8)
        learner = train_contextual_bandit(cfg, "sequential", FULL_SEQUENTIAL, seed=12, episodes=4)
        self.assertEqual(learner.diagnostics["bandit_training_reward"], "current_action_reward_only")
        self.assertGreaterEqual(learner.diagnostics["visited_states"], 1)
        self.assertGreaterEqual(learner.diagnostics["visited_state_actions"], 1)

    def test_terminal_liquidation_clears_pending_items(self) -> None:
        cfg = ScenarioConfig(name="test", horizon=1, terminal_liquidation=True)
        env = GovernanceEnvV7(cfg, "sequential", FULL_SEQUENTIAL, seed=13)
        env.incident_queue.append((5, 0.8))
        env.appeal_queue.append((5, True))
        env.pending_appeals = 1
        env.current_submission = Submission(
            risk=0.2,
            uncertainty=0.2,
            harmful=False,
            developer_quality=0.8,
            diversity_group=0,
        )
        _, _, _, truncated, info = env.step(0)
        self.assertTrue(truncated)
        self.assertLess(info["reward_components"]["terminal_liquidation_reward"], 1.0)
        self.assertEqual(len(env.incident_queue), 0)
        self.assertEqual(len(env.appeal_queue), 0)

    def test_nonsequential_observation_uses_same_information_shape(self) -> None:
        cfg = ScenarioConfig(name="test", horizon=5)
        seq_env = GovernanceEnvV7(cfg, "sequential", FULL_SEQUENTIAL, seed=14)
        nonseq_env = GovernanceEnvV7(cfg, "nonsequential", NONSEQUENTIAL, seed=14)
        seq_obs, _ = seq_env.reset(seed=14)
        nonseq_obs, _ = nonseq_env.reset(seed=14)
        self.assertEqual(seq_obs.shape, nonseq_obs.shape)
        self.assertEqual(seq_obs.shape[0], 17)

    def test_q_learning_evaluation_reports_unseen_state_rate(self) -> None:
        cfg = ScenarioConfig(name="test", horizon=10)
        learner = train_q_learning(cfg, "sequential", FULL_SEQUENTIAL, seed=15, episodes=4)
        row = evaluate_policy(
            cfg,
            "sequential",
            FULL_SEQUENTIAL,
            "q_learning",
            seed=16,
            train_seed=15,
            learned=learner,
        )
        self.assertIn("unseen_eval_state_rate", row)
        self.assertGreaterEqual(row["unseen_eval_state_rate"], 0.0)
        self.assertLessEqual(row["unseen_eval_state_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
