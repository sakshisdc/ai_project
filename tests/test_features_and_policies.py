"""Unit tests for features, policies, state equality, agents, and edge cases."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from adaptive_minimax.agents import FixedDepthAgent, IterativeDeepeningAgent, RuleAdaptiveAgent, MLAdaptiveAgent
from adaptive_minimax.evaluation import analyze, clear_cache, static_eval
from adaptive_minimax.experiments import load_calibration, play_game
from adaptive_minimax.fastmodels import compile_estimator
from adaptive_minimax.features import FEATURE_NAMES, IDX, extract_features, tactical_volatility
from adaptive_minimax.game import State
from adaptive_minimax.policies import MLPolicy, RuleBasedPolicy


def test_state_equality_and_hash():
    s1 = State.from_moves([3, 2, 4])
    s2 = State.from_moves([3, 2, 4])
    s3 = State.from_moves([3, 2, 1])

    assert s1 == s2
    assert s1 != s3
    assert s1 != "not_a_state"
    assert hash(s1) == hash(s2)
    assert len({s1, s2, s3}) == 2


def test_feature_extraction_and_names():
    s = State.from_moves([3, 3, 2, 4])
    parent = State.from_moves([3, 3, 2])
    feats = extract_features(s, static_eval(parent))

    assert len(feats) == len(FEATURE_NAMES) == 9
    assert abs(feats[IDX["phase"]] - 4 / 42.0) < 1e-9
    assert feats[IDX["branching"]] == 7.0
    assert abs(feats[IDX["static_eval"]]) <= 0.8
    assert feats[IDX["abs_eval"]] == abs(feats[IDX["static_eval"]])
    assert feats[IDX["swing"]] >= 0.0
    assert tactical_volatility(feats) == int(feats[IDX["my_win_now"]] + feats[IDX["opp_win_now"]])


def test_rule_based_policy():
    rule = RuleBasedPolicy(swing_threshold=0.5)
    # Tactical position: immediate win threat
    win_pos = State.from_moves([0, 0, 1, 1, 2, 2])
    parent = State.from_moves([0, 0, 1, 1, 2])
    assert rule.should_extend(win_pos, parent, -1.0, 1.0) is True

    # Quiet position: root opening
    quiet = State.from_moves([3])
    root = State()
    assert rule.should_extend(quiet, root, -1.0, 1.0) is False


def test_ml_policy():
    def mock_predict(x):
        return 0.8 if x[IDX["phase"]] > 0.1 else 0.2

    pol = MLPolicy(mock_predict, threshold=0.5)
    s_late = State.from_moves([3, 3, 2, 2, 4, 4])  # 6 plies > 0.1
    s_early = State.from_moves([3])               # 1 ply < 0.1

    assert pol.should_extend(s_late, None, -1.0, 1.0) is True
    assert pol.should_extend(s_early, None, -1.0, 1.0) is False


def test_self_play_tournament_no_collision():
    agent = IterativeDeepeningAgent()
    res = play_game(agent, agent, budget=100, opening=[3, 3], a_first=True)
    assert res["score"] in (0.0, 0.5, 1.0)
    assert res["plies"] >= 2
    assert res["nodes_a"] <= 100
    assert res["compliance_a"] == 1.0


def test_fixed_depth_agent_calibration_lookup():
    agent = FixedDepthAgent(calibration={1: 10, 2: 50, 3: 200, 4: 800})
    assert agent.depth_for(5) == 1
    assert agent.depth_for(50) == 2
    assert agent.depth_for(199) == 2
    assert agent.depth_for(200) == 3
    assert agent.depth_for(1000) == 4


def test_single_class_classifier_compilation():
    X = np.zeros((10, 2))
    y = np.ones(10, dtype=int)
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    fn = compile_estimator(clf)
    assert fn([0.0, 0.0]) == 1.0


def test_cache_clearing():
    s = State.from_moves([3, 2, 4])
    v1 = static_eval(s)
    clear_cache()
    v2 = static_eval(s)
    assert v1 == v2
