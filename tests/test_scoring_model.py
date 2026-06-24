"""
scoring_model 模块测试 — 使用 DEFAULT_WEIGHTS 动态计算期望值
"""
import pytest
from scoring_model import score_risk, DEFAULT_WEIGHTS

W = DEFAULT_WEIGHTS  # shorthand


def _feat(**kw) -> dict[str, float]:
    """构造 17 维特征 dict，未指定使用安全默认值（避免误触发间隔反归一化）"""
    defaults = {k: 0.0 for k in W}
    # 安全默认：无交易时这些特征不应贡献异常值
    defaults["tx_interval_mean_sec"] = 3600.0  # 1小时→归一化为0
    defaults.update(kw)
    return defaults


# ── 基础评分 ──────────────────────────────────────────

def test_score_all_zero():
    result = score_risk(_feat())
    assert result["score"] == pytest.approx(0.0)
    assert result["level"] == "LOW"


def test_score_all_one():
    result = score_risk(_feat(
        device_reuse_ratio=1.0, ip_change_freq=1.0, tx_freq=100.0,
        login_fail_ratio=1.0, amount_anomaly_score=1.0,
        behavior_time_anomaly=1.0, multi_region_risk=1.0,
    ))
    assert result["score"] == pytest.approx(100.0)
    assert result["level"] == "HIGH"


def test_score_mid_range():
    """中等风险 + 规则增强。17 维权重分散后钳制不再触发 100"""
    result = score_risk(_feat(
        device_reuse_ratio=0.5, ip_change_freq=0.5, tx_freq=50.0,
        login_fail_ratio=0.5, amount_anomaly_score=0.5,
    ))
    assert result["score"] > 80.0  # 仍然高分
    assert result["level"] == "HIGH"


# ── 规则增强 ──────────────────────────────────────────

def test_rule1_device_tx():
    """device_reuse>0.9 且 tx_freq>50 → +10；同时 device_reuse>0.2 → 规则5 +19"""
    result = score_risk(_feat(device_reuse_ratio=0.95, tx_freq=60.0))
    base = (W["device_reuse_ratio"] * 0.95 + W["tx_freq"] * min(1.0, 60.0 / 100.0)) * 100
    assert result["score"] == pytest.approx(min(100.0, base + 29.0))


def test_rule2_ip_login():
    """ip_change>0.8 且 login_fail>0.5 → +10"""
    result = score_risk(_feat(ip_change_freq=0.85, login_fail_ratio=0.6))
    base = (W["ip_change_freq"] * 0.85 + W["login_fail_ratio"] * 0.6) * 100
    assert result["score"] == pytest.approx(min(100.0, base + 10.0))


def test_rule_boost_both():
    """规则1+2+5+7同时触发 → +54 (规则1+10, 规则2+10, 规则5+19, 规则7+15)"""
    result = score_risk(_feat(
        device_reuse_ratio=0.95, tx_freq=60.0,
        ip_change_freq=0.85, login_fail_ratio=0.6,
    ))
    base = (W["device_reuse_ratio"] * 0.95 + W["ip_change_freq"] * 0.85
            + W["tx_freq"] * min(1.0, 60.0 / 100.0)
            + W["login_fail_ratio"] * 0.6) * 100
    expected = min(100.0, base + 54.0)
    assert result["score"] == pytest.approx(expected)


def test_rule_edge_boundaries():
    """边界值：规则1/2刚好不触发，但规则5/7触发→+34"""
    result = score_risk(_feat(
        device_reuse_ratio=0.9, tx_freq=50.0,
        ip_change_freq=0.8, login_fail_ratio=0.5,
    ))
    base = (W["device_reuse_ratio"] * 0.9 + W["ip_change_freq"] * 0.8
            + W["tx_freq"] * min(1.0, 50.0 / 100.0)
            + W["login_fail_ratio"] * 0.5) * 100
    assert result["score"] == pytest.approx(base + 34.0)


def test_rule3_amount_only():
    """amount_anomaly>0.7 → +15"""
    result = score_risk(_feat(amount_anomaly_score=0.8))
    base = W["amount_anomaly_score"] * 0.8 * 100
    assert result["score"] == pytest.approx(min(100.0, base + 15.0))


def test_rule4_login_fail_only():
    """login_fail>0.7 → +20"""
    result = score_risk(_feat(login_fail_ratio=0.8))
    base = W["login_fail_ratio"] * 0.8 * 100
    assert result["score"] == pytest.approx(min(100.0, base + 20.0))


def test_rule5_device_reuse_only():
    """device_reuse>0.2 → +19"""
    result = score_risk(_feat(device_reuse_ratio=0.3))
    base = W["device_reuse_ratio"] * 0.3 * 100
    assert result["score"] == pytest.approx(min(100.0, base + 19.0))


def test_rule6_ip_amount():
    """ip_change>0.1 且 amount_anomaly>0.3 → +15"""
    result = score_risk(_feat(ip_change_freq=0.2, amount_anomaly_score=0.5))
    base = (W["ip_change_freq"] * 0.2 + W["amount_anomaly_score"] * 0.5) * 100
    assert result["score"] == pytest.approx(base + 15.0)


def test_rule7_device_login():
    """device_reuse>0.15 且 login_fail>0.3 → +15"""
    result = score_risk(_feat(device_reuse_ratio=0.2, login_fail_ratio=0.4))
    base = (W["device_reuse_ratio"] * 0.2 + W["login_fail_ratio"] * 0.4) * 100
    assert result["score"] == pytest.approx(base + 15.0)


def test_rule8_tx_amount():
    """tx_freq>10 且 amount_anomaly>0.3 → +15"""
    result = score_risk(_feat(tx_freq=15.0, amount_anomaly_score=0.5))
    base = (W["tx_freq"] * min(1.0, 15.0 / 100.0) + W["amount_anomaly_score"] * 0.5) * 100
    assert result["score"] == pytest.approx(base + 15.0)


def test_rule9_behavior_tx():
    """behavior_time>0.3 且 tx_freq>20 → +15"""
    result = score_risk(_feat(behavior_time_anomaly=0.4, tx_freq=25.0))
    base = (W["behavior_time_anomaly"] * 0.4 + W["tx_freq"] * min(1.0, 25.0 / 100.0)) * 100
    assert result["score"] == pytest.approx(base + 15.0)


def test_rule10_multi_region():
    """multi_region>0 → +12"""
    result = score_risk(_feat(multi_region_risk=1.0))
    base = W["multi_region_risk"] * 1.0 * 100
    assert result["score"] == pytest.approx(base + 12.0)


# ── 边界值 ──────────────────────────────────────────

def test_score_clamp_upper():
    result = score_risk(_feat(
        device_reuse_ratio=1.0, ip_change_freq=1.0, tx_freq=200.0,
        login_fail_ratio=1.0, amount_anomaly_score=1.0,
        behavior_time_anomaly=1.0, multi_region_risk=1.0,
    ))
    assert result["score"] == pytest.approx(100.0)
    assert result["level"] == "HIGH"


def test_score_clamp_lower():
    result = score_risk(_feat(
        device_reuse_ratio=-0.5, ip_change_freq=-0.5, tx_freq=-10.0,
        login_fail_ratio=-0.5, amount_anomaly_score=-0.5,
    ))
    assert result["score"] == pytest.approx(0.0)
    assert result["level"] == "LOW"


def test_tx_freq_normalization():
    """tx_freq=200 归一化后=1.0"""
    result = score_risk(_feat(tx_freq=200.0))
    assert result["score"] == pytest.approx(W["tx_freq"] * 1.0 * 100)
    # tx_freq=200 > 10 触发规则8（但需要 amount_anomaly>0.3，这里 amount=0 不触发）


# ── 等级映射 ──────────────────────────────────────────

@pytest.mark.parametrize("score_value,expected_level,expected_range", [
    (0.0, "LOW", "0~29"),
    (15.0, "LOW", "0~29"),
    (29.0, "LOW", "0~29"),
    (30.0, "MEDIUM", "30~70"),
    (50.0, "MEDIUM", "30~70"),
    (70.0, "MEDIUM", "30~70"),
    (71.0, "HIGH", "71~100"),
    (85.0, "HIGH", "71~100"),
    (100.0, "HIGH", "71~100"),
])
def test_level_mapping(score_value, expected_level, expected_range):
    # 构造精确分数：用自定义权重(amount=1.0)直接控制分数，规避规则触发
    features = _feat(amount_anomaly_score=score_value / 100.0)
    custom_w = {"amount_anomaly_score": 1.0}
    result = score_risk(features, weights=custom_w)
    assert result["level"] == expected_level, \
        f"score={result['score']:.1f} expected_level={expected_level}"
    assert result["level_range"] == expected_range


# ── 输出结构 ──────────────────────────────────────────

def test_output_structure():
    result = score_risk(_feat())
    assert set(result.keys()) == {"score", "level", "level_range"}


def test_output_example():
    """CLAUDE.md 示例：7 特征 → HIGH"""
    result = score_risk({
        "device_reuse_ratio": 0.82,
        "ip_change_freq": 0.45,
        "tx_freq": 12.0,
        "login_fail_ratio": 0.30,
        "amount_anomaly_score": 0.67,
        "behavior_time_anomaly": 0.15,
        "multi_region_risk": 0.0,
    })
    # 基础分 + 规则5(+19) + 规则6(+15) + 规则8(+15) = +49
    base = (W["device_reuse_ratio"] * 0.82 + W["ip_change_freq"] * 0.45
            + W["tx_freq"] * min(1.0, 12.0 / 100.0)
            + W["login_fail_ratio"] * 0.30 + W["amount_anomaly_score"] * 0.67
            + W["behavior_time_anomaly"] * 0.15 + W["multi_region_risk"] * 0.0) * 100
    expected = min(100.0, base + 49.0)
    assert result["score"] == pytest.approx(expected)
    assert result["level"] == "HIGH"
