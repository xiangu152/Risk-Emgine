"""
API 集成测试 — 验证 API_README.md 中的示例代码可正常运行

运行: python -m pytest tests/test_api_integration.py -v
"""
import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from feature_extraction import extract_features, extract_features_per_user
from scoring_model import score_risk, score_risk_hybrid

DATA = "data/20260601_20260630_risk.csv"


def test_extract_features_quickstart():
    """API_README.md 快速开始示例：特征提取 + 打分"""
    features = extract_features(DATA)
    result = score_risk(features)

    # 检查返回结构
    assert isinstance(features, dict)
    assert len(features) >= 17, f"期望 >=17 特征, 实际 {len(features)}"

    assert isinstance(result, dict)
    assert set(result.keys()) == {"score", "level", "level_range"}
    assert 0.0 <= result["score"] <= 100.0
    assert result["level"] in ("LOW", "MEDIUM", "HIGH")
    assert result["level_range"] in ("0~29", "30~70", "71~100")


def test_extract_features_value_ranges():
    """API_README.md 特征表：值域检查"""
    features = extract_features(DATA)

    # 基础特征
    assert 0.0 <= features["device_reuse_ratio"] <= 1.0
    assert 0.0 <= features["ip_change_freq"] <= 1.0
    assert features["tx_freq"] >= 0.0
    assert 0.0 <= features["login_fail_ratio"] <= 1.0
    assert 0.0 <= features["amount_anomaly_score"] <= 1.0
    assert 0.0 <= features["behavior_time_anomaly"] <= 1.0
    assert features["multi_region_risk"] in (0.0, 1.0)

    # 速度特征
    assert features["tx_velocity_5min"] >= 0.0
    assert features["tx_velocity_1h"] >= 0.0
    assert features["device_switch_24h"] >= 0.0
    assert features["ip_switch_24h"] >= 0.0

    # 间隔特征
    assert features["tx_interval_mean_sec"] >= 0.0
    assert 0.0 <= features["tx_burst_ratio"] <= 1.0

    # 金额基线
    assert features["amount_user_deviation"] >= 0.0
    assert features["amount_user_max_ratio"] >= 0.0

    # 时间循环
    assert 0.0 <= features["night_tx_ratio"] <= 1.0
    assert 0.0 <= features["odd_hour_tx_ratio"] <= 1.0


def test_extract_features_per_user():
    """API_README.md per-user 模式示例"""
    per_user = extract_features_per_user(DATA)

    assert isinstance(per_user, dict)
    assert len(per_user) == 100, f"期望 100 账户, 实际 {len(per_user)}"

    # 每个账户有 17 特征
    for uid, features in per_user.items():
        assert isinstance(uid, str)
        assert uid.startswith("ACC_")
        assert len(features) >= 17
        for v in features.values():
            assert isinstance(v, float)
        break  # 只检查第一个


def test_score_risk_level_mapping():
    """API_README.md 风险等级：阈值正确"""
    per_user = extract_features_per_user(DATA)

    for uid, features in per_user.items():
        result = score_risk(features)
        score = result["score"]
        level = result["level"]

        if score <= 29:
            assert level == "LOW", f"{uid}: score={score}, level={level}"
            assert result["level_range"] == "0~29"
        elif score <= 70:
            assert level == "MEDIUM", f"{uid}: score={score}, level={level}"
            assert result["level_range"] == "30~70"
        else:
            assert level == "HIGH", f"{uid}: score={score}, level={level}"
            assert result["level_range"] == "71~100"


def test_batch_scoring():
    """API_README.md 后端批量打分示例"""
    per_user = extract_features_per_user(DATA)

    results = []
    for uid, features in per_user.items():
        r = score_risk(features)
        results.append({
            "account_id": uid,
            "score": r["score"],
            "level": r["level"],
        })

    assert len(results) == 100

    # 统计分布
    levels = [r["level"] for r in results]
    assert "LOW" in levels
    assert "MEDIUM" in levels
    # 高风险至少有几个
    high = [r for r in results if r["level"] == "HIGH"]
    assert len(high) > 0, "应该有至少一个高风险账户"


def test_score_risk_hybrid():
    """API_README.md ML 混合评分示例"""
    features = extract_features(DATA)

    # alpha=1.0 退化为纯规则
    result_pure = score_risk_hybrid(features, alpha=1.0)
    result_rule = score_risk(features)
    assert abs(result_pure["score"] - result_rule["score"]) < 0.01

    # alpha=0.0 纯 ML
    result_ml = score_risk_hybrid(features, alpha=0.0)
    assert "rule_score" in result_ml
    assert "ml_score" in result_ml
    assert 0.0 <= result_ml["rule_score"] <= 100.0
    assert 0.0 <= result_ml["ml_score"] <= 100.0


def test_json_output_example():
    """API_README.md JSON 输出示例格式"""
    features = extract_features(DATA)
    result = score_risk(features)

    # 可序列化为 JSON
    import json
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    assert parsed["score"] == result["score"]
    assert parsed["level"] == result["level"]


def test_error_handling():
    """输入异常时抛出 ValueError"""
    with pytest.raises(ValueError):
        extract_features("nonexistent_file.csv")


def test_empty_csv_raises():
    """空数据文件应报错"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("user_id,event_type,device_id,ip_address,timestamp\n")  # 仅表头
        tmp = f.name
    try:
        with pytest.raises(ValueError):
            extract_features(tmp)
    finally:
        os.unlink(tmp)
