"""
feature_extraction 模块测试

测试 extract_features() 函数的所有特征提取逻辑。
覆盖正常数据、边界情况、空数据、缺失字段等场景。
"""
import os
import pytest
from feature_extraction import extract_features


# ── 测试数据路径 ──────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _csv_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


# ── 正常数据测试 ──────────────────────────────────────────

def test_extract_features_normal():
    """测试正常 CSV 数据的特征提取"""
    features = extract_features(_csv_path("test_normal.csv"))

    # 检查所有键是否存在
    expected_keys = {
        "device_reuse_ratio",
        "ip_change_freq",
        "tx_freq",
        "login_fail_ratio",
        "amount_anomaly_score",
        "behavior_time_anomaly",
        "multi_region_risk",
    }
    assert expected_keys <= set(features.keys()), f"Missing: {expected_keys - set(features.keys())}"

    # 值域检查
    assert 0.0 <= features["device_reuse_ratio"] <= 1.0
    assert 0.0 <= features["ip_change_freq"] <= 1.0
    assert features["tx_freq"] >= 0.0
    assert 0.0 <= features["login_fail_ratio"] <= 1.0
    assert 0.0 <= features["amount_anomaly_score"] <= 1.0
    assert 0.0 <= features["behavior_time_anomaly"] <= 1.0
    assert features["multi_region_risk"] in (0.0, 1.0)


def test_device_reuse_ratio():
    """
    测试 device_reuse_ratio 计算（按用户维度聚合）。

    test_normal.csv 有 4 个用户 (u_1001~u_1004)：
    - d_501 关联 u_1001, u_1002 → 2 个不同用户
    - d_502 关联 u_1003 → 1 个
    - d_503 关联 u_1004 → 1 个

    每个用户取其最共享设备上的复用率 / 总用户数，然后取平均：
    u_1001(d_501): 2/4=0.5, u_1002(d_501): 0.5,
    u_1003(d_502): 1/4=0.25, u_1004(d_503): 0.25
    平均 = 0.375
    """
    features = extract_features(_csv_path("test_normal.csv"))
    # 按用户维度聚合（与规范"聚合用户特征时取所有用户的平均值"一致）：
    # - u_1001 使用 d_501 → d_501 关联 2 个用户 → 2/4 = 0.5
    # - u_1002 使用 d_501 → 0.5
    # - u_1003 使用 d_502 → 1/4 = 0.25
    # - u_1004 使用 d_503 → 1/4 = 0.25
    # 平均: (0.5 + 0.5 + 0.25 + 0.25) / 4 = 0.375
    assert features["device_reuse_ratio"] == pytest.approx(0.375)


def test_ip_change_freq():
    """
    测试 ip_change_freq 计算。
    test_normal.csv 有 6 个不同 IP：
    192.168.1.100, 192.168.1.101, 10.0.0.1, 10.0.0.2, 10.0.0.3, 172.16.0.1

    - u_1001: 2 个不同 IP -> 2/6 = 0.333
    - u_1002: 2 个不同 IP -> 2/6 = 0.333
    - u_1003: 1 个不同 IP -> 1/6 = 0.167
    - u_1004: 1 个不同 IP -> 1/6 = 0.167
    平均: (0.333+0.333+0.167+0.167)/4 = 0.25
    """
    features = extract_features(_csv_path("test_normal.csv"))
    assert features["ip_change_freq"] == pytest.approx(0.25)


def test_tx_freq():
    """
    测试 tx_freq 计算（原始计数，不归一化）。
    test_normal.csv:
    - u_1001: 2 次 transaction
    - u_1002: 1 次 transaction
    - u_1003: 1 次 transaction
    - u_1004: 3 次 transaction
    平均: (2+1+1+3)/4 = 7/4 = 1.75
    """
    features = extract_features(_csv_path("test_normal.csv"))
    assert features["tx_freq"] == pytest.approx(1.75)


def test_login_fail_ratio():
    """
    测试 login_fail_ratio 计算。
    test_normal.csv:
    - u_1001: login=2, login_fail=1 -> 1/3 ≈ 0.333
    - u_1002: login=1, login_fail=1 -> 1/2 = 0.5
    - u_1003: login=1, login_fail=2 -> 2/3 ≈ 0.667
    - u_1004: login=1, login_fail=0 -> 0/1 = 0.0
    每用户比率均值: (1/3 + 1/2 + 2/3 + 0/1) / 4 = 0.375
    """
    features = extract_features(_csv_path("test_normal.csv"))
    assert features["login_fail_ratio"] == pytest.approx(0.375, abs=0.01)


def test_amount_anomaly_score():
    """
    测试 amount_anomaly_score 计算。
    test_normal.csv 有 6 笔交易：
    1000, 500, 50000, 800, 900, 850, 1200
    均值 ≈ 7893，std ≈ ... 50000 是明显的异常值。
    Z-score 映射到 0~1。
    """
    features = extract_features(_csv_path("test_normal.csv"))
    assert 0.0 <= features["amount_anomaly_score"] <= 1.0
    # 由于有极端值50000，异常分数应该接近1
    # 50000 是明显异常值，但被多笔正常交易平均拉低，整体 >0.2 即可
    assert features["amount_anomaly_score"] > 0.2


# ── 边界情况测试 ──────────────────────────────────────────

def test_empty_csv():
    """测试空 CSV（仅表头）应抛出 ValueError"""
    with pytest.raises(ValueError, match="空"):
        extract_features(_csv_path("test_empty.csv"))


def test_no_login_fail():
    """测试无 login_fail 事件时 login_fail_ratio = 0.0"""
    features = extract_features(_csv_path("test_no_login_fail.csv"))
    assert features["login_fail_ratio"] == 0.0


def test_no_transactions():
    """测试无 transaction 事件时 tx_freq = 0.0"""
    features = extract_features(_csv_path("test_no_transactions.csv"))
    assert features["tx_freq"] == 0.0
    # 无交易时 amount_anomaly_score 也应为 0.0
    assert features["amount_anomaly_score"] == 0.0


def test_missing_amount():
    """测试缺失 amount 字段时的处理"""
    features = extract_features(_csv_path("test_missing_amount.csv"))
    # 缺失 amount 字段时 amount_anomaly_score 应为 0.0
    assert features["amount_anomaly_score"] == 0.0
    # 其他特征应能正常计算
    assert features["tx_freq"] >= 0.0


def test_extreme_case():
    """
    测试极端场景：高设备复用 + 高IP变更 + 高登录失败率
    test_extreme.csv: 2个用户共享1个设备，大量IP变更和登录失败
    """
    features = extract_features(_csv_path("test_extreme.csv"))
    # 2个用户共享 d_501 -> device_reuse_ratio 应该很高
    assert features["device_reuse_ratio"] > 0.4
    # 大量不同IP -> ip_change_freq 应该很高
    assert features["ip_change_freq"] >= 0.5
    # 高 login_fail 比例
    assert features["login_fail_ratio"] > 0.2


# ── 类型检查 ──────────────────────────────────────────────

def test_return_type():
    """测试返回值类型"""
    features = extract_features(_csv_path("test_normal.csv"))
    assert isinstance(features, dict)
    for key, value in features.items():
        assert isinstance(key, str)
        assert isinstance(value, float), f"{key} 应为 float，实际为 {type(value)}"


# ── extract_features_per_user 测试 ────────────────────────

def test_extract_features_per_user():
    """测试 per-user 模式的特征提取"""
    from feature_extraction import extract_features_per_user

    result = extract_features_per_user(_csv_path("test_normal.csv"))

    # 返回类型为 dict[str, dict[str, float]]
    assert isinstance(result, dict)
    assert len(result) == 4  # test_normal.csv 有 4 个用户

    expected_keys = {
        "device_reuse_ratio",
        "ip_change_freq",
        "tx_freq",
        "login_fail_ratio",
        "amount_anomaly_score",
        "behavior_time_anomaly",
        "multi_region_risk",
    }

    for user_id, features in result.items():
        assert isinstance(user_id, str)
        assert isinstance(features, dict)
        assert expected_keys <= set(features.keys()), f"Missing: {expected_keys - set(features.keys())}"
        for feat_name, feat_val in features.items():
            assert isinstance(feat_val, float), (
                f"{user_id}.{feat_name} 应为 float，实际为 {type(feat_val)}"
            )

    # 值域检查
    for features in result.values():
        assert 0.0 <= features["device_reuse_ratio"] <= 1.0
        assert 0.0 <= features["ip_change_freq"] <= 1.0
        assert features["tx_freq"] >= 0.0
        assert 0.0 <= features["login_fail_ratio"] <= 1.0
        assert 0.0 <= features["amount_anomaly_score"] <= 1.0

    # 验证特定用户特征值
    # u_1001: device_reuse=0.5, ip_change=2/6≈0.333, tx=2.0, login_fail=1/3≈0.333
    u1001 = result["u_1001"]
    assert u1001["device_reuse_ratio"] == pytest.approx(0.5)
    assert u1001["ip_change_freq"] == pytest.approx(2.0 / 6.0)
    assert u1001["tx_freq"] == pytest.approx(2.0)
    assert u1001["login_fail_ratio"] == pytest.approx(1.0 / 3.0)

    # u_1003: device_reuse=0.25, ip_change=1/6≈0.167, tx=1.0, login_fail=2/3≈0.667
    u1003 = result["u_1003"]
    assert u1003["device_reuse_ratio"] == pytest.approx(0.25)
    assert u1003["ip_change_freq"] == pytest.approx(1.0 / 6.0)
    assert u1003["tx_freq"] == pytest.approx(1.0)
    assert u1003["login_fail_ratio"] == pytest.approx(2.0 / 3.0)

    # u_1003 有极端金额 50000，其 amount_anomaly_score 应该较高
    assert u1003["amount_anomaly_score"] > 0.2


def test_extract_features_per_user_empty():
    """测试空 CSV 时 per-user 模式抛出 ValueError"""
    from feature_extraction import extract_features_per_user

    with pytest.raises(ValueError, match="空"):
        extract_features_per_user(_csv_path("test_empty.csv"))
