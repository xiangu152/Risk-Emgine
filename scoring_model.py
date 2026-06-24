"""
风险打分模型 — 加权求和 + 16 条规则增强 + 可选 ML 混合评分

接收 extract_features() 的返回值（17个特征）作为输入，
综合各特征权重计算基础分，叠加 16 条规则增强后输出最终评分和风险等级。

新增: score_risk_hybrid() — 规则模型 + RandomForest 加权融合
"""
from __future__ import annotations

import os
import numpy as np

# ── ML 模型路径 ────────────────────────────────────────
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_SCALER_PATH = os.path.join(_MODEL_DIR, "scaler.pkl")

# 模型注册表（优先级从高到低）: (模型路径, 标准化器路径, 特征列表, 是否需列选择)
# is_selected=True 表示: 先用 full scaler 标准化 23 特征，再按 feature_order 选列
_MODEL_REGISTRY: list[tuple[str, str, list[str] | None, bool]] = [
    # 优先级从高到低: Calibrated RF → 16特征Ensemble → 23特征Ensemble
    ("calibrated_rf.pkl", "scaler.pkl", None, False),
    ("ensemble_selected.pkl", "scaler.pkl", [
        "ip_change_freq", "tx_freq", "amount_anomaly_score",
        "behavior_time_anomaly", "tx_velocity_5min", "tx_velocity_1h",
        "ip_switch_24h", "tx_interval_mean_sec", "tx_burst_ratio",
        "amount_user_deviation", "amount_user_max_ratio",
        "night_tx_ratio", "odd_hour_tx_ratio",
        "device_ip_risk", "velocity_amount", "amount_exposure",
    ], True),
    ("ensemble.pkl", "scaler.pkl", None, False),
]

# 全程 23 特征（向后兼容 + calibrated_rf / ensemble 使用）
_ML_FEATURE_ORDER_FULL = [
    "device_reuse_ratio", "ip_change_freq", "tx_freq",
    "login_fail_ratio", "amount_anomaly_score",
    "behavior_time_anomaly", "multi_region_risk",
    "tx_velocity_5min", "tx_velocity_1h",
    "device_switch_24h", "ip_switch_24h",
    "tx_interval_mean_sec", "tx_burst_ratio",
    "amount_user_deviation", "amount_user_max_ratio",
    "night_tx_ratio", "odd_hour_tx_ratio",
    "device_ip_risk", "night_automation", "velocity_amount",
    "burst_max_ratio", "velocity_ratio", "amount_exposure",
]

# 向后兼容别名
_ML_FEATURE_ORDER = _ML_FEATURE_ORDER_FULL

# 懒加载缓存
_ml_scaler = None
_ml_model = None
_ml_feature_order = None  # 当前加载模型对应的特征顺序
_ml_is_selected = False   # True → 需先标准化全特征再选列


def _get_best_model() -> tuple[str, str, list[str], bool]:
    """返回最佳可用模型的 (模型路径, 标准化器路径, 特征列表, is_selected)。"""
    for fname, scaler_fname, features, is_selected in _MODEL_REGISTRY:
        path = os.path.join(_MODEL_DIR, fname)
        scaler_path = os.path.join(_MODEL_DIR, scaler_fname)
        if os.path.exists(path) and os.path.exists(scaler_path):
            feat = features if features is not None else _ML_FEATURE_ORDER_FULL
            return path, scaler_path, feat, is_selected
    raise FileNotFoundError(
        f"ML 模型未找到，请运行 train_ml_model.py 训练。"
    )


def _load_ml_model():
    """懒加载 ML 模型 + 标准化器（首次调用时加载，自动选择最佳可用模型）。

    Returns:
        (scaler, model, feature_order, is_selected)
        is_selected=True → 需先标准化 23 特征再按 feature_order 选列
    """
    global _ml_scaler, _ml_model, _ml_feature_order
    if _ml_model is None:
        import joblib
        best_path, scaler_path, feature_order, is_selected = _get_best_model()
        _ml_scaler = joblib.load(scaler_path)
        _ml_model = joblib.load(best_path)
        _ml_feature_order = feature_order
        global _ml_is_selected
        _ml_is_selected = is_selected
    return _ml_scaler, _ml_model, _ml_feature_order, _ml_is_selected


# ── 默认权重（可调，总和≈1.0）v1.4.0: 23 特征 ─────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "device_reuse_ratio":      0.10,
    "ip_change_freq":          0.08,
    "tx_freq":                 0.08,
    "login_fail_ratio":        0.08,
    "amount_anomaly_score":    0.10,
    "behavior_time_anomaly":   0.07,
    "multi_region_risk":       0.05,
    "tx_velocity_5min":        0.08,
    "tx_velocity_1h":          0.08,
    "device_switch_24h":       0.04,
    "ip_switch_24h":           0.05,
    "tx_interval_mean_sec":    0.04,
    "tx_burst_ratio":          0.04,
    "amount_user_deviation":   0.02,
    "amount_user_max_ratio":   0.02,
    "night_tx_ratio":          0.02,
    "odd_hour_tx_ratio":       0.02,
    # v1.3.0 — 交互特征
    "device_ip_risk":          0.02,
    "night_automation":        0.02,
    "velocity_amount":         0.03,
    "burst_max_ratio":         0.02,
    "velocity_ratio":          0.02,
    "amount_exposure":         0.02,
}


def _check_rules(features: dict[str, float]) -> tuple[float, int, list[str]]:
    """检查 16 条规则，返回 (总加分, 命中数, 命中规则名列表)。"""
    boost = 0.0
    fired: list[str] = []

    def _hit(name: str, condition: bool, bonus: float) -> None:
        nonlocal boost
        if condition:
            boost += bonus
            fired.append(name)

    _hit("R1_设备农场+高频", features.get("device_reuse_ratio", 0.0) > 0.9 and features.get("tx_freq", 0.0) > 50, 10.0)
    _hit("R2_IP撞库", features.get("ip_change_freq", 0.0) > 0.8 and features.get("login_fail_ratio", 0.0) > 0.5, 10.0)
    _hit("R3_极端金额", features.get("amount_anomaly_score", 0.0) > 0.7, 15.0)
    _hit("R4_暴力破解", features.get("login_fail_ratio", 0.0) > 0.7, 20.0)
    _hit("R5_设备共享", features.get("device_reuse_ratio", 0.0) > 0.2, 19.0)
    _hit("R6_账户盗用", features.get("ip_change_freq", 0.0) > 0.1 and features.get("amount_anomaly_score", 0.0) > 0.3, 15.0)
    _hit("R7_设备农场撞库", features.get("device_reuse_ratio", 0.0) > 0.15 and features.get("login_fail_ratio", 0.0) > 0.3, 15.0)
    _hit("R8_洗钱", features.get("tx_freq", 0.0) > 10 and features.get("amount_anomaly_score", 0.0) > 0.3, 15.0)
    _hit("R9_自动化脚本", features.get("behavior_time_anomaly", 0.0) > 0.3 and features.get("tx_freq", 0.0) > 20, 15.0)
    _hit("R10_多地区", features.get("multi_region_risk", 0.0) > 0, 12.0)
    _hit("R11_交易爆发", features.get("tx_velocity_5min", 0.0) > 5, 12.0)
    _hit("R12_账户接管变现", features.get("tx_velocity_1h", 0.0) > 15 and (features.get("device_switch_24h", 0.0) > 3 or features.get("ip_switch_24h", 0.0) > 5), 15.0)
    _hit("R13_自动化间隔", features.get("tx_interval_mean_sec", 999.0) < 30.0 and features.get("tx_freq", 0.0) > 5, 10.0)
    _hit("R14_超大额", features.get("amount_user_max_ratio", 0.0) > 10.0, 10.0)
    _hit("R15_批量操作", features.get("tx_burst_ratio", 0.0) > 0.5 and features.get("tx_freq", 0.0) > 10, 10.0)
    _hit("R16_凌晨异常", features.get("night_tx_ratio", 0.0) > 0.3, 8.0)

    return boost, len(fired), fired


def score_risk(
    features: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float | str]:
    """加权求和 + 规则增强，计算综合风险评分。

    Args:
        features: extract_features() 返回的特征 dict（23 维）
        weights: 可选自定义权重 dict，键名与 features 一致。
                 默认使用 DEFAULT_WEIGHTS（23 维，总和≈1.0）。

    Returns:
        {"score": float, "level": str, "level_range": str}
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # ── 加权求和 ──────────────────────────────────────────
    # tx_freq 需要归一化：min(1.0, tx_freq / 100)
    normalized_tx = min(1.0, features.get("tx_freq", 0.0) / 100.0)

    # velocity 特征的归一化
    v5   = min(1.0, features.get("tx_velocity_5min", 0.0) / 10.0)
    v1h  = min(1.0, features.get("tx_velocity_1h", 0.0) / 30.0)
    ds   = min(1.0, features.get("device_switch_24h", 0.0) / 5.0)
    isw  = min(1.0, features.get("ip_switch_24h", 0.0) / 10.0)
    # interval: 间隔→归一化 (反向，越低越危险)。
    # 当 tx_freq=0 时 interval=0 无意义，应贡献 0
    iv_raw = features.get("tx_interval_mean_sec", 3600.0)
    iv_n = 0.0 if iv_raw <= 0 else 1.0 - min(1.0, iv_raw / 3600.0)
    br   = features.get("tx_burst_ratio", 0.0)  # already 0~1
    # amount baseline
    aud  = min(1.0, features.get("amount_user_deviation", 0.0) / 5.0)
    amr  = min(1.0, features.get("amount_user_max_ratio", 0.0) / 10.0)
    # cyclic time: already 0~1 ratios
    nt   = features.get("night_tx_ratio", 0.0)
    ot   = features.get("odd_hour_tx_ratio", 0.0)

    weighted_sum = (
        weights.get("device_reuse_ratio", 0.0)     * features.get("device_reuse_ratio", 0.0)
        + weights.get("ip_change_freq", 0.0)       * features.get("ip_change_freq", 0.0)
        + weights.get("tx_freq", 0.0)              * normalized_tx
        + weights.get("login_fail_ratio", 0.0)     * features.get("login_fail_ratio", 0.0)
        + weights.get("amount_anomaly_score", 0.0) * features.get("amount_anomaly_score", 0.0)
        + weights.get("behavior_time_anomaly", 0.0)* features.get("behavior_time_anomaly", 0.0)
        + weights.get("multi_region_risk", 0.0)    * features.get("multi_region_risk", 0.0)
        + weights.get("tx_velocity_5min", 0.0)     * v5
        + weights.get("tx_velocity_1h", 0.0)       * v1h
        + weights.get("device_switch_24h", 0.0)    * ds
        + weights.get("ip_switch_24h", 0.0)        * isw
        + weights.get("tx_interval_mean_sec", 0.0) * iv_n
        + weights.get("tx_burst_ratio", 0.0)       * br
        + weights.get("amount_user_deviation", 0.0)* aud
        + weights.get("amount_user_max_ratio", 0.0)* amr
        + weights.get("night_tx_ratio", 0.0)       * nt
        + weights.get("odd_hour_tx_ratio", 0.0)    * ot
        + weights.get("device_ip_risk", 0.0)        * features.get("device_ip_risk", 0.0)
        + weights.get("night_automation", 0.0)       * features.get("night_automation", 0.0)
        + weights.get("velocity_amount", 0.0)        * features.get("velocity_amount", 0.0)
        + weights.get("burst_max_ratio", 0.0)        * features.get("burst_max_ratio", 0.0)
        + weights.get("velocity_ratio", 0.0)         * features.get("velocity_ratio", 0.0)
        + weights.get("amount_exposure", 0.0)        * features.get("amount_exposure", 0.0)
    )

    # 转为百分制
    score = weighted_sum * 100.0

    # ── 规则增强（16 条，v1.3.0） ──────────────────────────
    boost, rule_count, fired_rules = _check_rules(features)
    score += boost

    # ── 钳制到 0~100 ──────────────────────────────────────
    score = max(0.0, min(100.0, score))

    # ── 风险等级映射 (v1.4 Optuna多目标优化) ────────────────
    # 在验证集上 Kappa +0.417 (0.355→0.771), Cost -6 (8→2)
    if score <= 29.0:
        level = "LOW"
        level_range = "0~29"
    elif score <= 70.0:
        level = "MEDIUM"
        level_range = "30~70"
    else:
        level = "HIGH"
        level_range = "71~100"

    return {
        "score": score,
        "level": level,
        "level_range": level_range,
    }


def _compute_dynamic_alpha(features: dict[str, float], ml_proba: np.ndarray) -> float:
    """根据规则命中数和 ML 置信度动态调整 alpha。

    启发式规则:
    - 多规则命中 (>=3) → 信规则，alpha += 0.15
    - ML 高置信度 (max proba > 0.9) → 信 ML，alpha -= 0.10
    - 极端特征 (amount_anomaly > 0.7) → alpha += 0.05
    - 交易爆发 (tx_velocity_5min > 5) → alpha += 0.05

    Returns alpha ∈ [0.05, 0.8].
    """
    alpha = 0.2  # base

    _, rule_count, _ = _check_rules(features)

    if rule_count >= 3:
        alpha += 0.15
    elif rule_count >= 1:
        alpha += 0.05

    if ml_proba.max() > 0.9:
        alpha -= 0.10

    if features.get("amount_anomaly_score", 0.0) > 0.7:
        alpha += 0.05
    if features.get("tx_velocity_5min", 0.0) > 5:
        alpha += 0.05
    if features.get("tx_burst_ratio", 0.0) > 0.5:
        alpha += 0.03
    if features.get("night_tx_ratio", 0.0) > 0.3:
        alpha += 0.03

    return max(0.05, min(0.8, alpha))


def score_risk_hybrid(
    features: dict[str, float],
    weights: dict[str, float] | None = None,
    alpha: float | str = 0.2,
) -> dict[str, float | str]:
    """规则模型 + ML 混合评分。

    最终分 = α × 规则分 + (1-α) × ML概率分

    Args:
        features: extract_features() 返回的特征 dict
        weights: 可选自定义权重，传给 score_risk()
        alpha: 规则模型权重 (0~1)，默认 0.2（20%规则 + 80%ML）。
               alpha="auto" 按规则命中数/ML置信度/特征极端值动态调整。
               alpha=1.0 退化为纯规则模型，alpha=0.0 退化为纯 ML。

    Returns:
        {"score", "level", "level_range", "rule_score", "ml_score",
         "alpha_used" (if auto)}
    """
    # 1. 规则分
    rule_result = score_risk(features, weights)
    rule_score = rule_result["score"]

    # 2. ML 概率分
    scaler, model, feat_order, is_selected = _load_ml_model()

    if is_selected:
        # 先构建 23 特征 → 标准化 → 选择列 → 预测
        X_full = np.array([[features.get(k, 0.0) for k in _ML_FEATURE_ORDER_FULL]])
        X_scaled_full = scaler.transform(X_full)
        # 找到 feat_order 在 _ML_FEATURE_ORDER_FULL 中的索引
        col_indices = [_ML_FEATURE_ORDER_FULL.index(k) for k in feat_order]
        X_scaled = X_scaled_full[:, col_indices]
    else:
        X = np.array([[features.get(k, 0.0) for k in feat_order]])
        X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0]  # [P(LOW), P(MEDIUM), P(HIGH)]

    # 概率映射到 0-100: P(MEDIUM)*50 + P(HIGH)*100
    ml_score = float(proba[1] * 50.0 + proba[2] * 100.0)

    # 3. 动态 alpha
    alpha_used = alpha
    if alpha == "auto":
        alpha_used = _compute_dynamic_alpha(features, proba)

    # 4. 加权融合
    score = alpha_used * rule_score + (1.0 - alpha_used) * ml_score
    score = max(0.0, min(100.0, score))

    # 5. 等级映射 (v1.4 Optuna多目标优化)
    if score <= 29.0:
        level, level_range = "LOW", "0~29"
    elif score <= 70.0:
        level, level_range = "MEDIUM", "30~70"
    else:
        level, level_range = "HIGH", "71~100"

    result = {
        "score": score,
        "level": level,
        "level_range": level_range,
        "rule_score": rule_score,
        "ml_score": ml_score,
    }
    if alpha == "auto":
        result["alpha_used"] = alpha_used
    return result
