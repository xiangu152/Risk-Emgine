"""
特征提取模块 — 从行为日志 CSV 计算反欺诈特征向量

输入: CSV 行为日志
  - 兼容旧格式: user_id, event_type, device_id, ip_address, timestamp, amount
  - 兼容新格式: account_id, type, machine, ip_address, pub_time, money_size,
               behavior_type, behavior_period_time, region, behavior_log

输出: 23 个特征指标的 dict[str, float]

@data: 由调用方通过 csv_path 参数传入
"""
from __future__ import annotations

import re
import hashlib
import pandas as pd
import numpy as np

# ── 设备指纹正则 ──────────────────────────────────────
# 匹配模式: "设备指纹：AP-6B8F2D1E-2025" 或 "设备指纹:ABC123"
_FP_RE = re.compile(r"设备指纹[：:]\s*([A-Za-z0-9\-]+)")

# login_fail 模式: behavior_type 中包含这些关键字的视为登录失败
_LOGIN_FAIL_PATTERNS = ["fail", "error", "wrong", "incorrect", "password", "denied",
                         "unauthorized", "locked", "blocked", "invalid", "suspicious"]


def _extract_device_fingerprints(machine_str: str) -> list[str]:
    """从 machine 字段文本中提取所有设备指纹。

    一个 machine 字段可能包含多台设备描述（如公司手机+个人手机），
    每台设备用「设备指纹：XXX」标记。此函数返回所有匹配的指纹。

    Args:
        machine_str: machine 字段原始文本

    Returns:
        设备指纹列表；若无匹配则对整个文本做 hash 作为回退标识
    """
    if not isinstance(machine_str, str):
        return ["unknown_device"]
    matches = _FP_RE.findall(machine_str)
    if matches:
        return matches
    # 回退: hash 整个 machine 文本
    h = hashlib.md5(machine_str.encode()).hexdigest()[:12]
    return [f"fp_{h}"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：新格式 → 旧格式映射，同时保留 behavior 相关字段。

    返回的 DataFrame 同时包含新旧两套列名，方便后续计算。
    """
    COLUMN_MAP = {
        "account_id":  "user_id",
        "type":        "event_type",
        "machine":     "device_id_raw",
        "pub_time":    "timestamp",
        "money_size":  "amount",
    }
    mapped = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    # 补齐缺失列
    if "device_id_raw" not in mapped.columns:
        mapped["device_id_raw"] = mapped["device_id"]
    if "event_type" not in mapped.columns and "event_type" not in df.columns:
        mapped["event_type"] = mapped.get("type", "unknown")
    if "amount" not in mapped.columns:
        mapped["amount"] = mapped.get("money_size", 0.0)

    # 保留新字段用于新特征
    for col in ["behavior_type", "behavior_period_time", "region", "behavior_log"]:
        if col in df.columns and col not in mapped.columns:
            mapped[col] = df[col]

    return mapped


def _load_and_validate(csv_path: str) -> tuple[pd.DataFrame, bool]:
    """读取并校验 CSV，返回 (DataFrame, has_amount)。

    Raises:
        ValueError: CSV 无法读取、为空、或缺少必填字段时抛出
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        raise ValueError(f"无法读取CSV文件: {csv_path}")

    if df.empty or len(df) == 0:
        raise ValueError(f"CSV文件为空或无数据行: {csv_path}")

    # 统一列名
    df = _normalize_columns(df)

    # 确保必要字段存在
    required_cols = {"user_id", "event_type", "device_id_raw", "ip_address", "timestamp"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV缺少必填字段: {missing_cols}")

    # 检查 amount 字段
    has_amount = "amount" in df.columns

    # 解析时间戳 — 兼容混合格式（有/无时区），统一输出 UTC
    def _parse_ts(s):
        if not isinstance(s, str) or s.strip() == "":
            return pd.NaT
        s = s.strip()
        try:
            ts = pd.Timestamp(s)
        except Exception:
            try:
                ts = pd.Timestamp(s[:19])  # 去掉时区后缀重试
            except Exception:
                return pd.NaT
        # 统一转 UTC
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    df["timestamp"] = df["timestamp"].apply(_parse_ts)
    df = df[df["timestamp"].notna()]

    return df, has_amount


# ═══════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════

def extract_features(csv_path: str) -> dict[str, float]:
    """从行为日志 CSV 提取欺诈特征向量（聚合模式，所有用户取平均）。

    Returns 23 个特征（v1.4.0），向后兼容旧 7 特征输入。
    """
    per_user = extract_features_per_user(csv_path)
    keys = list(next(iter(per_user.values())).keys()) if per_user else []
    return {
        k: float(np.mean([u[k] for u in per_user.values()]))
        for k in keys
    }


def _extract_features_per_user_from_df(
    df: pd.DataFrame, has_amount: bool
) -> dict[str, dict[str, float]]:
    """内部实现：从已加载的 DataFrame 提取 per-user 特征（供 evaluate.py 调用避免重复 I/O）。"""

    total_users = df["user_id"].nunique()
    if total_users == 0:
        raise ValueError("CSV数据中无有效用户")

    all_users = sorted(df["user_id"].unique())

    # ── 1. 构建设备→账户映射（正则提取指纹 + 展开） ──
    fp_to_accounts: dict[str, set] = {}
    user_to_fps: dict[str, set] = {}
    for _, row in df[["user_id", "device_id_raw"]].drop_duplicates().iterrows():
        uid = row["user_id"]
        fps = _extract_device_fingerprints(row["device_id_raw"])
        user_to_fps.setdefault(uid, set()).update(fps)
        for fp in fps:
            fp_to_accounts.setdefault(fp, set()).add(uid)

    # ── 2. 预备统计 ────────────────────────────────────────
    total_ips = df["ip_address"].nunique()
    user_ip_counts = df.groupby("user_id")["ip_address"].nunique()

    tx_df = df[df["event_type"] == "transaction"]
    user_tx_counts = tx_df.groupby("user_id").size()

    # login_fail: 从 behavior_type 挖掘
    has_behavior = "behavior_type" in df.columns
    if has_behavior:
        fail_mask = df["behavior_type"].str.contains(
            "|".join(_LOGIN_FAIL_PATTERNS), case=False, na=False
        )
        login_mask = df["event_type"] == "login"
        login_fail_df = df[login_mask & fail_mask]
        total_login_df = df[login_mask]
    else:
        login_fail_df = df[df["event_type"] == "login_fail"]
        total_login_df = df[df["event_type"].isin(["login", "login_fail"])]

    login_fail_counts = login_fail_df.groupby("user_id").size()
    total_login_counts = total_login_df.groupby("user_id").size()

    # amount anomaly: 全局 MAD
    amount_anomaly_per_user = _calc_amount_anomaly_per_user(df, has_amount, all_users)

    # 全局金额统计（用于 per-user baseline）
    tx_amounts = df[(df["event_type"] == "transaction") & df.get("amount", pd.Series(dtype=float)).notna() & (df["amount"] > 0)]["amount"].astype(float)
    global_median = float(tx_amounts.median()) if len(tx_amounts) > 0 else 0.0
    global_mad = float(np.median(np.abs(tx_amounts - global_median)) * 1.4826) if len(tx_amounts) > 0 else 0.0

    # ── 3. 逐用户计算 ─────────────────────────────────────
    result: dict[str, dict[str, float]] = {}
    for user in all_users:
        fps = user_to_fps.get(user, set())
        max_shared = max((len(fp_to_accounts.get(fp, set())) for fp in fps), default=0)
        device_reuse = max_shared / total_users if total_users > 0 else 0.0
        ip_change = user_ip_counts.get(user, 0) / total_ips if total_ips > 0 else 0.0
        tx_f = float(user_tx_counts.get(user, 0))
        fails = login_fail_counts.get(user, 0)
        total = total_login_counts.get(user, 0)
        login_f = (fails / total) if total > 0 else 0.0
        amt_a = amount_anomaly_per_user.get(user, 0.0)
        bta = _calc_behavior_time_anomaly_per_user(df, user)
        mrr = _calc_multi_region_risk_per_user(df, user)
        vel = _calc_velocity_features_per_user(df, user)
        interv = _calc_interval_features_per_user(df, user)
        amt_base = _calc_amount_baseline_per_user(df, user, global_median, global_mad)
        cyclic = _calc_cyclic_time_features_per_user(df, user)

        # interaction features (v1.3.0)
        inter = {
            "device_ip_risk":   device_reuse * ip_change,
            "night_automation": cyclic["night_tx_ratio"] * interv["tx_burst_ratio"],
            "velocity_amount":  vel["tx_velocity_5min"] * amt_a,
            "burst_max_ratio":  interv["tx_burst_ratio"] * amt_base["amount_user_max_ratio"],
            "velocity_ratio":   vel["tx_velocity_1h"] / max(vel["tx_velocity_5min"], 0.01),
            "amount_exposure":  amt_a * tx_f / max(total_users, 1),
        }

        result[user] = {
            "device_reuse_ratio":    device_reuse,
            "ip_change_freq":        ip_change,
            "tx_freq":               tx_f,
            "login_fail_ratio":      login_f,
            "amount_anomaly_score":  amt_a,
            "behavior_time_anomaly": bta,
            "multi_region_risk":     mrr,
            "tx_velocity_5min":      vel["tx_velocity_5min"],
            "tx_velocity_1h":        vel["tx_velocity_1h"],
            "device_switch_24h":     vel["device_switch_24h"],
            "ip_switch_24h":         vel["ip_switch_24h"],
            "tx_interval_mean_sec":  interv["tx_interval_mean_sec"],
            "tx_burst_ratio":        interv["tx_burst_ratio"],
            "amount_user_deviation": amt_base["amount_user_deviation"],
            "amount_user_max_ratio": amt_base["amount_user_max_ratio"],
            "night_tx_ratio":        cyclic["night_tx_ratio"],
            "odd_hour_tx_ratio":     cyclic["odd_hour_tx_ratio"],
            # v1.3.0 — 交互特征
            "device_ip_risk":        inter["device_ip_risk"],
            "night_automation":      inter["night_automation"],
            "velocity_amount":       inter["velocity_amount"],
            "burst_max_ratio":       inter["burst_max_ratio"],
            "velocity_ratio":        inter["velocity_ratio"],
            "amount_exposure":       inter["amount_exposure"],
        }

    return result


def extract_features_per_user(csv_path: str) -> dict[str, dict[str, float]]:
    """为每个用户独立计算特征向量。"""
    df, has_amount = _load_and_validate(csv_path)
    return _extract_features_per_user_from_df(df, has_amount)


# 各特征计算函数
# ═══════════════════════════════════════════════════════════════

def _calc_device_reuse_ratio(df: pd.DataFrame, total_users: int) -> float:
    """设备复用率：正则提取设备指纹，统计跨账户共享。

    1. 对每行 machine 文本用正则提取所有「设备指纹：XXX」
    2. 构建 指纹→账户集合 的映射
    3. 每个用户的得分 = 其使用设备中「被最多账户共享的设备」的账户数 / total_users
    4. 所有用户取均值
    """
    # 展开: 每行 → 多个 (user_id, fingerprint)
    rows = []
    for _, row in df[["user_id", "device_id_raw"]].iterrows():
        uid = row["user_id"]
        for fp in _extract_device_fingerprints(row["device_id_raw"]):
            rows.append((uid, fp))

    if not rows:
        return 0.0

    fp_df = pd.DataFrame(rows, columns=["user_id", "fingerprint"])

    # 每个指纹关联多少不同用户
    fp_user_count = fp_df.groupby("fingerprint")["user_id"].nunique()

    # 每个用户使用哪些指纹
    user_fps = fp_df.groupby("user_id")["fingerprint"].apply(set)

    ratios = []
    for user in df["user_id"].unique():
        fps = user_fps.get(user, set())
        max_shared = max(
            (fp_user_count.get(fp, 0) for fp in fps),
            default=0,
        )
        ratios.append(max_shared / total_users)

    return float(np.mean(ratios)) if ratios else 0.0


def _calc_ip_change_freq(df: pd.DataFrame) -> float:
    """IP 变更频率：每用户不同 IP 数 / 总 IP 数，取均值。"""
    total_ips = df["ip_address"].nunique()
    if total_ips == 0:
        return 0.0
    user_ip_counts = df.groupby("user_id")["ip_address"].nunique()
    normalized = user_ip_counts / total_ips
    return float(normalized.mean())


def _calc_tx_freq(df: pd.DataFrame) -> float:
    """交易频率：每用户 transaction 事件数，取均值（原始计数）。"""
    tx_df = df[df["event_type"] == "transaction"]
    if tx_df.empty:
        return 0.0
    user_tx_counts = tx_df.groupby("user_id").size()
    all_users = df["user_id"].nunique()
    if all_users == 0:
        return 0.0
    total_tx = user_tx_counts.sum()
    return float(total_tx / all_users)


def _calc_login_fail_ratio(df: pd.DataFrame) -> float:
    """登录失败率：从 behavior_type 挖掘失败模式。

    优先使用 behavior_type 字段匹配失败关键词；
    回退到旧逻辑（event_type == "login_fail"）。
    """
    if "behavior_type" in df.columns:
        # 新格式: type=login 且 behavior_type 含失败关键词
        fail_pattern = "|".join(_LOGIN_FAIL_PATTERNS)
        login_mask = df["event_type"] == "login"
        fail_mask = df["behavior_type"].str.contains(fail_pattern, case=False, na=False)
        login_fail_count = (login_mask & fail_mask).sum()
        login_total = login_mask.sum()
    else:
        # 旧格式: event_type == "login_fail"
        login_fail_count = (df["event_type"] == "login_fail").sum()
        login_total = df["event_type"].isin(["login", "login_fail"]).sum()

    if login_total == 0:
        return 0.0
    return float(login_fail_count / login_total)


def _calc_amount_anomaly_score(df: pd.DataFrame, has_amount: bool) -> float:
    """金额异常分数：MAD 稳健 Z-score。"""
    if not has_amount or "amount" not in df.columns:
        return 0.0

    tx_mask = (df["event_type"] == "transaction") & df["amount"].notna() & (df["amount"] > 0)
    tx_df = df[tx_mask]

    if tx_df.empty or len(tx_df) < 2:
        return 0.0

    amounts = tx_df["amount"].astype(float)
    median_val = amounts.median()
    mad_val = np.median(np.abs(amounts - median_val)) * 1.4826
    if mad_val == 0 or pd.isna(mad_val):
        return 0.0

    robust_z = np.abs((amounts - median_val) / mad_val)
    anomaly_scores = np.minimum(1.0, robust_z / 3.0)
    return float(anomaly_scores.mean())


def _calc_behavior_time_anomaly(df: pd.DataFrame) -> float:
    """行为耗时异常：behavior_period_time < 5s 的行为占比。

    自动化脚本操作极快（<5s），与正常用户操作节奏明显不同。
    """
    if "behavior_period_time" not in df.columns:
        return 0.0

    bpt = df["behavior_period_time"].dropna()
    if len(bpt) == 0:
        return 0.0
    return float((bpt < 5.0).mean())


def _calc_multi_region_risk(df: pd.DataFrame) -> float:
    """多地区风险：存在多地区账户时返回 1.0，否则 0.0。"""
    if "region" not in df.columns:
        return 0.0

    regions_per_account = df.groupby("user_id")["region"].nunique()
    return 1.0 if (regions_per_account > 1).any() else 0.0


def _calc_amount_anomaly_per_user(
    df: pd.DataFrame, has_amount: bool, all_users: list[str]
) -> dict[str, float]:
    """Per-user 金额异常分：全局 MAD，每用户取交易异常分均值。"""
    result = {u: 0.0 for u in all_users}
    if not has_amount or "amount" not in df.columns:
        return result

    tx_mask = (df["event_type"] == "transaction") & df["amount"].notna() & (df["amount"] > 0)
    tx_df = df[tx_mask]
    if tx_df.empty or len(tx_df) < 2:
        return result

    amounts = tx_df["amount"].astype(float)
    median_val = amounts.median()
    mad_val = np.median(np.abs(amounts - median_val)) * 1.4826
    if mad_val == 0 or pd.isna(mad_val):
        return result

    robust_z = np.abs((amounts - median_val) / mad_val)
    anomaly_scores = np.minimum(1.0, robust_z / 3.0)
    tx_df = tx_df.copy()
    tx_df["_anomaly"] = anomaly_scores.values
    user_avg = tx_df.groupby("user_id")["_anomaly"].mean()
    for u in user_avg.index:
        result[u] = float(user_avg[u])
    return result


def _calc_behavior_time_anomaly_per_user(df: pd.DataFrame, user: str) -> float:
    """Per-user 行为耗时异常。"""
    if "behavior_period_time" not in df.columns:
        return 0.0
    bpt = df[df["user_id"] == user]["behavior_period_time"].dropna()
    if len(bpt) == 0:
        return 0.0
    return float((bpt < 5.0).mean())


def _calc_multi_region_risk_per_user(df: pd.DataFrame, user: str) -> float:
    """Per-user 多地区风险。"""
    if "region" not in df.columns:
        return 0.0
    regions = df[df["user_id"] == user]["region"].nunique()
    return 1.0 if regions > 1 else 0.0


# ═══════════════════════════════════════════════════════════════
# 滑动窗口速度特征 (v1.2.0)
# ═══════════════════════════════════════════════════════════════

def _sliding_max_count(
    timestamps: np.ndarray,
    window_ns: int,
    values: np.ndarray | None = None,
) -> float:
    """双指针滑动窗口：计算任意窗口内事件数（或去重值数）的最大值。

    Args:
        timestamps: 已排序的纳秒时间戳数组
        window_ns: 窗口大小（纳秒）
        values: 可选，用于去重计数的值数组

    Returns:
        窗口内最大计数
    """
    n = len(timestamps)
    if n == 0:
        return 0.0
    if values is None:
        # 简单计数
        max_count = 0
        right = 0
        for left in range(n):
            window_end = timestamps[left] + window_ns
            while right < n and timestamps[right] <= window_end:
                right += 1
            max_count = max(max_count, right - left)
            if right == n:
                break
        return float(max_count)
    else:
        # 去重计数
        max_distinct = 0
        right = 0
        seen: dict = {}
        distinct = 0
        for left in range(n):
            window_end = timestamps[left] + window_ns
            while right < n and timestamps[right] <= window_end:
                v = values[right]
                seen[v] = seen.get(v, 0) + 1
                if seen[v] == 1:
                    distinct += 1
                right += 1
            max_distinct = max(max_distinct, distinct)
            # 移出左边界
            v_left = values[left]
            seen[v_left] -= 1
            if seen[v_left] == 0:
                distinct -= 1
        return float(max_distinct)


# ═══════════════════════════════════════════════════════════════
# 交易间隔 + 金额基线特征 (v1.2.0)
# ═══════════════════════════════════════════════════════════════

def _calc_interval_features_per_user(df: pd.DataFrame, user: str) -> dict[str, float]:
    """Per-user 交易间隔特征。

    Returns:
        tx_interval_mean_sec: 平均交易间隔（秒），越低越像自动化脚本
        tx_burst_ratio: 60 秒内连续交易占比，越高越像卡片测试/批量盗刷
    """
    user_df = df[df["user_id"] == user]
    tx_df = user_df[user_df["event_type"] == "transaction"]
    if tx_df.empty or len(tx_df) < 2:
        return {"tx_interval_mean_sec": 0.0, "tx_burst_ratio": 0.0}

    ts = tx_df["timestamp"].dropna().sort_values()
    if len(ts) < 2:
        return {"tx_interval_mean_sec": 0.0, "tx_burst_ratio": 0.0}

    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    intervals_sec = np.diff(ts_ns) / 1_000_000_000  # ns → seconds

    mean_interval = float(np.mean(intervals_sec)) if len(intervals_sec) > 0 else 0.0
    burst_ratio = float((intervals_sec < 60).mean()) if len(intervals_sec) > 0 else 0.0

    return {
        "tx_interval_mean_sec": mean_interval,
        "tx_burst_ratio": burst_ratio,
    }


def _calc_amount_baseline_per_user(
    df: pd.DataFrame, user: str, global_median: float, global_mad: float
) -> dict[str, float]:
    """Per-user 金额基线偏差特征。

    Args:
        global_median: 全量交易金额中位数
        global_mad: 全量 MAD (median absolute deviation × 1.4826)

    Returns:
        amount_user_deviation: 用户均值偏离全局中位数的程度（Z-score）
        amount_user_max_ratio: 用户最大交易 / (用户均值 + 1)，极端值检测
    """
    user_df = df[df["user_id"] == user]
    tx_df = user_df[(user_df["event_type"] == "transaction")
                    & user_df["amount"].notna()
                    & (user_df["amount"] > 0)]
    if tx_df.empty:
        return {"amount_user_deviation": 0.0, "amount_user_max_ratio": 0.0}

    amounts = tx_df["amount"].astype(float)
    user_mean = amounts.mean()
    user_max = amounts.max()

    # 偏离全局中位数的 Z-score
    if global_mad > 0:
        deviation = abs(user_mean - global_median) / global_mad
    else:
        deviation = 0.0

    # 极端单笔：最大交易 / 用户均值
    max_ratio = user_max / (user_mean + 1.0)

    return {
        "amount_user_deviation": float(deviation),
        "amount_user_max_ratio": float(max_ratio),
    }


# ═══════════════════════════════════════════════════════════════
# 时间循环特征 (v1.2.0)
# ═══════════════════════════════════════════════════════════════

# 凌晨 1am-5am 是欺诈高峰，通过 tz_convert("Asia/Shanghai") 动态计算


def _calc_cyclic_time_features_per_user(df: pd.DataFrame, user: str) -> dict[str, float]:
    """Per-user 时间循环特征。

    Returns:
        night_tx_ratio: 凌晨 1am-5am 交易占比，越高越可疑
        odd_hour_tx_ratio: 非工作时间交易占比 (0-6, 22-23 点)
    """
    user_df = df[df["user_id"] == user]
    tx_df = user_df[user_df["event_type"] == "transaction"]
    if tx_df.empty:
        return {"night_tx_ratio": 0.0, "odd_hour_tx_ratio": 0.0}

    ts = tx_df["timestamp"].dropna()
    if len(ts) == 0:
        return {"night_tx_ratio": 0.0, "odd_hour_tx_ratio": 0.0}

    # 转为北京时间的小时 (0-23)
    hours = ts.dt.tz_convert("Asia/Shanghai").dt.hour
    total = len(hours)

    # 凌晨 1am-5am
    night_count = int(((hours >= 1) & (hours < 5)).sum())
    night_ratio = night_count / total if total > 0 else 0.0

    # 非工作时间: 0-6am 或 10pm-12am
    odd_count = int(((hours < 7) | (hours >= 22)).sum())
    odd_ratio = odd_count / total if total > 0 else 0.0

    return {
        "night_tx_ratio": float(night_ratio),
        "odd_hour_tx_ratio": float(odd_ratio),
    }


def _calc_velocity_features_per_user(df: pd.DataFrame, user: str) -> dict[str, float]:
    """Per-user 滑动窗口速度特征。

    Returns 4 keys:
        tx_velocity_5min, tx_velocity_1h, device_switch_24h, ip_switch_24h
    """
    user_df = df[df["user_id"] == user]
    tx_df = user_df[user_df["event_type"] == "transaction"]

    def _max_tx(window_minutes: int) -> float:
        if tx_df.empty:
            return 0.0
        ts = tx_df["timestamp"].dropna().sort_values()
        if len(ts) < 2:
            return float(len(ts))
        # convert to int64 ns via numpy datetime64
        ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        return _sliding_max_count(ts_ns, window_minutes * 60 * 1_000_000_000)

    # 交易速度
    v5 = _max_tx(5)
    v1h = _max_tx(60)

    # 设备切换 — 24h 窗口内最多不同设备
    dev_ts = user_df[["timestamp", "device_id_raw"]].dropna(subset=["timestamp"])
    dev_ts = dev_ts.sort_values("timestamp")
    dev_count = 0.0
    if len(dev_ts) >= 2:
        ts_ns = dev_ts["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        dv = dev_ts["device_id_raw"].astype(str).values
        dev_count = _sliding_max_count(ts_ns, 24 * 3600 * 1_000_000_000, dv)

    # IP 切换 — 24h 窗口内最多不同 IP
    ip_ts = user_df[["timestamp", "ip_address"]].dropna(subset=["timestamp"])
    ip_ts = ip_ts.sort_values("timestamp")
    ip_count = 0.0
    if len(ip_ts) >= 2:
        ts_ns = ip_ts["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        iv = ip_ts["ip_address"].astype(str).values
        ip_count = _sliding_max_count(ts_ns, 24 * 3600 * 1_000_000_000, iv)

    return {
        "tx_velocity_5min":  v5,
        "tx_velocity_1h":    v1h,
        "device_switch_24h": dev_count,
        "ip_switch_24h":     ip_count,
    }
