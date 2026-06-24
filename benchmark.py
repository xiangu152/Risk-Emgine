"""
性能 Benchmark — 特征提取 + 打分延迟与吞吐量
"""
import time
import os
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from feature_extraction import extract_features, extract_features_per_user, _load_and_validate, _extract_features_per_user_from_df
from scoring_model import score_risk, score_risk_hybrid

# ── 加载数据 ──────────────────────────────────────
print("=" * 65)
print("特征提取 + 风险打分 Benchmark")
print("=" * 65)

DATA = "data/20260601_20260630_risk.csv"
df_full = pd.read_csv(DATA)
n_rows, n_accounts = len(df_full), df_full["account_id"].nunique()
print(f"数据: {n_rows:,} 行 × {n_accounts} 账户")

# ── 1. 特征提取全量 ──────────────────────────────
print(f"\n{'─' * 65}")
print("1. 特征提取 (全量 CSV)")
print(f"{'─' * 65}")

t0 = time.perf_counter()
feat = extract_features(DATA)
t_feat = time.perf_counter() - t0
print(f"  extract_features()           {t_feat*1000:>8.1f} ms  ({len(feat)} 特征)")

t0 = time.perf_counter()
per_user = extract_features_per_user(DATA)
t_per = time.perf_counter() - t0
print(f"  extract_features_per_user()  {t_per*1000:>8.1f} ms  ({len(per_user)} 账户 × {len(next(iter(per_user.values())))} 特征)")

# 内存内路径 (跳过 CSV I/O)
df, has_amount = _load_and_validate(DATA)
t0 = time.perf_counter()
per_user_mem = _extract_features_per_user_from_df(df, has_amount)
t_mem = time.perf_counter() - t0
print(f"  _from_df() (跳过I/O)         {t_mem*1000:>8.1f} ms  (内存内)")

# ── 2. 打分延迟 ──────────────────────────────────
print(f"\n{'─' * 65}")
print("2. 打分延迟 (单个特征向量)")
print(f"{'─' * 65}")

# 用全量特征
sample = per_user[list(per_user.keys())[0]]

t0 = time.perf_counter()
for _ in range(1000):
    score_risk(sample)
t_rule = (time.perf_counter() - t0) / 1000
print(f"  score_risk() (规则)          {t_rule*1e6:>8.1f} µs/call")

t0 = time.perf_counter()
for _ in range(100):
    score_risk_hybrid(sample, alpha=0.0)
t_ml = (time.perf_counter() - t0) / 100
print(f"  score_risk_hybrid() (ML)     {t_ml*1000:>8.1f} ms/call  (含模型加载)")

# warm-up 后测纯推理
score_risk_hybrid(sample, alpha=0.0)
t0 = time.perf_counter()
for _ in range(100):
    score_risk_hybrid(sample, alpha=0.0)
t_ml_warm = (time.perf_counter() - t0) / 100
print(f"  score_risk_hybrid() (warmup) {t_ml_warm*1000:>8.1f} ms/call  (已加载)")

# ── 3. 全量吞吐量 ────────────────────────────────
print(f"\n{'─' * 65}")
print("3. 全量打分吞吐量")
print(f"{'─' * 65}")

# 规则模型
t0 = time.perf_counter()
for uid, feat in per_user.items():
    score_risk(feat)
t_rule_all = time.perf_counter() - t0
print(f"  score_risk() × {len(per_user)}     {t_rule_all*1000:>8.1f} ms  "
      f"({len(per_user)/t_rule_all:>7.0f} calls/s, {t_rule_all/len(per_user)*1000:.2f} ms/account)")

# ML 模型
t0 = time.perf_counter()
for uid, feat in list(per_user.items())[:20]:
    score_risk_hybrid(feat, alpha=0.0)
t_ml_batch = (time.perf_counter() - t0) / 20
print(f"  score_risk_hybrid() × 1      {t_ml_batch*1000:>8.1f} ms  "
      f"({1/t_ml_batch:>7.0f} calls/s, 预估全量 {t_ml_batch*len(per_user):.1f}s)")

# ── 4. 特征计算细分 ──────────────────────────────
print(f"\n{'─' * 65}")
print("4. 特征计算细分 (全量)")
print(f"{'─' * 65}")

# 拆解各阶段耗时
df2, has_amount2 = _load_and_validate(DATA)
total_users = df2["user_id"].nunique()

# 设备指纹提取
t0 = time.perf_counter()
for _, row in df2[["user_id", "device_id_raw"]].iterrows():
    from feature_extraction import _extract_device_fingerprints
    _extract_device_fingerprints(row["device_id_raw"])
t_dev = time.perf_counter() - t0
print(f"  设备指纹提取                {t_dev*1000:>8.1f} ms  ({len(df2):,} 行)")

# 滑动窗口速度
t0 = time.perf_counter()
from feature_extraction import _calc_velocity_features_per_user
for user in df2["user_id"].unique():
    _calc_velocity_features_per_user(df2, user)
t_vel = time.perf_counter() - t0
print(f"  速度特征计算 (4特征)        {t_vel*1000:>8.1f} ms  ({len(df2['user_id'].unique())} 账户)")

# 交易间隔
t0 = time.perf_counter()
from feature_extraction import _calc_interval_features_per_user
for user in df2["user_id"].unique():
    _calc_interval_features_per_user(df2, user)
t_int = time.perf_counter() - t0
print(f"  间隔特征计算 (2特征)        {t_int*1000:>8.1f} ms")

# 金额基线
t0 = time.perf_counter()
tx_amt = df2[(df2["event_type"]=="transaction") & df2["amount"].notna() & (df2["amount"]>0)]["amount"].astype(float)
gmed = float(tx_amt.median()) if len(tx_amt)>0 else 0.0
gmad = float(np.median(np.abs(tx_amt-gmed))*1.4826) if len(tx_amt)>0 else 0.0
from feature_extraction import _calc_amount_baseline_per_user
for user in df2["user_id"].unique():
    _calc_amount_baseline_per_user(df2, user, gmed, gmad)
t_amt = time.perf_counter() - t0
print(f"  金额基线计算 (2特征)        {t_amt*1000:>8.1f} ms")

# 时间循环
t0 = time.perf_counter()
from feature_extraction import _calc_cyclic_time_features_per_user
for user in df2["user_id"].unique():
    _calc_cyclic_time_features_per_user(df2, user)
t_cyc = time.perf_counter() - t0
print(f"  时间循环计算 (2特征)        {t_cyc*1000:>8.1f} ms")

# ── 5. 模型对比 ──────────────────────────────────
print(f"\n{'─' * 65}")
print("5. 模型推理延迟对比 (warmup后)")
print(f"{'─' * 65}")

import joblib
scaler = joblib.load("models/scaler.pkl")
# 使用23特征的模型（calibrated RF / original ensemble）
model_path = "models/calibrated_rf.pkl"
if not os.path.exists(model_path):
    model_path = "models/ensemble.pkl"
ml_model = joblib.load(model_path)

from scoring_model import _ML_FEATURE_ORDER_FULL as FNAMES
x = np.array([[sample.get(k,0) for k in FNAMES]])
x_s = scaler.transform(x)

# warmup
for _ in range(10):
    ml_model.predict(x_s)
t0 = time.perf_counter()
for _ in range(100):
    ml_model.predict(x_s)
t = (time.perf_counter() - t0) / 100
t0 = time.perf_counter()
for _ in range(100):
    ml_model.predict_proba(x_s)
tp = (time.perf_counter() - t0) / 100
print(f"  CalibratedRF  predict={t*1e6:>6.0f} µs  predict_proba={tp*1e6:>6.0f} µs")

# ── 6. 汇总 ──────────────────────────────────────
print(f"\n{'=' * 65}")
print("Benchmark 汇总")
print(f"{'=' * 65}")
print(f"  全量特征提取 (含I/O):  {t_feat*1000:.0f} ms  ({n_rows/t_feat:.0f} 行/s)")
print(f"  全量特征提取 (无I/O):  {t_mem*1000:.0f} ms  ({n_rows/t_mem:.0f} 行/s)")
print(f"  规则打分延迟:           {t_rule*1e6:.0f} µs/call")
print(f"  ML 打分延迟:            {t_ml_warm*1000:.1f} ms/call")
print(f"  全量规则打分:           {t_rule_all*1000:.0f} ms  ({100/t_rule_all:.0f} accounts/s)")
print(f"  单账户端到端 (规则):    {(t_mem*1000/n_accounts + t_rule*1000):.1f} ms")
print(f"  单账户端到端 (ML):      {(t_mem*1000/n_accounts + t_ml_warm*1000):.1f} ms")
