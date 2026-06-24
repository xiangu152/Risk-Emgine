"""
阈值校准 — 基于验证集 PR 曲线 + 成本感知优化 LOW/MEDIUM/HIGH 边界

当前固定阈值: ≤29 LOW, ≤70 MEDIUM, >70 HIGH
目标: 在验证集上找到最优分界点
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (precision_recall_curve, f1_score, fbeta_score,
    classification_report, confusion_matrix)
from feature_extraction import extract_features_per_user
from scoring_model import score_risk
import os

# ── 加载验证集评分 ────────────────────────────────
val = pd.read_csv("data/splits/val.csv")
tmp = "data/splits/_tmp_cal.csv"
val.to_csv(tmp, index=False)
per_user = extract_features_per_user(tmp)
labels = val.groupby("user_id")["label"].first()
os.remove(tmp)

scores, y_true = [], []
for uid in per_user:
    r = score_risk(per_user[uid])
    scores.append(r["score"])
    y_true.append(int(labels[uid]))

scores = np.array(scores)
y_true = np.array(y_true)

print("=" * 60)
print("验证集阈值校准 (16 账户)")
print("=" * 60)
print(f"分数范围: {scores.min():.0f} ~ {scores.max():.0f}")
print(f"分数均值: {scores.mean():.1f}, 中位数: {np.median(scores):.1f}")
print(f"label 分布: 0={int((y_true==0).sum())} 1={int((y_true==1).sum())} 2={int((y_true==2).sum())}")

# ── 三分类阈值扫参 ────────────────────────────────
print("\n--- 三分类阈值扫参 ---")
print(f"{'LOW≤':>8} {'HIGH>':>8} {'Acc':>6} {'Kappa':>6} {'正常误报':>8} {'异常漏报':>8}")
best_acc, best_kappa = 0, 0
best_low, best_high = 30, 70
results = []
for low in range(15, 50, 5):
    for high in range(50, 90, 5):
        if low >= high:
            continue
        preds = np.where(scores <= low, 0, np.where(scores > high, 2, 1))
        acc = (preds == y_true).mean()
        # Cohen's Kappa
        from sklearn.metrics import cohen_kappa_score
        k = cohen_kappa_score(y_true, preds)

        # 正常误报: true=0 but pred!=0
        fp_normal = int(((y_true == 0) & (preds != 0)).sum())
        # 异常漏报: true>0 but pred==0
        fn_anomaly = int(((y_true > 0) & (preds == 0)).sum())
        results.append((low, high, acc, k, fp_normal, fn_anomaly))
        if k > best_kappa:
            best_kappa = k
            best_low, best_high = low, high
            best_acc = acc

# 排序展示
results.sort(key=lambda x: (-x[3], -x[2]))  # by kappa, then accuracy
for low, high, acc, k, fp, fn in results[:12]:
    marker = " ← 最优" if low == best_low and high == best_high else ""
    print(f"  ≤{low:>3}  >{high:>3}  {acc:.1%}  {k:.3f}     {fp:>3}/{(y_true==0).sum()}     {fn:>3}/{(y_true>0).sum()}{marker}")

# ── 当前 vs 最优 ──────────────────────────────────
print(f"\n--- 对比 ---")
print(f"当前: LOW≤29 MEDIUM≤70 HIGH>70")
print(f"最优: LOW≤{best_low} MEDIUM≤{best_high} HIGH>{best_high}")

for name, low_t, high_t in [("当前", 29, 70), ("最优", best_low, best_high)]:
    preds = np.where(scores <= low_t, 0, np.where(scores > high_t, 2, 1))
    print(f"\n  {name} (LOW≤{low_t}, HIGH>{high_t}):")
    print(f"  Accuracy: {(preds==y_true).mean():.1%}")
    print(f"  Kappa: {cohen_kappa_score(y_true, preds):.3f}")
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_true, preds, labels=[0,1,2])
    print(f"    真\\预  LOW MED HIGH")
    for i, label in enumerate(["真-LOW","真-MED","真-HIGH"]):
        print(f"    {label} {cm[i][0]:>4} {cm[i][1]:>4} {cm[i][2]:>5}")

# ── 成本感知：FP vs FN 成本比 ─────────────────────
print(f"\n--- 成本感知阈值 ---")
# 欺诈领域: FN(漏报) cost >> FP(误报) cost
# 设 FP cost = 1, FN cost = C (C越大越不能容忍漏报)
for C in [1, 2, 5, 10]:
    best_cost_low, best_cost_high = 30, 70
    best_cost = 1e9
    for low in range(15, 55, 5):
        for high in range(45, 90, 5):
            if low >= high:
                continue
            preds = np.where(scores <= low, 0, np.where(scores > high, 2, 1))
            fp = int(((y_true == 0) & (preds != 0)).sum())
            fn = int(((y_true > 0) & (preds == 0)).sum())
            cost = fp * 1 + fn * C
            if cost < best_cost:
                best_cost = cost
                best_cost_low, best_cost_high = low, high
    print(f"  FP:FN = 1:{C} → 最优: LOW≤{best_cost_low} HIGH>{best_cost_high} (cost={best_cost})")

# ── 训练集 + 测试集交叉验证 ──────────────────────
print(f"\n--- 训练集 & 测试集验证 ---")
for name, path in [("Train", "data/splits/train.csv"), ("Test", "data/splits/test.csv")]:
    df = pd.read_csv(path)
    tmp2 = f"data/splits/_tmp_cal2.csv"
    df.to_csv(tmp2, index=False)
    pu = extract_features_per_user(tmp2)
    lb = df.groupby("user_id")["label"].first()
    sc, yt = [], []
    for u in pu:
        r = score_risk(pu[u])
        sc.append(r["score"])
        yt.append(int(lb[u]))
    os.remove(tmp2)
    sc, yt = np.array(sc), np.array(yt)
    for label, lt, ht in [("当前", 30, 70), ("最优", best_low, best_high)]:
        pr = np.where(sc <= lt, 0, np.where(sc > ht, 2, 1))
        acc = (pr == yt).mean()
        k = cohen_kappa_score(yt, pr)
        print(f"  {name} {label}: Acc={acc:.1%} Kappa={k:.3f}")

print(f"\n建议: 调整 LOW≤{best_low} MEDIUM≤{best_high} HIGH>{best_high}")
