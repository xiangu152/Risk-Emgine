"""
完整评估脚本 — 训练集/验证集/测试集 全部跑一遍，对比 ground truth

用法: uv run python run_eval.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from scipy.stats import spearmanr
from feature_extraction import extract_features_per_user
from scoring_model import score_risk, score_risk_hybrid

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "splits")
SPLITS = ["train", "val", "test"]
LEVEL_CN = {0: "低风险", 1: "中风险", 2: "高风险"}


def run_split(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    tmp = os.path.join(DATA_DIR, f"_tmp_{name}.csv")
    df.to_csv(tmp, index=False)

    per_user = extract_features_per_user(tmp)
    labels = df.groupby("user_id")["label"].first()
    os.remove(tmp)

    rows = []
    for uid in per_user:
        r_rule = score_risk(per_user[uid])
        r_hyb = score_risk_hybrid(per_user[uid], alpha=0.2)  # 规则+ML 混合
        rows.append({
            "account_id": uid,
            "true_label": int(labels[uid]),
            "rule_score": r_rule["score"],
            "rule_level": r_rule["level"],
            "hyb_score": r_hyb["score"],
            "hyb_level": r_hyb["level"],
        })

    df_r = pd.DataFrame(rows)
    df_r["rule_pred"] = df_r["rule_level"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
    df_r["hyb_pred"] = df_r["hyb_level"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})

    y_true = df_r["true_label"].values
    y_rule = df_r["rule_pred"].values
    y_hyb = df_r["hyb_pred"].values

    acc_r = accuracy_score(y_true, y_rule)
    acc_h = accuracy_score(y_true, y_hyb)
    k_r = cohen_kappa_score(y_true, y_rule)
    k_h = cohen_kappa_score(y_true, y_hyb)
    sr_r, _ = spearmanr(y_true, df_r["rule_score"])
    sr_h, _ = spearmanr(y_true, df_r["hyb_score"])

    # 找出混合模型修复的漏报
    fixed = df_r[(df_r["rule_pred"] != df_r["true_label"]) & (df_r["hyb_pred"] == df_r["true_label"])]
    broken = df_r[(df_r["rule_pred"] == df_r["true_label"]) & (df_r["hyb_pred"] != df_r["true_label"])]

    print(f"\n{'='*70}")
    print(f"  {name.upper()}  ({len(df_r)} 账户)")
    print(f"{'='*70}")
    print(f"  {'':>15} {'纯规则':>10} {'规则+ML混合':>14}")
    print(f"  {'准确率':>15} {acc_r:>9.1%} {acc_h:>14.1%}")
    print(f"  {'Kappa':>15} {k_r:>10.3f} {k_h:>14.3f}")
    print(f"  {'Spearman r':>15} {sr_r:>10.3f} {sr_h:>14.3f}")
    if len(fixed) > 0 or len(broken) > 0:
        print(f"  混合修复: {len(fixed)} 个   混合误伤: {len(broken)} 个")
    print()

    print(f"  {'账户':<14} {'真实':<8} {'规则':>6} {'混合':>6} {'规则分':>7} {'混合分':>7}")
    print(f"  {'-'*60}")
    ok_r = ok_h = 0
    for _, row in df_r.iterrows():
        r_ok = row["true_label"] == row["rule_pred"]
        h_ok = row["true_label"] == row["hyb_pred"]
        ok_r += r_ok
        ok_h += h_ok
        if not r_ok and h_ok:
            tag = "🔧 混合修复"
        elif r_ok and not h_ok:
            tag = "⚠️ 混合误伤"
        elif r_ok:
            tag = "✅"
        else:
            tag = "❌ 都错"
        print(f"  {row['account_id']:<14} {LEVEL_CN[row['true_label']]:<8} "
              f"{row['rule_pred']:>5} {row['hyb_pred']:>5} "
              f"{row['rule_score']:>6.0f} {row['hyb_score']:>6.0f}  {tag}")

    print(f"\n  规则命中: {ok_r}/{len(df_r)}   混合命中: {ok_h}/{len(df_r)}")

    return {
        "name": name, "accounts": len(df_r),
        "rule_acc": round(acc_r, 4), "hyb_acc": round(acc_h, 4),
        "rule_kappa": round(k_r, 4), "hyb_kappa": round(k_h, 4),
        "rule_spearman": round(sr_r, 4), "hyb_spearman": round(sr_h, 4),
    }


# ── 主流程 ──────────────────────────────────────────
print("风控模型评估 — 纯规则 vs 规则+ML混合 对比 ground truth")
results = []
for name in SPLITS:
    if os.path.exists(f"{DATA_DIR}/{name}.csv"):
        results.append(run_split(name))
    else:
        print(f"\n  {name}: 文件不存在，跳过")

# 汇总
print(f"\n{'='*70}")
print("  汇总")
print(f"{'='*70}")
print(f"  {'集合':<8} {'账户':>5} {'规则':>7} {'混合':>7} {'规则Kappa':>10} {'混合Kappa':>10}")
for r in results:
    print(f"  {r['name']:<8} {r['accounts']:>5} {r['rule_acc']:>6.1%} {r['hyb_acc']:>6.1%} "
          f"{r['rule_kappa']:>10.3f} {r['hyb_kappa']:>10.3f}")
