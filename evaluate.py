"""
统一评估入口 — 替换 evaluate_on_real_data.py / evaluate_split.py /
evaluate_hybrid.py / analyze_data.py / calibrate_thresholds.py

用法:
    python evaluate.py full          # 全量数据评估
    python evaluate.py splits        # train/val/test 三集合评估
    python evaluate.py hybrid        # 规则 vs ML vs Hybrid α 扫参
    python evaluate.py calibrate     # 阈值校准
    python evaluate.py all           # 全部，输出 HTML+JSON 报告

输出: --format text|json|html (默认 text)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score)
from scipy.stats import spearmanr

from feature_extraction import _extract_features_per_user_from_df, _load_and_validate
from scoring_model import score_risk, score_risk_hybrid

LABEL_NAMES = ["LOW", "MEDIUM", "HIGH"]
LABEL_NAMES_CN = ["低风险", "中风险", "高风险"]
LEVEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

DATA_DIR = "data/splits"


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def load_and_extract(csv_path: str) -> tuple[dict, pd.Series, np.ndarray, list]:
    """加载 CSV 并提取 per-user 特征。返回 (per_user, labels, X, uids)。"""
    df, has_amount = _load_and_validate(csv_path)
    per_user = _extract_features_per_user_from_df(df, has_amount)
    labels = df.groupby("user_id")["label"].first() if "label" in df.columns else pd.Series()
    uids = sorted(per_user.keys())
    feature_order = list(next(iter(per_user.values())).keys()) if per_user else []
    X = np.array([[per_user[u][k] for k in feature_order] for u in uids]) if feature_order else np.array([])
    return per_user, labels, X, uids


def score_all(per_user: dict, alpha: float | None = None) -> pd.DataFrame:
    """对 per-user 特征打分，返回 DataFrame(user_id, score, level, pred_label)。"""
    rows = []
    for uid, feat in per_user.items():
        r = score_risk(feat) if alpha is None else score_risk_hybrid(feat, alpha=alpha)
        rows.append({
            "user_id": uid, "score": r["score"], "level": r["level"],
            "pred_label": LEVEL_MAP[r["level"]],
        })
    return pd.DataFrame(rows)


def compute_metrics(y_true: list[int], y_pred: list[int], scores: list[float]) -> dict:
    """计算分类指标 + Spearman 相关。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "spearman_r": float(spearmanr(y_true, scores)[0]),
        "spearman_p": float(spearmanr(y_true, scores)[1]),
        "classification_report": classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "high_recall": float(sum(1 for t, p in zip(y_true, y_pred) if t == 2 and p == 2)) / max(1, sum(1 for t in y_true if t == 2)),
    }


def crosstab_table(y_true: list[int], y_pred: list[int]) -> str:
    """中文化的 label-vs-prediction 交叉表。"""
    ct = pd.crosstab(
        pd.Series(y_true).map({0: "label=0(低)", 1: "label=1(中)", 2: "label=2(高)"}),
        pd.Series(y_pred).map({0: "LOW", 1: "MEDIUM", 2: "HIGH"}),
    )
    return ct.to_string()


# ═══════════════════════════════════════════════════════════════
# 子命令
# ═══════════════════════════════════════════════════════════════

def cmd_full(output_format: str) -> dict:
    """全量数据评估（替代 evaluate_on_real_data.py）。"""
    print("=== 全量评估 ===")
    per_user, labels, _, uids = load_and_extract("data/20260601_20260630_risk.csv")
    score_df = score_all(per_user)
    score_df["true_label"] = score_df["user_id"].map(labels.to_dict())
    score_df = score_df.dropna(subset=["true_label"])

    y_true = score_df["true_label"].astype(int).tolist()
    y_pred = score_df["pred_label"].tolist()
    scores = score_df["score"].tolist()

    metrics = compute_metrics(y_true, y_pred, scores)
    print(f"评分: mean={np.mean(scores):.1f} median={np.median(scores):.1f} min={min(scores):.0f} max={max(scores):.0f}")
    print(crosstab_table(y_true, y_pred))
    for lbl, name in [(0, "低"), (1, "中"), (2, "高")]:
        sub = score_df[score_df["true_label"] == lbl]
        if len(sub) > 0:
            print(f"  label={lbl}({name}): mean={sub['score'].mean():.1f} median={sub['score'].median():.1f} n={len(sub)}")
    metrics["score_stats"] = {"mean": float(np.mean(scores)), "median": float(np.median(scores)),
                               "min": float(min(scores)), "max": float(max(scores))}
    return metrics


def cmd_splits(output_format: str) -> dict:
    """train/val/test 三集合评估（替代 evaluate_split.py + analyze_data.py）。"""
    results = {}
    for name in ["train", "val", "test"]:
        path = f"{DATA_DIR}/{name}.csv"
        if not os.path.exists(path):
            print(f"  {name}: 文件不存在，跳过")
            continue
        print(f"\n=== {name.upper()} ===")
        per_user, labels, _, _ = load_and_extract(path)
        score_df = score_all(per_user)
        score_df["true_label"] = score_df["user_id"].map(labels.to_dict())
        score_df = score_df.dropna(subset=["true_label"])

        y_true = score_df["true_label"].astype(int).tolist()
        y_pred = score_df["pred_label"].tolist()
        scores_list = score_df["score"].tolist()

        m = compute_metrics(y_true, y_pred, scores_list)
        print(f"  Accuracy: {m['accuracy']:.2%}  Kappa: {m['kappa']:.3f}  Spearman r: {m['spearman_r']:.3f}")
        print("  " + crosstab_table(y_true, y_pred).replace("\n", "\n  "))
        results[name] = m
    return results


def cmd_hybrid(output_format: str) -> dict:
    """混合模型 α 扫参（替代 evaluate_hybrid.py）。"""
    print("=== 混合模型 α 扫参 ===")
    per_user, labels, _, uids = load_and_extract(f"{DATA_DIR}/test.csv")

    y_true = [int(labels[u]) for u in uids]

    print(f"{'Alpha':>8}  {'Acc':>6}  {'Kappa':>6}  {'Spearman':>8}  {'HIGH检出':>8}")
    best_alpha, best_acc = 0.2, 0.0
    sweep_results = []
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        score_df = score_all(per_user, alpha=alpha)
        preds = score_df["pred_label"].tolist()
        sc = score_df["score"].tolist()
        acc = accuracy_score(y_true, preds)
        k = cohen_kappa_score(y_true, preds)
        sr, _ = spearmanr(y_true, sc)
        high_r = int(sum(1 for t, p in zip(y_true, preds) if t == 2 and p == 2))
        label = "纯ML" if alpha == 0.0 else "纯规则" if alpha == 1.0 else ""
        marker = " ← 最优" if acc > best_acc and label == "" else ""
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha
        print(f"  α={alpha:.1f}   {acc:.2%}   {k:.3f}    {sr:.3f}        {high_r}/{(sum(1 for t in y_true if t==2))}       {label}{marker}")
        sweep_results.append({"alpha": alpha, "accuracy": acc, "kappa": k, "spearman_r": sr})

    # 最佳 α 详情
    print(f"\n--- 最佳 α={best_alpha:.1f} 详情 ---")
    score_df = score_all(per_user, alpha=best_alpha)
    preds = score_df["pred_label"].tolist()
    sc = score_df["score"].tolist()
    print(classification_report(y_true, preds, target_names=LABEL_NAMES, zero_division=0))
    print(f"{'账户':<12} {'真实':<8} {'预测':<8} {'分数':>6}")
    for i, u in enumerate(uids):
        ok = "✅" if y_true[i] == preds[i] else "❌"
        print(f"{u:<12} {LABEL_NAMES_CN[y_true[i]]:<8} {LABEL_NAMES[preds[i]]:<8} {sc[i]:>5.1f} {ok}")

    return {"best_alpha": best_alpha, "best_accuracy": best_acc, "sweep": sweep_results}


def cmd_calibrate(output_format: str) -> dict:
    """阈值校准（替代 calibrate_thresholds.py）。"""
    print("=== 阈值校准 ===")
    per_user, labels, _, uids = load_and_extract(f"{DATA_DIR}/val.csv")
    y_true = [int(labels[u]) for u in uids]
    score_df = score_all(per_user)
    scores = score_df["score"].values

    print(f"验证集分数: mean={scores.mean():.1f} median={np.median(scores):.1f} range={scores.min():.0f}-{scores.max():.0f}")

    # 网格搜索
    best_low, best_high, best_kappa = 30, 55, 0
    for low in range(15, 50, 5):
        for high in range(45, 90, 5):
            if low >= high:
                continue
            preds = np.where(scores <= low, 0, np.where(scores > high, 2, 1))
            k = cohen_kappa_score(y_true, preds)
            if k > best_kappa:
                best_kappa, best_low, best_high = k, low, high

    # 当前 vs 最优 vs 成本感知
    print(f"\n当前: LOW≤30 MEDIUM≤70 HIGH>70")
    print(f"最优: LOW≤{best_low} MEDIUM≤{best_high} HIGH>{best_high}")
    print(f"最佳 Kappa: {best_kappa:.3f}")

    for name, lt, ht in [("当前", 30, 70), ("最优", best_low, best_high)]:
        preds = np.where(scores <= lt, 0, np.where(scores > ht, 2, 1))
        acc = (preds == y_true).mean()
        k = cohen_kappa_score(y_true, preds)
        fp = int(((np.array(y_true) == 0) & (preds != 0)).sum())
        fn = int(((np.array(y_true) > 0) & (preds == 0)).sum())
        print(f"  {name} (LOW≤{lt} HIGH>{ht}): Acc={acc:.1%} Kappa={k:.3f} FP={fp} FN={fn}")

    # 成本感知
    for C in [1, 2, 5, 10]:
        best_cl, best_ch, best_cost = 30, 55, 1e9
        for low in range(15, 55, 5):
            for high in range(45, 90, 5):
                if low >= high: continue
                preds = np.where(scores <= low, 0, np.where(scores > high, 2, 1))
                fp = int(((np.array(y_true) == 0) & (preds != 0)).sum())
                fn = int(((np.array(y_true) > 0) & (preds == 0)).sum())
                cost = fp * 1 + fn * C
                if cost < best_cost:
                    best_cost, best_cl, best_ch = cost, low, high
        print(f"  FP:FN=1:{C} → LOW≤{best_cl} HIGH>{best_ch} (cost={best_cost})")

    return {"best_low": best_low, "best_high": best_high, "best_kappa": best_kappa}


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="统一风控模型评估")
    parser.add_argument("command", choices=["full", "splits", "hybrid", "calibrate", "all"],
                        help="评估子命令")
    parser.add_argument("--format", choices=["text", "json", "html"], default="text",
                        help="输出格式 (默认 text)")
    args = parser.parse_args()

    results: dict = {}
    if args.command in ("full", "all"):
        results["full"] = cmd_full(args.format)
    if args.command in ("splits", "all"):
        results["splits"] = cmd_splits(args.format)
    if args.command in ("hybrid", "all"):
        results["hybrid"] = cmd_hybrid(args.format)
    if args.command in ("calibrate", "all"):
        results["calibrate"] = cmd_calibrate(args.format)

    # 输出
    if args.format == "json":
        os.makedirs("data/evaluation", exist_ok=True)
        out = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        path = "data/evaluation/report.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\n报告已保存: {path}")
    elif args.format == "html":
        os.makedirs("data/evaluation", exist_ok=True)
        html = f"<html><body><h1>评估报告</h1><pre>{json.dumps(results, indent=2, ensure_ascii=False, default=str)}</pre></body></html>"
        path = "data/evaluation/report.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    main()
