"""成本感知阈值优化 — 用代价矩阵搜索最优LOW/HIGH边界 (已优化为29/70)"""
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from feature_extraction import _extract_features_per_user_from_df, _load_and_validate
from scoring_model import score_risk_hybrid

# 代价矩阵: (预测, 真实) -> 代价
COST = {
    (0,0):0, (1,1):0, (2,2):0,
    (0,1):2, (0,2):10,  # 漏报中/高风险代价高
    (1,0):1, (2,0):3,    # 误报低风险代价低
    (1,2):5, (2,1):1,
}

for name in ["val", "test"]:
    df, ha = _load_and_validate(f"data/splits/{name}.csv")
    pu = _extract_features_per_user_from_df(df, ha)
    labels = df.groupby("user_id")["label"].first()
    uids = sorted(pu.keys())
    y_true = [int(labels[u]) for u in uids]
    scores = np.array([score_risk_hybrid(pu[u], alpha=0.2)["score"] for u in uids])

    if name == "val":
        best_cost, bl, bh = 1e9, 30, 55
        for low in range(10, 50, 5):
            for high in range(40, 95, 5):
                if low >= high: continue
                preds = np.where(scores <= low, 0, np.where(scores > high, 2, 1))
                c = sum(COST.get((int(p), int(t)), 0) for p, t in zip(preds, y_true))
                if c < best_cost:
                    best_cost, bl, bh = c, low, high

    for label, lt, ht in [("当前", 30, 55), ("最优", bl, bh)]:
        preds = np.where(scores <= lt, 0, np.where(scores > ht, 2, 1))
        k = cohen_kappa_score(y_true, preds)
        acc = (preds == y_true).mean()
        fp = int(((np.array(y_true) == 0) & (preds != 0)).sum())
        fn_high = int(((np.array(y_true) == 2) & (preds == 0)).sum())
        cost_val = sum(COST.get((int(p), int(t)), 0) for p, t in zip(preds, y_true))
        print(f"  {name} {label} (LOW<={lt} HIGH>{ht}): Acc={acc:.1%} Kappa={k:.3f} "
              f"cost={cost_val} FP={fp} FN_high={fn_high}")

print(f"\n建议: 调整 LOW<={bl} HIGH>{bh}")
