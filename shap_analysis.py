"""SHAP 分析 — 验证 7 特征贡献度，科学淘汰/增强特征"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端的 SVG 输出
import matplotlib.pyplot as plt
import shap
import joblib
import os
from feature_extraction import extract_features_per_user

FEATURE_NAMES = [
    "device_reuse_ratio", "ip_change_freq", "tx_freq",
    "login_fail_ratio", "amount_anomaly_score",
    "behavior_time_anomaly", "multi_region_risk",
]
FEATURE_LABELS = [
    "设备复用率", "IP变更频率", "交易频率", "登录失败率",
    "金额异常", "行为耗时异常", "多地区风险",
]

# ── 1. 加载模型和数据 ──────────────────────────────
print("加载模型...")
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/calibrated_rf.pkl")

print("提取特征...")
def load_set(name):
    df = pd.read_csv(f"data/splits/{name}.csv")
    tmp = f"data/splits/_tmp_shap_{name}.csv"
    df.to_csv(tmp, index=False)
    per_user = extract_features_per_user(tmp)
    labels = df.groupby("user_id")["label"].first()
    X, y, uids = [], [], []
    for uid in per_user:
        X.append([per_user[uid][k] for k in FEATURE_NAMES])
        y.append(int(labels[uid]))
        uids.append(uid)
    os.remove(tmp)
    return np.array(X), np.array(y), uids

X_train, y_train, _ = load_set("train")
X_val,   y_val,   _ = load_set("val")
X_test,  y_test,  test_uids = load_set("test")
X_all = np.vstack([X_train, X_val, X_test])
y_all = np.hstack([y_train, y_val, y_test])
X_all_scaled = scaler.transform(X_all)

# ── 2. SHAP TreeExplainer ─────────────────────────
print("计算 SHAP 值...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_all_scaled)  # [n_samples, n_features, n_classes]

# 取 label=2 (高风险) 的 SHAP，因为欺诈检测中高风险类最重要
# 也可以用所有类的 mean |SHAP|
shap_class2 = shap_values[:, :, 2]  # shape: [100, 7]
shap_mean_abs = np.abs(shap_values).mean(axis=(0, 2))  # 跨样本跨类别均值

# 也看 label=1 和 label=0 的 SHAP 贡献
shap_class0 = shap_values[:, :, 0]
shap_class1 = shap_values[:, :, 1]

# ── 3. 特征重要性排序 ─────────────────────────────
print("\n" + "=" * 70)
print("SHAP 特征贡献度分析 (全量 100 账户)")
print("=" * 70)

# 综合 SHAP (所有样本所有类别的平均绝对 SHAP)
print(f"\n{'特征':<20} {'RF importance':>13} {'SHAP |mean|':>12} {'高风险SHAP':>11} {'结论'}")
print("-" * 75)
rf_imp = model.feature_importances_
for i in np.argsort(-shap_mean_abs):
    status = "✅ 有效" if shap_mean_abs[i] > 0.05 else "⚠️ 弱信号" if shap_mean_abs[i] > 0.01 else "❌ 无信号"
    print(f"  {FEATURE_LABELS[i]:<16} {rf_imp[i]:>12.3f} {shap_mean_abs[i]:>11.3f} {np.abs(shap_class2[:, i]).mean():>10.3f}  {status}")

# ── 4. 高风险样本 SHAP 详解 ──────────────────────
print("\n" + "=" * 70)
print("高风险样本 (label=2) 的 SHAP 驱动特征")
print("=" * 70)
high_risk_idx = np.where(y_all == 2)[0]
print(f"共 {len(high_risk_idx)} 个 label=2 样本\n")

# 构建全量 uid 列表
all_uids = []
for name in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits/{name}.csv")
    for uid in sorted(df["user_id"].unique()):
        all_uids.append(uid)

for idx in high_risk_idx:
    uid = all_uids[idx] if idx < len(all_uids) else f"idx_{idx}"
    top3 = np.argsort(-np.abs(shap_class2[idx]))[:3]
    print(f"  {uid}:")
    for rank, fi in enumerate(top3):
        direction = "↑ 推高风险" if shap_class2[idx, fi] > 0 else "↓ 降低风险"
        print(f"    #{rank+1} {FEATURE_LABELS[fi]}: SHAP={shap_class2[idx, fi]:+.3f} {direction}")
    print()

# ── 5. 每个特征的 SHAP 分布 ──────────────────────
print("=" * 70)
print("特征 SHAP 方向性分析 (正值=推高风险, 负值=降低风险)")
print("=" * 70)
for i, name in enumerate(FEATURE_LABELS):
    pos_ratio = (shap_class2[:, i] > 0).mean()
    mean_pos = shap_class2[shap_class2[:, i] > 0, i].mean() if pos_ratio > 0 else 0
    mean_neg = shap_class2[shap_class2[:, i] < 0, i].mean() if pos_ratio < 1 else 0
    print(f"  {name:<14} | 推高:{pos_ratio:.0%} (avg SHAP={mean_pos:+.3f}) | 降低:{1-pos_ratio:.0%} (avg SHAP={mean_neg:+.3f})")

# ── 6. 可视化 (SVG) ──────────────────────────────
os.makedirs("data/model", exist_ok=True)
print("\n生成 SHAP 可视化...")

# 6a. 特征重要性 summary bar
plt.figure(figsize=(10, 5))
shap.summary_plot(shap_class2, X_all_scaled, feature_names=FEATURE_NAMES,
                  plot_type="bar", show=False, max_display=7)
plt.title("SHAP Feature Importance (label=HIGH)")
plt.tight_layout()
plt.savefig("data/model/shap_importance_bar.svg", dpi=120, bbox_inches="tight")
plt.close()
print("  → data/model/shap_importance_bar.svg")

# 6b. Beeswarm summary plot
plt.figure(figsize=(10, 5))
shap.summary_plot(shap_class2, X_all_scaled, feature_names=FEATURE_NAMES,
                  show=False, max_display=7)
plt.title("SHAP Beeswarm (label=HIGH)")
plt.tight_layout()
plt.savefig("data/model/shap_beeswarm.svg", dpi=120, bbox_inches="tight")
plt.close()
print("  → data/model/shap_beeswarm.svg")

# 6c. 热力图 — shap 0.52+ 需要 Explanation 对象
try:
    shap.plots.heatmap(shap.Explanation(shap_class2, data=X_all_scaled,
                        feature_names=FEATURE_NAMES),
                        max_display=7, show=False)
    plt.title("SHAP Heatmap (label=HIGH)")
    plt.tight_layout()
    plt.savefig("data/model/shap_heatmap.svg", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → data/model/shap_heatmap.svg")
except Exception as e:
    print(f"  ⚠ heatmap 跳过: {e}")

# ── 7. 具体 feature dependence ────────────────────
# 只画有信号的 4 个特征
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
active_features = [4, 5, 2, 1]  # amount_anomaly, behavior_time, tx_freq, ip_change
for i, ax_idx in enumerate(active_features):
    ax = axes[i // 2][i % 2]
    try:
        shap.dependence_plot(ax_idx, shap_class2, X_all_scaled,
                             feature_names=FEATURE_NAMES,
                             interaction_index=None, ax=ax, show=False)
    except Exception:
        # fallback: manual scatter
        ax.scatter(X_all[:, ax_idx], shap_class2[:, ax_idx], alpha=0.6, s=20)
        ax.set_xlabel(FEATURE_NAMES[ax_idx])
        ax.set_ylabel("SHAP value")
    ax.set_title(f"{FEATURE_NAMES[ax_idx]} — Impact on HIGH risk")
plt.tight_layout()
plt.savefig("data/model/shap_dependence.svg", dpi=120, bbox_inches="tight")
plt.close()
print("  → data/model/shap_dependence.svg")

# ── 8. 建议 ───────────────────────────────────────
print("\n" + "=" * 70)
print("📋 特征决策建议")
print("=" * 70)

recommendations = [
    ("amount_anomaly_score", rf_imp[4], shap_mean_abs[4],
     "保留 ✅ — 贡献度最高，MAD Z-score 设计合理", "维持"),
    ("behavior_time_anomaly", rf_imp[5], shap_mean_abs[5],
     "保留 ✅ — 第二重要，<5s 阈值有效", "可尝试阈值扫参优化"),
    ("tx_freq", rf_imp[2], shap_mean_abs[2],
     "保留 ✅ — 第三重要，可与滑动窗口合并增强", "增加 5min/1h 窗口变体"),
    ("ip_change_freq", rf_imp[1], shap_mean_abs[1],
     "保留 ✅ — 有弱信号", "增加 IP 关联账户数（图特征）"),
    ("device_reuse_ratio", rf_imp[0], shap_mean_abs[0],
     "⚠️ 需重新设计 — 当前数据中信号极弱", "改用图特征：连通分量大小 + 度中心性"),
    ("login_fail_ratio", rf_imp[3], shap_mean_abs[3],
     "⚠️ 需重新设计 — 数据中 behavior_type 失败事件太少", "扩大失败关键词匹配 + 增加序列特征"),
    ("multi_region_risk", rf_imp[6], shap_mean_abs[6],
     "⚠️ 保留但已知局限 — 当前数据恒为 0", "不删，等数据变化后激活；可降权"),
]

for feat_name, rf, shap_v, rec, action in recommendations:
    print(f"\n  {feat_name}")
    print(f"    RF importance={rf:.3f}  SHAP|mean|={shap_v:.3f}")
    print(f"    诊断: {rec}")
    print(f"    行动: {action}")

print("\n完成。所有图表已保存到 data/model/")
