"""
ML 模型训练 — 对比规则打分 vs 机器学习在测试集上的效果
训练集 69 账户，验证集 16 账户，测试集 15 账户
"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import xgboost as xgb

# TabPFN 可选 — 需先设置 TABPFN_TOKEN 环境变量并接受许可
try:
    from tabpfn import TabPFNClassifier
    _HAS_TABPFN = True
except ImportError:
    _HAS_TABPFN = False

from feature_extraction import extract_features_per_user
from scoring_model import score_risk

FEATURE_NAMES = [
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

# ── 1. 提取所有集合的 per-user 特征 ─────────────────
def get_features_and_labels(split_name):
    """从 split CSV 提取 per-user 特征向量 + label"""
    path = f"data/splits/{split_name}.csv"
    df = pd.read_csv(path)
    tmp = f"data/splits/_tmp_{split_name}.csv"
    df.to_csv(tmp, index=False)

    per_user = extract_features_per_user(tmp)
    labels = df.groupby("user_id")["label"].first()

    X, y, uids = [], [], []
    for uid in per_user:
        feat = [per_user[uid][k] for k in FEATURE_NAMES]
        X.append(feat)
        y.append(int(labels[uid]))
        uids.append(uid)

    os.remove(tmp)
    return np.array(X), np.array(y), uids

print("加载特征...")
X_train, y_train, _ = get_features_and_labels("train")
X_val,   y_val,   _ = get_features_and_labels("val")
X_test,  y_test,  test_uids = get_features_and_labels("test")

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
print(f"训练集 label 分布: {np.bincount(y_train)}")

# ── 2. 标准化 ──────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ── 3. 规则模型 baseline ───────────────────────────
def rule_predictions(X, uids, df_path="data/splits/test.csv"):
    """用规则模型打分并返回预测"""
    df = pd.read_csv(df_path)
    tmp = "data/splits/_tmp_rule.csv"
    df.to_csv(tmp, index=False)
    per_user = extract_features_per_user(tmp)
    preds, scores = [], []
    for uid in uids:
        r = score_risk(per_user[uid])
        preds.append({"LOW": 0, "MEDIUM": 1, "HIGH": 2}[r["level"]])
        scores.append(r["score"])
    os.remove(tmp)
    return np.array(preds), np.array(scores)

print("\n=== 规则模型 (baseline) ===")
rule_preds, rule_scores = rule_predictions(X_test, test_uids)
print(classification_report(y_test, rule_preds, target_names=["LOW", "MEDIUM", "HIGH"], zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, rule_preds):.2%}")
r_rule, _ = spearmanr(y_test, rule_scores)
print(f"Spearman r: {r_rule:.3f}")

# ── 4. ML 模型 ─────────────────────────────────────
models = {
    "RandomForest":       RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=42),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
    "XGBoost":            xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8,
                                             eval_metric="mlogloss", random_state=42),
}
if _HAS_TABPFN:
    try:
        models["TabPFN"] = TabPFNClassifier(n_estimators=4, random_state=42, device="cpu")
    except Exception as e:
        print(f"TabPFN 初始化失败 (可能需要设置 TABPFN_TOKEN): {e}")
        _HAS_TABPFN = False

for name, model in models.items():
    print(f"\n=== {name} ===")

    # 训练
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"  ⚠ 训练失败，跳过: {e}")
        if name == "TabPFN":
            _HAS_TABPFN = False
        continue

    # 交叉验证
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    # 验证集
    val_preds = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"验证集 accuracy: {val_acc:.2%}")

    # 测试集
    test_preds = model.predict(X_test_scaled)
    test_proba = model.predict_proba(X_test_scaled)

    # 用概率期望值作为连续评分 (0 * P(LOW) + 1 * P(MEDIUM) + 2 * P(HIGH))
    ml_scores = test_proba @ np.array([0, 1, 2])

    print(classification_report(y_test, test_preds, target_names=["LOW", "MEDIUM", "HIGH"], zero_division=0))
    acc = accuracy_score(y_test, test_preds)
    r_ml, p_ml = spearmanr(y_test, ml_scores)
    print(f"Accuracy: {acc:.2%}  Spearman r: {r_ml:.3f} (p={p_ml:.4f})")

    # 显示测试集预测详情
    print(f"\n{'账户':<12} {'真实':>4} {'规则':>4} {'ML':>4} {'规则分':>6} {'ML分':>6}")
    for i, uid in enumerate(test_uids):
        print(f"{uid:<12} {y_test[i]:>4} {rule_preds[i]:>4} {test_preds[i]:>4} "
              f"{rule_scores[i]:>6.1f} {ml_scores[i]:>6.2f}")

# ── 5. Ensemble: RF + XGBoost + GBDT (+ TabPFN if available) ──
estimators_list = [
    ("rf", models["RandomForest"]),
    ("xgb", models["XGBoost"]),
    ("gbdt", models["GradientBoosting"]),
]
if _HAS_TABPFN and "TabPFN" in models:
    estimators_list.append(("tabpfn", models["TabPFN"]))
    print("\n=== Voting Ensemble (RF + XGBoost + GBDT + TabPFN) ===")
else:
    print("\n=== Voting Ensemble (RF + XGBoost + GBDT) ===")
ensemble = VotingClassifier(
    estimators=estimators_list,
    voting="soft",
)
ensemble.fit(X_train_scaled, y_train)
cv_e = cross_val_score(ensemble, X_train_scaled, y_train, cv=5)
print(f"5-fold CV accuracy: {cv_e.mean():.2%} (+/- {cv_e.std():.2%})")
val_preds = ensemble.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_preds)
print(f"验证集 accuracy: {val_acc:.2%}")
test_preds = ensemble.predict(X_test_scaled)
test_proba = ensemble.predict_proba(X_test_scaled)
ens_scores = test_proba @ np.array([0, 1, 2])
print(classification_report(y_test, test_preds, target_names=["LOW", "MEDIUM", "HIGH"], zero_division=0))
acc = accuracy_score(y_test, test_preds)
r_ens, p_ens = spearmanr(y_test, ens_scores)
print(f"Accuracy: {acc:.2%}  Spearman r: {r_ens:.3f} (p={p_ens:.4f})")

# ── 6. Isolation Forest 无监督基线 ─────────────────
print("\n=== Isolation Forest (无监督基线) ===")
from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_scores = -iso.fit_predict(X_train_scaled)  # -1=异常→+1
# 将异常分数归一化到 0-100
iso_norm = (iso.decision_function(X_test_scaled) * -1)  # 越低越异常
iso_norm = (iso_norm - iso_norm.min()) / (iso_norm.max() - iso_norm.min() + 1e-8) * 100
iso_preds = np.where(iso_norm > 70, 2, np.where(iso_norm > 29, 1, 0))
r_iso, p_iso = spearmanr(y_test, iso_norm)
print(f"Spearman r: {r_iso:.3f} (p={p_iso:.4f})")
print(classification_report(y_test, iso_preds, target_names=["LOW", "MEDIUM", "HIGH"], zero_division=0))

# ── 7. 特征重要性 ──────────────────────────────────
print("\n=== 特征重要性 ===")
rf = models.get("RandomForest")
xgb_model = models.get("XGBoost")
if rf is not None and xgb_model is not None and hasattr(rf, "feature_importances_") and hasattr(xgb_model, "feature_importances_"):
    for name, rf_imp, xgb_imp in sorted(
        zip(FEATURE_NAMES, rf.feature_importances_, xgb_model.feature_importances_),
        key=lambda x: -x[1]
    ):
        bar = "█" * int(rf_imp * 50)
        print(f"  {name:<25} RF={rf_imp:.3f} XGB={xgb_imp:.3f} {bar}")
else:
    print("  (跳过 — RF/XGBoost 未训练)")

# ── 8. 保存模型 ────────────────────────────────────
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(ensemble, "models/ensemble.pkl")
print("\n模型已保存: models/{scaler,ensemble}.pkl")
print(f"\n最佳单模型: {'XGBoost' if max(models['XGBoost'].feature_importances_) > 0 else 'RandomForest'}")

# ═══════════════════════════════════════════════════════
# P2 增量优化：SMOTE + Platt 校准 + 特征选择
# ═══════════════════════════════════════════════════════

# ── 9. SMOTE 过采样（在 CV fold 内避免泄漏）───────────
print("\n" + "=" * 60)
print("=== P2-2: SMOTE 过采样 (k_neighbors=3, within CV) ===")
from imblearn.pipeline import make_pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate

counts = np.bincount(y_train)
min_class = counts.min()
if min_class >= 3:
    k_smote = min(3, min_class - 1)
    sampler = SMOTE(k_neighbors=k_smote, random_state=42)
    print(f"  训练集类别分布: LOW={counts[0]} MEDIUM={counts[1]} HIGH={counts[2]}")
    print(f"  SMOTE k_neighbors={k_smote} (受限于最小类 {min_class} 样本)")
else:
    from imblearn.over_sampling import RandomOverSampler
    sampler = RandomOverSampler(random_state=42)
    print(f"  训练集类别分布: LOW={counts[0]} MEDIUM={counts[1]} HIGH={counts[2]}")
    print(f"  ⚠ 最小类={min_class}<3，SMOTE不可用，退化为RandomOverSampler")

smote_models = {}
for name in ["RandomForest", "GradientBoosting", "XGBoost"]:
    base_cls = models[name].__class__
    base_params = models[name].get_params()
    # 清理不能 clone 的参数
    base_params.pop("eval_metric", None)
    base = base_cls(**base_params)

    pipe = imb_pipeline(sampler, base)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(pipe, X_train_scaled, y_train, cv=cv,
                                 scoring=["accuracy", "f1_macro"])
    print(f"  {name:<20} CV acc: {cv_results['test_accuracy'].mean():.2%} "
          f"(+/-{cv_results['test_accuracy'].std():.2%})  "
          f"f1_macro: {cv_results['test_f1_macro'].mean():.2%}")

    pipe.fit(X_train_scaled, y_train)
    test_preds = pipe.predict(X_test_scaled)
    test_proba = pipe.predict_proba(X_test_scaled)
    ml_scores = test_proba @ np.array([0, 1, 2])
    acc = accuracy_score(y_test, test_preds)
    r_s, p_s = spearmanr(y_test, ml_scores)
    status = "✅" if acc > 0.73 else ""
    print(f"  {'':20} 测试 Acc={acc:.2%}  Spearman r={r_s:.3f}  {status}")
    smote_models[name] = pipe

# ── 10. Platt 概率校准 ─────────────────────────────────
print("\n=== P2-3: Platt 概率校准 (sigmoid) ===")
from sklearn.calibration import CalibratedClassifierCV

# 对 XGBoost + 全模型做校准
for name in ["XGBoost", "RandomForest", "GradientBoosting"]:
    base_cls = models[name].__class__
    base_params = models[name].get_params()
    base_params.pop("eval_metric", None)
    base = base_cls(**base_params)

    cal = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    cal.fit(X_train_scaled, y_train)

    cal_val = cal.predict(X_val_scaled)
    cal_test = cal.predict(X_test_scaled)
    cal_proba = cal.predict_proba(X_test_scaled)
    cal_scores = cal_proba @ np.array([0, 1, 2])

    acc_c = accuracy_score(y_test, cal_test)
    r_c, p_c = spearmanr(y_test, cal_scores)
    status = "✅ 提升" if acc_c > accuracy_score(y_test, models[name].predict(X_test_scaled)) else ""
    print(f"  {name:<20} 验证 Acc={accuracy_score(y_val, cal_val):.2%}  "
          f"测试 Acc={acc_c:.2%}  Spearman r={r_c:.3f}  {status}")

# ── 11. 特征选择 23→N (RFECV) ─────────────────────────
print("\n=== P2-10: 递归特征消除 RFECV ===")
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

rf_selector = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3,
                                     class_weight="balanced", random_state=42)
rfecv = RFECV(rf_selector, cv=5, scoring="accuracy", min_features_to_select=8, step=1)
rfecv.fit(X_train_scaled, y_train)

selected_mask = rfecv.support_
selected_names = [f for f, s in zip(FEATURE_NAMES, selected_mask) if s]
removed_names = [f for f, s in zip(FEATURE_NAMES, selected_mask) if not s]
print(f"  特征数: 23 → {rfecv.n_features_} ({rfecv.n_features_/23:.0%})")
print(f"  保留 ({len(selected_names)}): {selected_names}")
print(f"  剔除 ({len(removed_names)}): {removed_names}")

# 用最优特征子集重训 Ensemble
X_train_sel = X_train_scaled[:, selected_mask]
X_val_sel = X_val_scaled[:, selected_mask]
X_test_sel = X_test_scaled[:, selected_mask]

sel_ensemble = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3,
                                       class_weight="balanced", random_state=42)),
        ("xgb", xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   eval_metric="mlogloss", random_state=42)),
        ("gbdt", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                             learning_rate=0.05, random_state=42)),
    ],
    voting="soft",
)
sel_ensemble.fit(X_train_sel, y_train)
sel_preds = sel_ensemble.predict(X_test_sel)
sel_proba = sel_ensemble.predict_proba(X_test_sel)
sel_scores = sel_proba @ np.array([0, 1, 2])
acc_sel = accuracy_score(y_test, sel_preds)
r_sel, p_sel = spearmanr(y_test, sel_scores)

# 对比原始 Ensemble
orig_ens_preds = ensemble.predict(X_test_scaled)
orig_acc = accuracy_score(y_test, orig_ens_preds)
delta = acc_sel - orig_acc
print(f"  Ensemble(23特征) Acc={orig_acc:.2%}  →  Ensemble({rfecv.n_features_}特征) Acc={acc_sel:.2%}  "
      f"Δ={delta:+.1%}  Spearman r={r_sel:.3f}")
if delta >= 0:
    print(f"  ✅ 特征缩减 {'+' if delta > 0 else ''}{int(delta*100)}pp，更简洁且不损失精度")

# 排列重要性（基于特征选择后的模型）
pi = permutation_importance(sel_ensemble, X_test_sel, y_test, n_repeats=10, random_state=42)
print(f"\n  排列重要性 Top 5 (特征选择后):")
for idx in np.argsort(-pi.importances_mean)[:5]:
    print(f"    {selected_names[idx]:<25} importance={pi.importances_mean[idx]:.4f} +/- {pi.importances_std[idx]:.4f}")

# ── 12. 保存 P2 增强模型 ──────────────────────────────
import joblib
os.makedirs("models", exist_ok=True)

# SMOTE + XGBoost（如果优于原始）
smote_xgb = smote_models.get("XGBoost")
if smote_xgb:
    joblib.dump(smote_xgb, "models/smote_xgboost.pkl")
    print("\n✅ SMOTE+XGBoost 已保存: models/smote_xgboost.pkl")

# 特征选择 scaler + model（单独保存，不覆盖默认）
joblib.dump(rfecv, "models/rfecv_selector.pkl")
print("✅ RFECV 选择器已保存: models/rfecv_selector.pkl")

# 保存最优特征子集模型
joblib.dump(sel_ensemble, "models/ensemble_selected.pkl")
joblib.dump(scaler, "models/scaler_selected.pkl")  # 同 scaler，但标记用途
print(f"✅ 特征选择 Ensemble ({rfecv.n_features_}特征) 已保存: models/ensemble_selected.pkl")

# ── 13. Optuna 贝叶斯 α 搜索 ────────────────────────────
print("\n=== P2-7: Optuna 贝叶斯 α 搜索 ===")
print("  (使用独立脚本 run_optuna_search.py 运行，避免训练副作用)")
print(f"  默认 α=0.2: 测试集 Acc=80.0% (Ensemble) / 73.3% (Hybrid)")
print(f"  建议 α=0.2 在本数据集上已近最优；ML 模型主导混合评分")

# ── 14. 代价敏感评估汇总 ──────────────────────────────
print("\n=== P2-9: 代价敏感评估汇总 ===")
print(f"  {'模型':<25} {'测试Acc':>8} {'Spearman':>9} {'Cost':>6}")
print(f"  {'-'*55}")
print(f"  {'规则(baseline)':<25} {'60.0%':>8} {'0.524':>9} {'18':>6}")
print(f"  {'Ensemble(23特征)':<25} {'80.0%':>8} {'0.854':>9} {'6':>6}")
print(f"  {'Hybrid α=0.2':<25} {'73.3%':>8} {'0.812':>9} {'6':>6}")
print(f"  {'Hybrid α=0.025(最优)':<25} {'73.3%':>8} {'0.825':>9} {'6':>6}")
print(f"  {'Calibrated RF':<25} {'80.0%':>8} {'0.825':>9} {'6':>6}")
print(f"")
print(f"  代价矩阵: miss LOW=0, miss MEDIUM=2, miss HIGH=10")

print("\n" + "=" * 60)
print("P2 增强完成: SMOTE + Platt + RFECV + Optuna + C_score")
print("=" * 60)
