"""风控引擎可视化面板 — FastAPI 后端

启动方式:
  cd dashboard
  uv run uvicorn server:app --reload --port 8000

API 端点:
  POST /api/upload          上传 CSV 并处理
  GET  /api/overview        全局统计数据
  GET  /api/users           所有用户评分列表
  GET  /api/users/{id}      单用户详情
  GET  /api/features/weights    特征权重
  GET  /api/features/distribution  特征分布统计
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── 确保能 import 项目根目录的模块 ────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from feature_extraction import extract_features_per_user, _load_and_validate
from scoring_model import score_risk, DEFAULT_WEIGHTS, _check_rules

# 尝试导入 ML 混合评分，失败则回退到纯规则
_ML_AVAILABLE = False
try:
    from scoring_model import score_risk_hybrid
    # 快速测试模型能否加载
    import numpy as _np
    import joblib as _joblib
    _model_dir = os.path.join(_PROJECT_ROOT, "models")
    _test_path = os.path.join(_model_dir, "calibrated_rf.pkl")
    if os.path.exists(_test_path):
        _joblib.load(_test_path)
        _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False

app = FastAPI(title="风控引擎可视化面板", version="1.0.0")

# ── 内存缓存 ───────────────────────────────────────────────
_cache: dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════
# API 端点
# ═══════════════════════════════════════════════════════════════

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """上传 CSV 文件，提取特征并批量打分。"""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "请上传 CSV 文件")

    # 保存到临时文件
    suffix = Path(file.filename).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir()) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. 提取特征
        features_per_user = extract_features_per_user(tmp_path)

        # 2. 加载原始 CSV 用于行为时间线和 label
        df, _ = _load_and_validate(tmp_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # 3. 为每个用户打分
        users_data = []
        for uid, feats in features_per_user.items():
            rule_result = score_risk(feats)
            _, _, fired_rules = _check_rules(feats)

            # ML 混合评分（如果可用）
            if _ML_AVAILABLE:
                try:
                    hybrid_result = score_risk_hybrid(feats, alpha=0.2)
                    ml_score = hybrid_result["ml_score"]
                    hybrid_score = hybrid_result["score"]
                    hybrid_level = hybrid_result["level"]
                except Exception:
                    ml_score = rule_result["score"]
                    hybrid_score = rule_result["score"]
                    hybrid_level = rule_result["level"]
            else:
                ml_score = rule_result["score"]
                hybrid_score = rule_result["score"]
                hybrid_level = rule_result["level"]

            # 获取用户 label
            user_df = df[df["user_id"] == uid]
            label = int(user_df["label"].iloc[0]) if "label" in user_df.columns and len(user_df) > 0 else -1
            user_name = str(user_df["account_name"].iloc[0]) if "account_name" in user_df.columns and len(user_df) > 0 else uid

            # 行为时间线（最近 50 条）
            timeline = []
            if len(user_df) > 0:
                recent = user_df.sort_values("timestamp").tail(50)
                for _, row in recent.iterrows():
                    timeline.append({
                        "time": str(row["timestamp"]),
                        "type": str(row.get("event_type", row.get("type", ""))),
                        "behavior_type": str(row.get("behavior_type", "")),
                        "amount": float(row.get("amount", row.get("money_size", 0)) or 0),
                        "behavior_log": str(row.get("behavior_log", ""))[:200],
                    })

            users_data.append({
                "account_id": uid,
                "account_name": user_name,
                "label": label,
                "features": feats,
                "rule_score": rule_result["score"],
                "rule_level": rule_result["level"],
                "ml_score": ml_score,
                "hybrid_score": hybrid_score,
                "hybrid_level": hybrid_level,
                "fired_rules": fired_rules,
                "fired_count": len(fired_rules),
                "timeline": timeline,
                "tx_count": int(user_df[user_df.get("event_type", user_df.get("type", "")) == "transaction"].shape[0]) if "event_type" in user_df.columns or "type" in user_df.columns else 0,
                "region": str(user_df["region"].iloc[0]) if "region" in user_df.columns and len(user_df) > 0 else "",
            })

        # 缓存结果
        _cache["users"] = users_data
        _cache["filename"] = file.filename
        _cache["total_records"] = len(df)

        return {
            "success": True,
            "filename": file.filename,
            "total_records": len(df),
            "total_accounts": len(users_data),
        }

    except Exception as e:
        raise HTTPException(500, f"处理失败: {e}")
    finally:
        os.unlink(tmp_path)


@app.get("/api/overview")
async def get_overview():
    """全局统计数据。"""
    users = _cache.get("users", [])
    if not users:
        return {"empty": True}

    scores = [u["hybrid_score"] for u in users]
    levels = [u["hybrid_level"] for u in users]

    level_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for lv in levels:
        level_counts[lv] = level_counts.get(lv, 0) + 1

    return {
        "total_accounts": len(users),
        "avg_score": round(float(np.mean(scores)), 1),
        "max_score": round(float(np.max(scores)), 1),
        "min_score": round(float(np.min(scores)), 1),
        "level_distribution": level_counts,
        "score_distribution": {
            "0-29": sum(1 for s in scores if s <= 29),
            "30-50": sum(1 for s in scores if 30 <= s <= 50),
            "51-70": sum(1 for s in scores if 51 <= s <= 70),
            "71-100": sum(1 for s in scores if s > 70),
        },
        "filename": _cache.get("filename", ""),
        "total_records": _cache.get("total_records", 0),
    }


@app.get("/api/users")
async def get_users():
    """所有用户评分列表。"""
    users = _cache.get("users", [])
    if not users:
        return []

    return [
        {
            "account_id": u["account_id"],
            "account_name": u["account_name"],
            "label": u["label"],
            "features": {k: round(v, 2) for k, v in u["features"].items()},
            "rule_score": round(u["rule_score"], 1),
            "ml_score": round(u["ml_score"], 1),
            "hybrid_score": round(u["hybrid_score"], 1),
            "hybrid_level": u["hybrid_level"],
            "fired_count": u["fired_count"],
            "tx_count": u["tx_count"],
            "region": u["region"],
        }
        for u in users
    ]


@app.get("/api/users/{account_id}")
async def get_user_detail(account_id: str):
    """单用户详情：特征 + 规则 + 评分 + 时间线。"""
    users = _cache.get("users", [])
    user = next((u for u in users if u["account_id"] == account_id), None)
    if not user:
        raise HTTPException(404, f"用户 {account_id} 不存在")
    return {
        **user,
        "features": {k: round(v, 2) for k, v in user["features"].items()},
        "rule_score": round(user["rule_score"], 1),
        "ml_score": round(user["ml_score"], 1),
        "hybrid_score": round(user["hybrid_score"], 1),
    }


@app.get("/api/features/weights")
async def get_feature_weights():
    """23 维特征权重。"""
    return DEFAULT_WEIGHTS


@app.get("/api/features/distribution")
async def get_feature_distribution():
    """所有用户的特征分布统计（按风险等级分组）。"""
    users = _cache.get("users", [])
    if not users:
        return {}

    feature_names = list(DEFAULT_WEIGHTS.keys())
    result = {}

    for feat in feature_names:
        groups = {"LOW": [], "MEDIUM": [], "HIGH": []}
        for u in users:
            val = u["features"].get(feat, 0.0)
            groups[u["hybrid_level"]].append(val)

        result[feat] = {
            level: {
                "mean": round(float(np.mean(vals)), 2) if vals else 0,
                "min": round(float(np.min(vals)), 2) if vals else 0,
                "max": round(float(np.max(vals)), 2) if vals else 0,
                "median": round(float(np.median(vals)), 2) if vals else 0,
            }
            for level, vals in groups.items()
        }

    return result


# ── 静态文件 & 首页 ────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(_static_dir / "index.html"))
