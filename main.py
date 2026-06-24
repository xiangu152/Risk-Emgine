"""反欺诈规则挖掘系统 — 主入口

用法:
  uv run python main.py [csv_path]           聚合模式（默认）
  uv run python main.py --per-user [csv_path] 每用户独立评分
  uv run python main.py --per-user -u u_1006  查看指定用户
"""
import sys
from feature_extraction import extract_features, extract_features_per_user
from scoring_model import score_risk


def main() -> None:
    # ── 解析命令行参数 ────────────────────────────────────────
    args = sys.argv[1:]

    per_user_mode = "--per-user" in args
    target_user: str | None = None

    if per_user_mode:
        args.remove("--per-user")
        if "-u" in args:
            u_idx = args.index("-u")
            if u_idx + 1 < len(args):
                target_user = args[u_idx + 1]
                args.pop(u_idx + 1)
            args.pop(u_idx)

    csv_path = args[0] if args else "data/behavior_logs.csv"

    try:
        if per_user_mode:
            features_per_user = extract_features_per_user(csv_path)
            users_to_show = (
                [target_user]
                if target_user
                else sorted(features_per_user.keys())
            )
            for user in users_to_show:
                if user not in features_per_user:
                    print(f"[错误] 用户不存在: {user}")
                    continue
                feats = features_per_user[user]
                result = score_risk(feats)
                print(
                    f"用户 {user} -> score={result['score']},"
                    f" level={result['level']}"
                )
        else:
            features = extract_features(csv_path)
            print(f"提取特征: {features}")
            result = score_risk(features)
            print(f"风险评分: {result}")
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {csv_path}")
        print(
            "请确保 CSV 文件存在，"
            "或运行 `uv run python scripts/generate_behavior_data.py` 生成测试数据。"
        )
    except Exception as e:
        print(f"[错误] {e}")


if __name__ == "__main__":
    main()
