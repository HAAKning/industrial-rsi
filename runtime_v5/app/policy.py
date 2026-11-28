# -*- coding: utf-8 -*-
"""漂移监测、策略白名单与受控决策。

设计原则（对应赛题"安全可控 + 工业落地"评分点）：
1. 大模型只能从 ALLOWED 白名单里选动作，选定后由本模块的 guard() 校验，
   越界即回落到启发式策略——模型不能直接改代码、改架构或无限迭代。
2. 每轮必须给出 baseline 对比（父检查点 vs 本轮候选），指标不改善就回滚，
   回滚本身也是一次"决策"，写进实验记录。
3. 所有阈值集中在这里，便于在报告里逐条解释。
"""
import json
import math

# ---------------------------------------------------------------- 策略空间
# 全局白名单是"最宽"的可选范围；每个数据集再按实测的敏感度收紧（见 DATASET_LIMITS）。
ALLOWED = {
    "fine_tune": {"epochs": [1, 2, 3], "lr": [0.001, 0.002]},     # 低扰动，稳
    "push": {"epochs": [4, 6, 8], "lr": [0.002, 0.003]},          # 正常推进
    "aggressive": {"epochs": [6, 10, 12], "lr": [0.004, 0.005]},  # 漂移大时
    "hold": {"epochs": [1], "lr": [0.001]},                       # 几乎不动，保分
}

# 依据 work/sweep 的 48 组网格实验收紧（public/validation 上测，父检查点→续训后）：
#   A 最优 e8/lr0.003（+0.0123 macro_f1），16 组全部改善，lr≤0.003 稳定；
#   B 最优 e2/lr0.005（-43.7 rmse），但 e8 明显过拟合（e8/lr0.003 只 -2.6），
#     lr 越小越需要少步数，因此 B 的 epochs 上限定在 4；
#   C 最优 e8/lr0.001（-1.13 rmse），lr≥0.003 在 1~2 轮反而比父模型差，
#     因此 C 的 lr 上限压到 0.002、大步数只配小 lr。
_NARROW = {"epochs": [1], "lr": [0.001]}
DATASET_LIMITS = {
    "A": {"fine_tune": {"epochs": [1, 2], "lr": [0.002, 0.003]},
          "push": {"epochs": [4, 8], "lr": [0.003]},
          "aggressive": {"epochs": [8], "lr": [0.003, 0.005]},
          "hold": _NARROW},
    "B": {"fine_tune": {"epochs": [1, 2], "lr": [0.003, 0.005]},
          "push": {"epochs": [4], "lr": [0.002, 0.003]},
          "aggressive": {"epochs": [2, 4], "lr": [0.005]},
          "hold": _NARROW},
    "C": {"fine_tune": {"epochs": [1, 2], "lr": [0.001, 0.002]},
          "push": {"epochs": [4, 8], "lr": [0.001]},
          "aggressive": {"epochs": [8], "lr": [0.002]},
          "hold": _NARROW},
}

# 漂移等级 → 默认动作（与网格实验结论一致：低漂移小步、高漂移加步数但不加学习率）
DEFAULT_PLAN = {
    "low": ("fine_tune", 1, 0.001),
    "medium": ("fine_tune", 2, 0.002),
    "high": ("push", 4, 0.003),
}
MAX_TOTAL_EPOCHS = 40          # 单轮硬上限，防无限迭代
WORSE_TOLERANCE = 0.0          # 候选必须严格不劣于父模型才采纳


def limits_for(name, dataset=None):
    """取某数据集下该动作的可选 epochs/lr（没有该数据集配置就退回全局白名单）。"""
    table = DATASET_LIMITS.get(dataset or "", {})
    return table.get(name) or ALLOWED.get(name)


def _arr(rows):
    import numpy as np
    return np.asarray(rows, dtype=np.float64)


def feature_drift(train_x, val_x):
    """逐特征标准化均值偏移：|E_val - E_train| / (std_train + eps)。

    返回均值/最大值/超阈值特征占比，作为"输入分布漂移"的粗指标。
    """
    tx, vx = _arr(train_x), _arr(val_x)
    if tx.ndim != 2 or tx.shape[1] == 0 or vx.size == 0:
        return {"mean_z": 0.0, "max_z": 0.0, "frac_z_gt_1": 0.0}
    mt, st = tx.mean(axis=0), tx.std(axis=0)
    z = abs(vx.mean(axis=0) - mt) / (st + 1e-9)
    return {"mean_z": float(z.mean()), "max_z": float(z.max()),
            "frac_z_gt_1": float((z > 1.0).mean())}


def target_shift(train_y, val_y):
    ty, vy = _arr(train_y), _arr(val_y)
    mt, st = float(ty.mean()), float(ty.std())
    return {"mean_z": abs(float(vy.mean()) - mt) / (st + 1e-9),
            "train_mean": mt, "val_mean": float(vy.mean())}


def drift_level(dr):
    """把漂移指标压成一个等级，决定可用策略的上限。"""
    if dr["feature"]["mean_z"] >= 0.35 or dr["feature"]["frac_z_gt_1"] >= 0.25:
        return "high"
    if dr["feature"]["mean_z"] >= 0.15 or dr["feature"]["frac_z_gt_1"] >= 0.10:
        return "medium"
    return "low"


def allowed_actions(level):
    if level == "high":
        return ["push", "aggressive", "fine_tune", "hold"]
    if level == "medium":
        return ["fine_tune", "push", "hold"]
    return ["fine_tune", "hold", "push"]


def guard(strategy, level, dataset=None):
    """校验大模型给出的策略；不合法就回落到与漂移等级匹配的默认策略。

    两重校验：动作名必须在白名单且与漂移等级相称；epochs/lr 必须落在
    该数据集实测过的取值里（越界就换成该动作下的首选值，不放大也不缩小步长）。
    """
    name = strategy.get("strategy") if isinstance(strategy, dict) else None
    if not isinstance(name, str) or name not in ALLOWED or name not in allowed_actions(level):
        fallback = default_plan(level, dataset)
        fallback["why"] = "guard: 模型给出的策略不在白名单，回落到按数据集收紧的默认策略"
        return fallback
    limits = limits_for(name, dataset)
    epochs = strategy.get("epochs")
    lr = strategy.get("lr")
    guarded = False
    if type(epochs) is not int or epochs not in limits["epochs"]:
        epochs = limits["epochs"][0]
        guarded = True
    if type(lr) not in (int, float) or float(lr) not in [float(x) for x in limits["lr"]]:
        lr = limits["lr"][0]
        guarded = True
    return {"strategy": name, "epochs": min(epochs, MAX_TOTAL_EPOCHS),
            "lr": float(lr), "why": str(strategy.get("why", ""))[:400], "guarded": guarded}


def default_plan(level, dataset=None):
    """按漂移等级给出默认策略，再把取值夹到该数据集的允许范围内。"""
    name, epochs, lr = DEFAULT_PLAN[level]
    limits = limits_for(name, dataset)
    if epochs not in limits["epochs"]:
        epochs = limits["epochs"][0]
    if float(lr) not in [float(x) for x in limits["lr"]]:
        lr = limits["lr"][0]
    return {"strategy": name, "epochs": epochs, "lr": float(lr),
            "why": "默认策略（按漂移等级）", "guarded": True}


def history_trend(history):
    """从平台历史里取本数据集最近几轮的指标，判断是变好还是退化。"""
    rows = []
    for item in history or []:
        if item.get("dataset_key") not in (None, "A", "B", "C"):
            continue
        m = item.get("measurements") or item.get("metrics") or {}
        rows.append({"round": item.get("round_index"), "metrics": m})
    return rows[-5:]


def summarize_metrics(report):
    """把 baseline 训练报告压成可比的一行。"""
    return {name: round(float(value), 4)
            for name, value in (report.get("validation_after") or {}).items()}


def is_improvement(before, after, key):
    """A 看 macro_f1（越大越好），B/C 看 rmse（越小越好）。"""
    if key not in before or key not in after or before[key] is None:
        return False
    if key in ("macro_f1", "accuracy"):
        return after[key] > before[key] + WORSE_TOLERANCE
    return after[key] < before[key] - WORSE_TOLERANCE


def main_metric(task):
    return "macro_f1" if task == "classification" else "rmse"


def render_diagnostics(ctx):
    """给大模型的诊断摘要（纯 JSON 串，减少 token）。"""
    return json.dumps(ctx, ensure_ascii=False, sort_keys=True)[:6000]
