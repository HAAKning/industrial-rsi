# -*- coding: utf-8 -*-
"""选手模块：真正的训练 / 预测逻辑，供 /app/harness.py 调用。

一轮 train 的闭环（对应赛题要求的"读历史—做实验—依据结果决策"）：
  1. 读本轮公开数据投影与父检查点；
  2. 算输入分布漂移与目标漂移，读平台历史里本数据集最近几轮成绩；
  3. 把诊断摘要交给大模型，请它从白名单里选一个策略（guard 会校验越界）；
  4. 用 baselines 引擎从父检查点续训，得到候选检查点；
  5. 比较候选与父模型的验证指标：改善则采纳，否则回滚到父检查点并如实记录；
  6. submit_candidate + finish，把假设、改动、理由、用过的调用号写清楚。

预测走平台的 predict 协议：stdin 一行样本，stdout 一行结果。
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

import numpy as np

import policy
from baselines.data import read_dataset, json_value
from baselines.model import Model
from baselines.train import train_run

CONFIG_DIR = APP / "baselines" / "configs"

SYSTEM_PROMPT = (
    "你是一个工业模型自进化系统的决策器。你不写代码，只做选择。\n"
    "每轮你会拿到：数据集、父检查点、公开训练/验证数据的漂移诊断、以及历史轮次成绩。\n"
    "你必须只输出一行 JSON：{\"strategy\": 名称, \"epochs\": 整数, \"lr\": 数值, \"why\": \"不超过60字的理由\"}。\n"
    "可选 strategy 与对应的 epochs/lr 取值见输入中的 allowed_actions。\n"
    "判断原则：漂移小、历史平稳 → 小步微调（省算力）；漂移大或历史退化 → 加大步长；\n"
    "只有一次机会时必须保守，宁可小幅提升也不要破坏已验证的模型。\n"
    "只输出 JSON，不要解释，不要代码块。"
)


# ---------------------------------------------------------------- 工具函数
def _env(name, default=None):
    return os.environ.get(name, default)


def _read_jsonl_rows(rows):
    """rnd.data.*_rows() 返回的可能是 dict 行或字符串行，统一成 dict。"""
    out = []
    for row in rows:
        out.append(row if isinstance(row, dict) else json.loads(row))
    return out


def _dump_rows(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _public_paths(rnd, workdir):
    """拿到本轮公开投影的两份 JSONL；平台给路径就用路径，否则自己落盘。"""
    task = getattr(rnd, "task", None) or {}
    projection = (task.get("public_projection") if isinstance(task, dict) else None) or {}
    train_path = (projection.get("train_path") or projection.get("train")
                  or projection.get("training"))
    val_path = projection.get("validation_path") or projection.get("validation")
    if train_path and val_path and os.path.exists(train_path) and os.path.exists(val_path):
        return train_path, val_path
    data = getattr(rnd, "data", None)
    if data is None:
        raise RuntimeError("找不到公开数据：既没有 public_projection，也没有 rnd.data")
    train_rows = _read_jsonl_rows(data.training_rows())
    val_rows = _read_jsonl_rows(data.validation_rows())
    return (_dump_rows(train_rows, workdir / "train.jsonl"),
            _dump_rows(val_rows, workdir / "validation.jsonl"))


def _profile(profile_dir, key):
    """把包里的 configs/<key>.json 复制到工作目录，避免改到只读的 /app。"""
    src = CONFIG_DIR / f"{key}.json"
    dst = Path(profile_dir) / f"{key}.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return str(dst)


def _ask_model(rnd, ctx):
    """请大模型选策略；任何异常都降级为启发式，不中断整轮。

    返回 (proposal_or_None, call_id_or_None, note)。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": policy.render_diagnostics(ctx)},
    ]
    try:
        receipt = rnd.model(messages, max_tokens=512, temperature=0)
    except Exception as exc:                      # noqa: BLE001 - 平台异常不应炸掉训练
        return None, None, f"model_unavailable: {type(exc).__name__}: {exc}"[:300]
    text = getattr(receipt, "text", None) or ""
    call_id = getattr(receipt, "call_id", None)
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1]
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        proposal = json.loads(raw[start:end])
    except Exception:                              # noqa: BLE001
        return None, call_id, "model_reply_unparsable"
    return proposal, call_id, ""


def _previous_attempts(state_dir):
    """读上一轮带过来的 experiments.jsonl，给决策器看最近几次尝试的结果。"""
    path = Path(state_dir) / "experiments.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-3:]:
        try:
            rec = json.loads(line)
        except Exception:                          # noqa: BLE001
            continue
        rows.append({"round": rec.get("round"), "strategy": (rec.get("plan") or {}).get("strategy"),
                     "epochs": (rec.get("plan") or {}).get("epochs"),
                     "lr": (rec.get("plan") or {}).get("lr"),
                     "improved": rec.get("improved"),
                     "after": rec.get("after")})
    return rows


# ---------------------------------------------------------------- 协议自描述
def describe():
    """如果平台的 harness 模板来问我们要自描述，就给它这一份。

    模板自己也能拼出同样的 JSON（见 SDK 的 template/harness.py），这里提供
    一份只是为了不依赖模板版本——多一个入口不花钱。
    """
    return {
        "protocol": "contest.harness.workspace.v2",
        "model_access": "host_gateway_rpc",
        "tasks": ["classification", "regression"],
    }


# ---------------------------------------------------------------- 训练
def train(rnd):
    key = rnd.dataset_key
    task = (getattr(rnd, "task", None) or {}).get("task")
    if not task:
        task = "classification" if key == "A" else "regression"
    round_index = getattr(rnd, "round_index", 1)
    workdir = Path(_env("WORKSPACE_DIR", "/workspace")) / f"round-{round_index:02d}"
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. 父检查点
    parent_path = Path(_env("PARENT_CHECKPOINT") or "/previous/checkpoint.npz")
    if not parent_path.is_file():
        parent_path = Path(rnd.parent.path("checkpoint.npz"))
    if not parent_path.is_file():
        raise FileNotFoundError(f"父检查点不可用: {parent_path}")
    parent_copy = workdir / "parent.npz"
    shutil.copyfile(parent_path, parent_copy)

    # 2. 公开数据与漂移诊断
    train_path, val_path = _public_paths(rnd, workdir)
    classes = json.loads((CONFIG_DIR / f"{key}.json").read_text(encoding="utf-8")).get("classes", [])
    tr = read_dataset(train_path, task, classes)
    va = read_dataset(val_path, task, classes, tr["paths"])
    drift = {
        "feature": policy.feature_drift(tr["x"], va["x"]),
        "target": policy.target_shift(tr["y"], va["y"]),
    }
    level = policy.drift_level(drift)
    trend = policy.history_trend(getattr(rnd, "history", lambda: [])())

    ctx = {
        "dataset_key": key,
        "task": task,
        "round_index": round_index,
        "main_metric": policy.main_metric(task),
        "train_rows": len(tr["ids"]),
        "validation_rows": len(va["ids"]),
        "drift": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                  for k, v in drift.items()},
        "drift_level": level,
        "allowed_actions": {name: policy.limits_for(name, key) for name in policy.allowed_actions(level)},
        "history": trend,
        "previous_attempts": _previous_attempts(Path(os.environ.get("STATE_DIR", "/state"))),
        "parent_checkpoint_sha256": getattr(rnd.parent, "checkpoint_sha256", None),
    }

    # 3. 决策（大模型提议 + guard 校验）
    asked, call_id, note = _ask_model(rnd, ctx)
    if asked is None:
        plan = policy.default_plan(level, key)
        plan["why"] = f"模型不可用（{note}），按漂移等级走默认策略"
    else:
        plan = policy.guard(asked, level, key)

    # 4. 续训
    out_dir = workdir / "candidate"
    started = time.time()
    report = train_run(
        _profile(workdir, key),
        train_path,
        val_path,
        out_dir,
        resume=str(parent_copy),
        epochs=int(plan["epochs"]),
        learning_rate=float(plan["lr"]),
        stage_id=f"r{round_index}-{plan['strategy']}",
    )
    seconds = time.time() - started

    # 5. 采纳或回滚
    metric = policy.main_metric(task)
    before = report["validation_before"]
    after = report["validation_after"]
    improved = policy.is_improvement(before, after, metric)
    chosen_dir = out_dir if improved else workdir / "rollback"
    if not improved:
        chosen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent_copy, chosen_dir / "checkpoint.npz")
        (chosen_dir / "training.json").write_text(json.dumps(
            {"note": "rollback: 候选未达父模型指标，保留父检查点",
             "candidate_metrics": after, "parent_metrics": before},
            ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. 提交候选
    hypothesis = f"{key} 第{round_index}轮：漂移等级={level}，策略={plan['strategy']}"
    change = (f"学习率 {plan['lr']}，epochs {plan['epochs']}；"
              f"{metric} {before.get(metric)} -> {after.get(metric)}；"
              f"{'采纳' if improved else '回滚保留父模型'}")
    submitted = rnd.submit_candidate(
        str(chosen_dir),
        hypothesis=hypothesis,
        change_summary=change[:4000],
        configuration={"epochs": plan["epochs"], "lr": plan["lr"], "strategy": plan["strategy"],
                       "drift_level": level, "improved": improved, "seconds": round(seconds, 3)},
        seed=42,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=["python3", "/app/harness.py", "predict"],
    )
    artifact_id = submitted["artifact_id"] if isinstance(submitted, dict) else submitted

    # 7. 状态：把本轮实验记录留给下一轮
    try:
        state = rnd.state_workspace(str(workdir / "state"))
        with open(os.path.join(str(state), "experiments.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"round": round_index, "dataset_key": key, "drift_level": level,
                                 "plan": plan, "before": before, "after": after,
                                 "improved": improved, "seconds": round(seconds, 3),
                                 "parent_sha": ctx["parent_checkpoint_sha256"]},
                                ensure_ascii=False) + "\n")
        rnd.save_state(state)
    except Exception:                              # noqa: BLE001
        pass

    # 8. 结束本轮
    call_ids = list(getattr(rnd, "call_ids", []) or ([] if call_id is None else [call_id]))
    history_ids = [item.get("id") for item in (getattr(rnd, "history", lambda: [])() or [])
                   if isinstance(item, dict) and item.get("id")]
    payload = dict(
        selected_artifact_id=artifact_id,
        hypothesis=hypothesis,
        change_summary=change[:4000],
        reason=(f"{plan['why']}｜验证集 {metric}：{before.get(metric)} -> {after.get(metric)}，"
                f"{'采纳新候选' if improved else '未改善，回滚到父模型'}")[:2000],
        configuration={"epochs": plan["epochs"], "lr": plan["lr"], "strategy": plan["strategy"]},
        seed=42,
        history_ids=history_ids,
        model_call_ids=call_ids,
    )
    rnd.finish(**payload)
    return {"artifact_id": artifact_id, "improved": improved, "metrics": after}


# ---------------------------------------------------------------- 预测
def _load_model(model_dir=None):
    path = _env("CONTEST_ARTIFACT_PATH")
    if path and os.path.exists(path):
        return Model.load(path)
    base = Path(model_dir or _env("MODEL_DIR", "/model"))
    for name in ("checkpoint.npz", "model.npz"):
        candidate = base / name
        if candidate.exists():
            return Model.load(str(candidate))
    raise FileNotFoundError(f"/model 下找不到检查点: {base}")


def predict(sample, model_dir=None):
    """协议入口。model_dir 可省：平台的模板可能只传 sample，产物路径走
    CONTEST_ARTIFACT_PATH 或 /model，两种调用方式都能接。"""
    model = _load_model(model_dir)
    inputs = getattr(sample, "inputs", None)
    if inputs is None and isinstance(sample, dict):
        inputs = sample.get("inputs")
    from baselines.predict import predict as core_predict
    result = core_predict(model, inputs)
    task = getattr(sample, "task", None) or (sample.get("task") if isinstance(sample, dict) else None)
    if task and task != model.meta["task"]:
        raise ValueError("prediction task mismatch")
    if model.meta["task"] == "classification":
        return result["outputs"]
    # 回归：协议要求 {"value": 非负有限数}，RUL 不允许为负，负值截到 0
    value = float(result["outputs"]["value"])
    return {"value": 0.0 if value < 0.0 else value}
