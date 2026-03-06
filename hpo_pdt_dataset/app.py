import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import dice_ml
import numpy as np
import pandas as pd
import streamlit as st
import torch
from catboost import CatBoostRegressor
from dice_ml import Dice
from jsonschema import validate
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEFAULT_CSV_PATH = Path("surrogate_ready_dataset/patchcore_surrogate_dataset_xgb.csv")
DEFAULT_MODEL_PATH = Path("surrogate_pkl_cfs_metadata/surrogate_catboost.cbm")
DEFAULT_EXPORT_DIR = Path("surrogate_pkl_cfs_metadata")
DEFAULT_LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LLM_MODEL_OPTIONS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
TARGET = "value"

FEATURES = [
    "params_backbone",
    "params_batch_size",
    "params_center_crop_key",
    "params_coreset_sampling_ratio",
    "params_image_size_key",
    "params_layers_key",
    "params_max_patches_per_image",
    "params_num_neighbors",
    "params_pre_trained",
    "params_reduction",
    "params_soft_corruption_level",
    "params_soft_review_budget",
    "params_soft_train_fraction",
]

CATEGORICAL = [
    "params_backbone",
    "params_batch_size",
    "params_center_crop_key",
    "params_image_size_key",
    "params_layers_key",
    "params_max_patches_per_image",
    "params_pre_trained",
    "params_reduction",
    "params_soft_corruption_level",
]

SOFT_TRAIN_FRACTION_RANGE = (0.20, 1.00)
SOFT_CORRUPTION_LEVELS = ["none", "mild", "strong"]
SOFT_REVIEW_BUDGETS = [i / 1000 for i in range(5, 501, 5)]
SOFT_REVIEW_BUDGET_BOUNDS = (SOFT_REVIEW_BUDGETS[0], SOFT_REVIEW_BUDGETS[-1])

DEFAULT_EPS = 0.01
DEFAULT_K = 5
MAX_RETRIES = 3

DEFAULT_USER_TEXT = (
    "I want better quality.\n"
    "Improve quality by +0.02.\n"
    "Freeze review budget to the current value.\n"
    "Allow corruption level to vary.\n"
    "Keep train fraction between 0.30 and 0.60.\n"
    "Return 7 options with a balanced strategy."
)

TARGET_LABEL = "Predicted quality (F1 score)"
FEATURE_LABELS = {
    "params_backbone": "Backbone",
    "params_batch_size": "Batch size",
    "params_center_crop_key": "Center crop",
    "params_coreset_sampling_ratio": "Coreset sampling ratio",
    "params_image_size_key": "Image size",
    "params_layers_key": "Layer set",
    "params_max_patches_per_image": "Max patches per image",
    "params_num_neighbors": "Nearest neighbors",
    "params_pre_trained": "Use pretrained weights",
    "params_reduction": "Reduction method",
    "params_soft_corruption_level": "Corruption level",
    "params_soft_review_budget": "Review budget",
    "params_soft_train_fraction": "Training fraction",
    TARGET: TARGET_LABEL,
}

EXAMPLE_REQUESTS = [
    (
        "Quality +0.02, budget fixed",
        "Improve quality by +0.02.\nFreeze review budget to the current value.\nReturn 7 options with a balanced strategy.",
    ),
    (
        "High quality, minimal change",
        "Minimum quality should be 0.85.\nKeep train fraction to the current value.\nReturn 5 options, closest strategy.",
    ),
    (
        "Broader exploration",
        "Improve quality by +0.01.\nAllow review budget to vary.\nAllow corruption level to vary.\nReturn 10 options with a diverse strategy.",
    ),
    (
        "Strict safe band",
        "Keep quality between 0.80 and 0.92.\nKeep train fraction between 0.40 and 0.70.\nReturn 6 options with a balanced strategy.",
    ),
]

QUERY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["objective", "soft_constraints", "selection"],
    "properties": {
        "objective": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type"],
            "properties": {
                "type": {"type": "string", "enum": ["target_min", "delta_improve", "target_band"]},
                "value": {"type": "number"},
                "delta": {"type": "number"},
                "lower": {"type": "number"},
                "upper": {"type": "number"},
                "eps": {"type": "number"},
            },
        },
        "soft_constraints": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "params_soft_train_fraction",
                "params_soft_review_budget",
                "params_soft_corruption_level",
            ],
            "properties": {
                "params_soft_train_fraction": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["mode"],
                    "properties": {
                        "mode": {"type": "string", "enum": ["free", "fixed", "range"]},
                        "value": {"type": "number"},
                        "lower": {"type": "number"},
                        "upper": {"type": "number"},
                    },
                },
                "params_soft_review_budget": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["mode"],
                    "properties": {
                        "mode": {"type": "string", "enum": ["free", "fixed", "range"]},
                        "value": {"type": "number"},
                        "lower": {"type": "number"},
                        "upper": {"type": "number"},
                    },
                },
                "params_soft_corruption_level": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["mode"],
                    "properties": {
                        "mode": {"type": "string", "enum": ["free", "fixed", "allowed"]},
                        "value": {"type": "string", "enum": ["none", "mild", "strong"]},
                        "allowed": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "enum": SOFT_CORRUPTION_LEVELS},
                        },
                    },
                },
            },
        },
        "selection": {
            "type": "object",
            "additionalProperties": False,
            "required": ["k"],
            "properties": {
                "k": {"type": "integer", "minimum": 1, "maximum": 50}
            },
        },
    },
}

CONTRACT_PROMPT = f"""
Return ONLY valid JSON. No markdown. No extra text.

Keys required: objective, soft_constraints, selection.

objective.type in: target_min | delta_improve | target_band

soft_constraints must contain:
- params_soft_train_fraction: mode free|fixed|range
- params_soft_review_budget:  mode free|fixed|range
- params_soft_corruption_level: mode free|fixed|allowed

selection must contain:
- k (1..50)

Numeric constraints:
- eps > 0
- if target_band: lower <= upper
- if range: lower <= upper

Intent rules:
- If user provides explicit numeric value/range for a soft constraint, keep that value/range (do not widen to free/full range).

Allowed corruption values: {SOFT_CORRUPTION_LEVELS}
Train fraction allowed range: {SOFT_TRAIN_FRACTION_RANGE}
Review budget allowed values are multiples of 0.005 in [0.005, 0.5].
"""


def feature_label(feat: str) -> str:
    if feat in FEATURE_LABELS:
        return FEATURE_LABELS[feat]
    return feat.replace("params_", "").replace("_", " ").strip().title()


def rename_df_columns_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Rename dataframe columns using FEATURE_LABELS where possible."""
    if df is None or df.empty:
        return df
    return df.rename(columns=lambda c: FEATURE_LABELS.get(c, c))

def rename_violations_text_for_ui(s: Any) -> Any:
    """
    In eval_df, the 'violations' column contains internal feature names joined by commas.
    Convert them to user-friendly labels.
    """
    if s is None:
        return s
    txt = str(s).strip()
    if not txt or txt.lower() == "none":
        return s
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    parts = [feature_label(p) for p in parts]  # feature_label uses FEATURE_LABELS fallback
    return ", ".join(parts)

def feature_value_text(feat: str, value: Any) -> str:
    if value is None:
        return "-"
    if feat in CATEGORICAL:
        return str(value)
    try:
        v = float(value)
    except Exception:
        return str(value)
    if feat in {"params_soft_train_fraction", "params_soft_review_budget", TARGET}:
        return f"{v:.3f}"
    return f"{v:.4g}"

def option_name(cf_id: int) -> str:
    cid = int(cf_id)
    if 0 <= cid < 26:
        return f"Option {chr(ord('A') + cid)}"
    return f"Option {cid + 1}"


def option_short_title(record: Dict[str, Any]) -> str:
    violated = record.get("violated_constraints") or []
    if violated:
        return f"Risk on {feature_label(str(violated[0]))}"
    top = record.get("top_changes") or []
    if top:
        feat = str(top[0]).split(":", 1)[0].strip()
        return f"Focus on {feature_label(feat)}"
    n_changes = int(record.get("changed_feature_count", 0))
    if n_changes == 0:
        return "No setup changes"
    if n_changes <= 2:
        return "Small setup update"
    return "Broader setup update"


def describe_tradeoff(record: Dict[str, Any]) -> str:
    violated = record.get("violated_constraints") or []
    if violated:
        names = ", ".join(feature_label(str(v)) for v in violated[:2])
        if len(violated) > 2:
            names += ", ..."
        return f"May violate: {names}."
    top = record.get("top_changes") or []
    if top:
        return f"Main adjustment: {top[0]}."
    return "No notable tradeoff detected."


def describe_objective_user(objective: Dict[str, Any], factual_score: float) -> str:
    t = objective.get("type")
    if t == "target_min":
        return f"Reach at least {float(objective.get('value', factual_score)):.3f} quality."
    if t == "delta_improve":
        delta = float(objective.get("delta", 0.0))
        return f"Improve quality by at least {delta:+.3f} from the current {factual_score:.3f}."
    if t == "target_band":
        lo = float(objective.get("lower", factual_score))
        hi = float(objective.get("upper", factual_score))
        return f"Keep quality between {lo:.3f} and {hi:.3f}."
    return "Use default quality objective."


def describe_soft_constraint_user(feat: str, spec: Dict[str, Any], factual_x: Dict[str, Any]) -> str:
    label = feature_label(feat)
    mode = spec.get("mode", "free")
    if mode == "free":
        return f"{label}: can vary."
    if mode == "fixed":
        val = spec.get("value", factual_x.get(feat))
        return f"{label}: fixed at {feature_value_text(feat, val)}."
    if mode == "range":
        lo = spec.get("lower")
        hi = spec.get("upper")
        return (
            f"{label}: between {feature_value_text(feat, lo)} and {feature_value_text(feat, hi)}."
            if lo is not None and hi is not None
            else f"{label}: range requested."
        )
    if mode == "allowed":
        vals = spec.get("allowed", [])
        return f"{label}: allowed values = {', '.join(map(str, vals))}."
    return f"{label}: custom rule."


def summarize_query_for_user(query: Dict[str, Any]) -> List[str]:
    factual_score = float(query.get("factual", {}).get("value_surrogate", 0.0))
    lines = [describe_objective_user(query["objective"], factual_score)]
    for feat, spec in query["soft_constraints"].items():
        lines.append(describe_soft_constraint_user(feat, spec, factual_x=query["factual"]["x"]))
    lines.append(f"Generate up to {int(query['selection']['k'])} options.")
    
    return lines


def compose_user_request_text(free_text: str, guided_lines: List[str]) -> str:
    parts: List[str] = []
    base = (free_text or "").strip()
    if base:
        parts.append(base)
    if guided_lines:
        parts.append("\n".join(guided_lines))
    return "\n".join(parts).strip()


def q01(x: float, lo: float, hi: float) -> float:
    x = max(lo, min(hi, float(x)))
    return round(x * 100.0) / 100.0


def snap_budget(x: float) -> float:
    x = float(x)
    return min(SOFT_REVIEW_BUDGETS, key=lambda v: abs(v - x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _model_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def extract_first_json_object(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced JSON braces in model output.")


def generate_greedy(
    tokenizer: AutoTokenizer, model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor], max_new_tokens: int
) -> torch.Tensor:
    gen_cfg = GenerationConfig(
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return model.generate(**inputs, generation_config=gen_cfg)


@torch.inference_mode()
def llama_generate_json(
    tokenizer: AutoTokenizer, model: AutoModelForCausalLM, user_text: str, max_new_tokens: int = 400
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "Output ONLY valid JSON. No extra text."},
            {"role": "user", "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "SYSTEM: Output ONLY valid JSON.\nUSER:\n" + user_text + "\nASSISTANT:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(_model_device(model))
    out = generate_greedy(tokenizer, model, inputs, max_new_tokens=max_new_tokens)
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return extract_first_json_object(decoded)


def fallback_query_core(
    x_factual: Dict[str, Any],
    y_factual_sur: float,
    eps: float = DEFAULT_EPS,
    k: int = DEFAULT_K,
) -> Dict[str, Any]:
    _ = y_factual_sur
    return {
        "objective": {"type": "delta_improve", "delta": 0.02, "eps": eps},
        "soft_constraints": {
            "params_soft_train_fraction": {"mode": "fixed", "value": float(x_factual["params_soft_train_fraction"])},
            "params_soft_review_budget": {"mode": "fixed", "value": float(x_factual["params_soft_review_budget"])},
            "params_soft_corruption_level": {"mode": "fixed", "value": str(x_factual["params_soft_corruption_level"])},
        },
        "selection": {"k": int(k)},
    }

def repair_and_snap(query: Dict[str, Any], x_factual: Dict[str, Any], y_factual_sur: float) -> Dict[str, Any]:
    q = copy.deepcopy(query)
    obj = q.get("objective", {})
    t = obj.get("type")

    if t == "target_band":
        lo = obj.get("lower", None)
        hi = obj.get("upper", None)
        if lo is None or hi is None:
            lo, hi = float(y_factual_sur), float(y_factual_sur)
        lo, hi = float(lo), float(hi)
        if lo > hi:
            lo, hi = hi, lo
        q["objective"] = {"type": "target_band", "lower": lo, "upper": hi, "eps": float(obj.get("eps", 0.01))}
    elif t == "target_min":
        v = obj.get("value", float(y_factual_sur))
        q["objective"] = {"type": "target_min", "value": float(v), "eps": float(obj.get("eps", 0.01))}
    elif t == "delta_improve":
        d = float(obj.get("delta", 0.0))
        q["objective"] = {"type": "delta_improve", "delta": d, "eps": float(obj.get("eps", 0.01))}
    else:
        lo = hi = float(y_factual_sur)
        q["objective"] = {"type": "target_band", "lower": lo, "upper": hi, "eps": 0.01}

    sel = q.get("selection", {})
    sel["k"] = int(sel.get("k", DEFAULT_K))
    sel["k"] = max(1, min(50, sel["k"]))
    q["selection"] = sel

    sc = q.get("soft_constraints", {})
    out: Dict[str, Dict[str, Any]] = {}

    def fallback_fixed(feat: str) -> Dict[str, Any]:
        return {"mode": "fixed", "value": x_factual.get(feat)}

    rb = sc.get("params_soft_review_budget", {"mode": "free"})
    mode = rb.get("mode", "free")
    if mode == "fixed":
        v = rb.get("value", x_factual.get("params_soft_review_budget"))
        v = float(snap_budget(_clamp(float(v), *SOFT_REVIEW_BUDGET_BOUNDS)))
        out["params_soft_review_budget"] = {"mode": "fixed", "value": v}
    elif mode == "range":
        lo = rb.get("lower", None)
        hi = rb.get("upper", None)
        if lo is None or hi is None:
            out["params_soft_review_budget"] = fallback_fixed("params_soft_review_budget")
        else:
            lo, hi = float(lo), float(hi)
            lo, hi = (hi, lo) if lo > hi else (lo, hi)
            lo = float(snap_budget(_clamp(lo, *SOFT_REVIEW_BUDGET_BOUNDS)))
            hi = float(snap_budget(_clamp(hi, *SOFT_REVIEW_BUDGET_BOUNDS)))
            out["params_soft_review_budget"] = {"mode": "range", "lower": lo, "upper": hi}
    elif mode == "free":
        out["params_soft_review_budget"] = {"mode": "free"}
    else:
        out["params_soft_review_budget"] = fallback_fixed("params_soft_review_budget")

    tf = sc.get("params_soft_train_fraction", {"mode": "free"})
    mode = tf.get("mode", "free")
    if mode == "fixed":
        v = tf.get("value", x_factual.get("params_soft_train_fraction"))
        v = q01(float(v), *SOFT_TRAIN_FRACTION_RANGE)
        out["params_soft_train_fraction"] = {"mode": "fixed", "value": v}
    elif mode == "range":
        lo = tf.get("lower", None)
        hi = tf.get("upper", None)
        if lo is None or hi is None:
            out["params_soft_train_fraction"] = fallback_fixed("params_soft_train_fraction")
        else:
            lo, hi = float(lo), float(hi)
            lo, hi = (hi, lo) if lo > hi else (lo, hi)
            lo = q01(lo, *SOFT_TRAIN_FRACTION_RANGE)
            hi = q01(hi, *SOFT_TRAIN_FRACTION_RANGE)
            out["params_soft_train_fraction"] = {"mode": "range", "lower": lo, "upper": hi}
    elif mode == "free":
        out["params_soft_train_fraction"] = {"mode": "free"}
    else:
        out["params_soft_train_fraction"] = fallback_fixed("params_soft_train_fraction")

    cl = sc.get("params_soft_corruption_level", {"mode": "free"})
    mode = cl.get("mode", "free")
    allowed_set = set(SOFT_CORRUPTION_LEVELS)
    if mode == "fixed":
        v = str(cl.get("value", x_factual.get("params_soft_corruption_level")))
        if v not in allowed_set:
            v = str(x_factual.get("params_soft_corruption_level", "none"))
            if v not in allowed_set:
                v = "none"
        out["params_soft_corruption_level"] = {"mode": "fixed", "value": v}
    elif mode == "allowed":
        allowed = cl.get("allowed", None)
        if not allowed:
            out["params_soft_corruption_level"] = fallback_fixed("params_soft_corruption_level")
        else:
            vals = [str(a) for a in allowed if str(a) in allowed_set]
            vals = sorted(set(vals))
            if vals:
                out["params_soft_corruption_level"] = {"mode": "allowed", "allowed": vals}
            else:
                out["params_soft_corruption_level"] = fallback_fixed("params_soft_corruption_level")
    elif mode == "free":
        out["params_soft_corruption_level"] = {"mode": "free"}
    else:
        out["params_soft_corruption_level"] = fallback_fixed("params_soft_corruption_level")

    q["soft_constraints"] = out
    return q


REQUIRED_SOFT_KEYS = {
    "params_soft_train_fraction",
    "params_soft_review_budget",
    "params_soft_corruption_level",
}


def normalize_objective(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("objective must be an object/dict")
    obj = dict(obj)
    t = obj.get("type", "delta_improve")

    if t == "target_min" and "value" not in obj and "target_min" in obj:
        obj["value"] = obj.pop("target_min")
        return obj
    if t == "delta_improve" and "delta" not in obj and "value" in obj:
        obj["delta"] = obj.pop("value")
        return obj
    if t == "target_band" and "lower" not in obj and "upper" not in obj and "value" in obj:
        val = obj.pop("value")
        if isinstance(val, (list, tuple)) and len(val) == 2:
            obj["lower"], obj["upper"] = val
        return obj
    if t == "target_band" and "target_band" in obj:
        val = obj.pop("target_band")
        if isinstance(val, (list, tuple)) and len(val) == 2:
            obj["lower"], obj["upper"] = val
        return obj
    return obj


def normalize_soft_constraints(sc: Dict[str, Any], x_factual: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sc, dict):
        raise ValueError("soft_constraints must be an object/dict")

    if REQUIRED_SOFT_KEYS.issubset(sc.keys()) and all(isinstance(sc[k], dict) for k in REQUIRED_SOFT_KEYS):
        return {k: sc[k] for k in REQUIRED_SOFT_KEYS}

    if "mode" in sc and ("values" in sc or "value" in sc):
        vals = sc.get("values", sc.get("value"))
        if isinstance(vals, dict):
            if REQUIRED_SOFT_KEYS.issubset(vals.keys()):
                return normalize_soft_constraints(vals, x_factual=x_factual)
            sc = vals

    out: Dict[str, Dict[str, Any]] = {}

    tf_mode = sc.get("params_soft_train_fraction")
    if isinstance(tf_mode, str) and tf_mode in {"free", "fixed", "range"}:
        if tf_mode == "free":
            out["params_soft_train_fraction"] = {"mode": "free"}
        elif tf_mode == "fixed":
            v = sc.get("train_fraction", sc.get("params_soft_train_fraction_value", sc.get("value")))
            if v is None:
                v = x_factual["params_soft_train_fraction"]
            out["params_soft_train_fraction"] = {"mode": "fixed", "value": v}
        else:
            lo = sc.get("train_fraction_min", sc.get("min", sc.get("lower")))
            hi = sc.get("train_fraction_max", sc.get("max", sc.get("upper")))
            if lo is None or hi is None:
                lo, hi = SOFT_TRAIN_FRACTION_RANGE
            out["params_soft_train_fraction"] = {"mode": "range", "lower": lo, "upper": hi}
    elif isinstance(sc.get("params_soft_train_fraction"), dict):
        out["params_soft_train_fraction"] = sc["params_soft_train_fraction"]

    rb_mode = sc.get("params_soft_review_budget")
    if isinstance(rb_mode, str) and rb_mode in {"free", "fixed", "range"}:
        if rb_mode == "free":
            out["params_soft_review_budget"] = {"mode": "free"}
        elif rb_mode == "fixed":
            v = sc.get("review_budget", sc.get("params_soft_review_budget_value", sc.get("value")))
            if v is None:
                v = x_factual["params_soft_review_budget"]
            out["params_soft_review_budget"] = {"mode": "fixed", "value": v}
        else:
            lo = sc.get("review_budget_min", sc.get("min_budget", sc.get("lower_budget")))
            hi = sc.get("review_budget_max", sc.get("max_budget", sc.get("upper_budget")))
            if lo is None or hi is None:
                lo, hi = SOFT_REVIEW_BUDGETS[0], SOFT_REVIEW_BUDGETS[-1]
            out["params_soft_review_budget"] = {"mode": "range", "lower": lo, "upper": hi}
    elif isinstance(sc.get("params_soft_review_budget"), dict):
        out["params_soft_review_budget"] = sc["params_soft_review_budget"]

    cl_mode = sc.get("params_soft_corruption_level")
    if isinstance(cl_mode, str) and cl_mode in {"free", "fixed", "allowed"}:
        if cl_mode == "free":
            out["params_soft_corruption_level"] = {"mode": "free"}
        elif cl_mode == "fixed":
            v = sc.get("corruption_level", sc.get("params_soft_corruption_level_value", sc.get("value")))
            if v is None:
                v = x_factual["params_soft_corruption_level"]
            out["params_soft_corruption_level"] = {"mode": "fixed", "value": v}
        else:
            vals = sc.get("allowed", sc.get("values", sc.get("levels")))
            out["params_soft_corruption_level"] = {"mode": "allowed", "allowed": vals if vals is not None else ["none"]}
    elif isinstance(sc.get("params_soft_corruption_level"), dict):
        out["params_soft_corruption_level"] = sc["params_soft_corruption_level"]

    if not REQUIRED_SOFT_KEYS.issubset(out.keys()):
        flat_hits = [k for k in REQUIRED_SOFT_KEYS if k in sc and not isinstance(sc[k], dict)]
        if flat_hits:
            for k in flat_hits:
                out[k] = {"mode": "fixed", "value": sc[k]}

    missing = REQUIRED_SOFT_KEYS - set(out.keys())
    if missing:
        raise ValueError(f"soft_constraints missing keys after normalization: {sorted(missing)}")
    return out


def normalize_selection(sel: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sel, dict):
        sel = {}
    return {"k": int(sel.get("k", DEFAULT_K))}


def _iter_user_lines(user_text: str):
    for part in re.split(r"[\r\n;]+", user_text.lower()):
        p = part.strip()
        if p:
            yield p


def extract_explicit_preferences(user_text: str, x_factual: Dict[str, Any] = None) -> Dict[str, Any]:
    x_factual = x_factual or {}
    prefs: Dict[str, Any] = {}
    soft: Dict[str, Any] = {}
    selection: Dict[str, Any] = {}
    objective = None

    for ln in _iter_user_lines(user_text):
        m_min = re.search(
            r"(?:min(?:imum)?\s*(?:performance|quality)?\s*(?:should\s*be|is|:)?|(?:performance|quality)\s*(?:should\s*be|is|:)?\s*at\s*least)\s*([0-9]*\.?[0-9]+)",
            ln,
        )
        if m_min:
            objective = {"type": "target_min", "value": float(m_min.group(1)), "eps": DEFAULT_EPS}
            continue

        m_delta = re.search(r"(?:improv(?:e|ing|ement)|increase|raise)\D*([+\-]?[0-9]*\.?[0-9]+)", ln)
        if m_delta and objective is None:
            d = float(m_delta.group(1))
            objective = {"type": "delta_improve", "delta": max(0.0, d), "eps": DEFAULT_EPS}

    if objective is not None:
        prefs["objective"] = objective

    for ln in _iter_user_lines(user_text):
        if "train fraction" in ln or "training fraction" in ln:
            m_tf_range = re.search(r"(?:between|from)\s*([0-9]*\.?[0-9]+)\s*(?:and|to|-)\s*([0-9]*\.?[0-9]+)", ln)
            if not m_tf_range:
                m_tf_range = re.search(r"([0-9]*\.?[0-9]+)\s*(?:and|to|-)\s*([0-9]*\.?[0-9]+)", ln)
            if m_tf_range:
                lo, hi = float(m_tf_range.group(1)), float(m_tf_range.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                soft["params_soft_train_fraction"] = {"mode": "range", "lower": lo, "upper": hi}
            else:
                m_tf_fixed = re.search(
                    r"(?:train(?:ing)?\s*fraction)\s*(?:should\s*be|=|is|:)?\s*([0-9]*\.?[0-9]+)", ln
                )
                if m_tf_fixed:
                    soft["params_soft_train_fraction"] = {"mode": "fixed", "value": float(m_tf_fixed.group(1))}

        if "review budget" in ln:
            if "current value" in ln or "freeze" in ln:
                v_cur = x_factual.get("params_soft_review_budget", None)
                if v_cur is not None:
                    soft["params_soft_review_budget"] = {"mode": "fixed", "value": float(v_cur)}
                    continue

            m_rb_range = re.search(r"(?:between|from)\s*([0-9]*\.?[0-9]+)\s*(?:and|to|-)\s*([0-9]*\.?[0-9]+)", ln)
            if not m_rb_range:
                m_rb_range = re.search(r"([0-9]*\.?[0-9]+)\s*(?:and|to|-)\s*([0-9]*\.?[0-9]+)", ln)
            if m_rb_range:
                lo, hi = float(m_rb_range.group(1)), float(m_rb_range.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                soft["params_soft_review_budget"] = {"mode": "range", "lower": lo, "upper": hi}
            else:
                m_rb_fixed = re.search(r"(?:review\s*budget)\s*(?:should\s*be|=|is|:)?\s*([0-9]*\.?[0-9]+)", ln)
                if m_rb_fixed:
                    soft["params_soft_review_budget"] = {"mode": "fixed", "value": float(m_rb_fixed.group(1))}

        if "corruption" in ln:
            if any(w in ln for w in ["free", "any", "vary", "variable"]):
                soft["params_soft_corruption_level"] = {"mode": "free"}
            else:
                vals = [v for v in SOFT_CORRUPTION_LEVELS if re.search(rf"\b{re.escape(v)}\b", ln)]
                vals = sorted(set(vals))
                if len(vals) == 1:
                    soft["params_soft_corruption_level"] = {"mode": "fixed", "value": vals[0]}
                elif len(vals) > 1:
                    soft["params_soft_corruption_level"] = {"mode": "allowed", "allowed": vals}

        m_k = re.search(r"\b(\d+)\s*(?:options?|cfs?|counterfactuals?)\b", ln)
        if m_k:
            selection["k"] = int(m_k.group(1))

    if soft:
        prefs["soft_constraints"] = soft
    if selection:
        prefs["selection"] = selection
    return prefs


def apply_explicit_preferences(query: Dict[str, Any], user_text: str, x_factual: Dict[str, Any] = None) -> Dict[str, Any]:
    q = copy.deepcopy(query)
    prefs = extract_explicit_preferences(user_text, x_factual=x_factual)

    if "objective" in prefs:
        q["objective"] = prefs["objective"]
    if "soft_constraints" in prefs:
        q.setdefault("soft_constraints", {})
        for k, v in prefs["soft_constraints"].items():
            q["soft_constraints"][k] = v
    if "selection" in prefs:
        q.setdefault("selection", {})
        if "k" in prefs["selection"]:
            q["selection"]["k"] = int(prefs["selection"]["k"])
    return q


def build_query_core(
    user_text: str,
    x_factual: Dict[str, Any],
    y_factual_sur: float,
    llama_json_fn: Callable[[str], str],
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            prompt = CONTRACT_PROMPT + "\n\nUSER REQUEST:\n" + user_text.strip()
            raw = llama_json_fn(prompt)
            draft = json.loads(raw)

            core = dict(draft)
            if "objective" in core:
                core["objective"] = normalize_objective(core["objective"])
            if "soft_constraints" in core:
                core["soft_constraints"] = normalize_soft_constraints(core["soft_constraints"], x_factual=x_factual)
            if "selection" in core:
                core["selection"] = normalize_selection(core["selection"])

            validate(instance=core, schema=QUERY_SCHEMA)
            repaired = repair_and_snap(core, x_factual=x_factual, y_factual_sur=y_factual_sur)
            repaired = apply_explicit_preferences(repaired, user_text=user_text, x_factual=x_factual)
            repaired = repair_and_snap(repaired, x_factual=x_factual, y_factual_sur=y_factual_sur)
            validate(instance=repaired, schema=QUERY_SCHEMA)
            repaired["_meta"] = {"used_fallback": False, "attempts": attempt + 1}
            return repaired
        except Exception as exc:
            last_error = exc

    core = fallback_query_core(x_factual=x_factual, y_factual_sur=y_factual_sur)
    core = apply_explicit_preferences(core, user_text=user_text, x_factual=x_factual)
    core = repair_and_snap(core, x_factual=x_factual, y_factual_sur=y_factual_sur)
    core["_meta"] = {"used_fallback": True, "reason": str(last_error)[:250] if last_error else "unknown"}
    return core


def objective_to_desired_range(query: Dict[str, Any]) -> List[float]:
    obj = query["objective"]
    factual_base = float(query["factual"]["value_surrogate"])

    if obj["type"] == "target_min":
        lo, hi = float(obj["value"]), 1.0
    elif obj["type"] == "delta_improve":
        lo, hi = factual_base + float(obj["delta"]), 1.0
    elif obj["type"] == "target_band":
        lo, hi = float(obj["lower"]), float(obj["upper"])
    else:
        raise ValueError(f"Unknown objective type: {obj['type']}")

    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    if lo > hi:
        lo, hi = hi, lo
    return [lo, hi]



def generate_cf_table(
    df: pd.DataFrame, cb: CatBoostRegressor, query: Dict[str, Any], desired_range: List[float]
) -> Tuple[pd.DataFrame, Dict[str, Any], int, int, int]:
    df_dice = df[FEATURES + [TARGET]].copy()
    continuous = [c for c in FEATURES if c not in CATEGORICAL]
    data_dice = dice_ml.Data(dataframe=df_dice, continuous_features=continuous, outcome_name=TARGET)
    model_dice = dice_ml.Model(model=cb, backend="sklearn", model_type="regressor")
    exp_dice = Dice(data_dice, model_dice, method="random")
    x0_dice = pd.DataFrame([query["factual"]["x"]])[FEATURES].copy()

    def domain_for_feature(feat: str):
        if feat in CATEGORICAL:
            return sorted(df_dice[feat].dropna().astype(str).unique().tolist())
        col = pd.to_numeric(df_dice[feat], errors="coerce")
        return [float(np.nanmin(col)), float(np.nanmax(col))]

    def factual_value(feat: str):
        v = x0_dice.iloc[0][feat]
        return str(v) if feat in CATEGORICAL else float(v)

    def safe_range(lo: Any, hi: Any, feat: str):
        try:
            lo = float(lo)
            hi = float(hi)
            if lo > hi:
                return None
            dom_lo, dom_hi = domain_for_feature(feat)
            lo = max(dom_lo, lo)
            hi = min(dom_hi, hi)
            if lo > hi:
                return None
            return [float(lo), float(hi)]
        except Exception:
            return None

    def safe_allowed(allowed: List[Any], feat: str):
        try:
            dom = set(map(str, domain_for_feature(feat)))
            allowed_str = [str(a) for a in allowed]
            allowed_str = [a for a in allowed_str if a in dom]
            return sorted(set(allowed_str)) if allowed_str else None
        except Exception:
            return None

    strict_permitted_range: Dict[str, Any] = {}
    for feat, spec in query["soft_constraints"].items():
        mode = spec.get("mode", "free")

        if mode == "free":
            strict_permitted_range[feat] = domain_for_feature(feat)
        elif mode == "fixed":
            if feat in CATEGORICAL:
                v = str(spec.get("value", factual_value(feat)))
                ok = safe_allowed([v], feat) or safe_allowed([factual_value(feat)], feat)
                strict_permitted_range[feat] = ok
            else:
                v = spec.get("value", factual_value(feat))
                r = safe_range(v, v, feat) or safe_range(factual_value(feat), factual_value(feat), feat)
                strict_permitted_range[feat] = r
        elif mode == "range":
            if feat in CATEGORICAL:
                strict_permitted_range[feat] = safe_allowed([factual_value(feat)], feat)
            else:
                lo = spec.get("lower", None)
                hi = spec.get("upper", None)
                if lo is None or hi is None:
                    strict_permitted_range[feat] = safe_range(factual_value(feat), factual_value(feat), feat)
                else:
                    r = safe_range(lo, hi, feat)
                    strict_permitted_range[feat] = (
                        r if r is not None else safe_range(factual_value(feat), factual_value(feat), feat)
                    )
        elif mode == "allowed":
            if feat not in CATEGORICAL:
                strict_permitted_range[feat] = safe_range(factual_value(feat), factual_value(feat), feat)
            else:
                allowed = spec.get("allowed", None)
                ok = safe_allowed(allowed, feat) if allowed is not None else None
                strict_permitted_range[feat] = ok if ok is not None else safe_allowed([factual_value(feat)], feat)
        else:
            if feat in CATEGORICAL:
                strict_permitted_range[feat] = safe_allowed([factual_value(feat)], feat)
            else:
                strict_permitted_range[feat] = safe_range(factual_value(feat), factual_value(feat), feat)

    strict_permitted_range = {k: v for k, v in strict_permitted_range.items() if v is not None}
    full_domain_permitted = {feat: domain_for_feature(feat) for feat in FEATURES}

    k = int(query["selection"]["k"])
    base_pool = int(max(20, 5 * k))

    lo0, hi0 = float(desired_range[0]), float(desired_range[1])
    lo0 = float(np.clip(lo0, 0.0, 1.0))
    hi0 = float(np.clip(hi0, 0.0, 1.0))
    if lo0 > hi0:
        lo0, hi0 = hi0, lo0

    lo1, hi1 = max(0.0, lo0 - 0.02), hi0
    if lo1 > hi1:
        lo1, hi1 = hi1, lo1

    attempts = [
        ([lo0, hi0], strict_permitted_range, base_pool),
        ([lo1, hi1], strict_permitted_range, max(base_pool, 60)),
        ([0.0, 1.0], full_domain_permitted, max(base_pool, 120)),
    ]

    for attempt_range, attempt_permitted, attempt_pool in attempts:
        try:
            dice_res = exp_dice.generate_counterfactuals(
                x0_dice,
                total_CFs=attempt_pool,
                desired_range=attempt_range,
                features_to_vary=FEATURES,
                permitted_range=attempt_permitted,
            )
        except Exception as exc:
            if "no counterfactuals found" in str(exc).lower():
                continue
            raise

        ex = dice_res.cf_examples_list[0]
        cf_raw = ex.final_cfs_df_sparse.copy() if hasattr(ex, "final_cfs_df_sparse") else ex.final_cfs_df.copy()
        if TARGET not in cf_raw.columns:
            raise ValueError(f"DiCE CF table missing '{TARGET}'. Columns: {list(cf_raw.columns)}")

        cf_df = cf_raw[FEATURES + [TARGET]].copy()
        for c in CATEGORICAL:
            if c in cf_df.columns:
                cf_df[c] = cf_df[c].astype(str)

        before_filter_count = len(cf_df)
        lo, hi = attempt_range
        cf_df = cf_df[(cf_df[TARGET] >= lo) & (cf_df[TARGET] <= hi)].copy()
        after_range_filter_count = len(cf_df)

        if after_range_filter_count == 0:
            continue

        cf_df = cf_df.sort_values(TARGET, ascending=False).reset_index(drop=True).head(k)
        return cf_df, attempt_permitted, attempt_pool, before_filter_count, after_range_filter_count

    empty_cf_df = pd.DataFrame(columns=FEATURES + [TARGET])
    return empty_cf_df, strict_permitted_range, base_pool, 0, 0


def export_artifacts(
    cf_df: pd.DataFrame,
    query: Dict[str, Any],
    desired_range: List[float],
    permitted_range: Dict[str, Any],
    model_path: Path,
    export_dir: Path,
) -> Tuple[pd.DataFrame, Path, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    cfs_out = cf_df.copy()
    cfs_out.insert(0, "cf_id", range(len(cfs_out)))
    cfs_out = cfs_out[["cf_id"] + FEATURES + [TARGET]]

    cfs_path = export_dir / "cfs.csv"
    cfs_out.to_csv(cfs_path, index=False)

    meta = {
        "surrogate_model_path": str(model_path),
        "splits_dir": "patchcore_splits",
        "target": TARGET,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "factual_index": int(query["factual"]["index"]),
        "factual_value_surrogate": float(query["factual"]["value_surrogate"]),
        "objective": query["objective"],
        "soft_constraints": query["soft_constraints"],
        "selection": query["selection"],
        "desired_range": desired_range,
        "permitted_range": permitted_range,
        "n_cfs": int(len(cfs_out)),
    }
    meta_path = export_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfs_out, cfs_path, meta_path


def to_py(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def objective_ok(score: float, objective: Dict[str, Any], factual_value: float) -> bool:
    t = objective.get("type")
    if t == "target_min":
        return float(score) >= float(objective.get("value", factual_value))
    if t == "delta_improve":
        return float(score) >= float(factual_value) + float(objective.get("delta", 0.0))
    if t == "target_band":
        lo = float(objective.get("lower", 0.0))
        hi = float(objective.get("upper", 1.0))
        return lo <= float(score) <= hi
    return False


def soft_ok(row: pd.Series, feat: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    mode = spec.get("mode", "free")
    val = row.get(feat)

    if mode == "free":
        return True, "free"
    if mode == "fixed":
        target = spec.get("value")
        if feat in CATEGORICAL:
            ok = str(val) == str(target)
        else:
            ok = abs(float(val) - float(target)) <= 1e-12
        return ok, f"fixed={target}"
    if mode == "range":
        lo = spec.get("lower", None)
        hi = spec.get("upper", None)
        if lo is None or hi is None:
            return False, "range missing bounds"
        v = float(val)
        ok = float(lo) <= v <= float(hi)
        return ok, f"range=[{float(lo)}, {float(hi)}]"
    if mode == "allowed":
        allowed = [str(a) for a in spec.get("allowed", [])]
        ok = str(val) in set(allowed)
        return ok, f"allowed={allowed}"
    return False, f"unknown mode={mode}"


def major_changes(row: pd.Series, factual_x: Dict[str, Any], max_items: int = 4) -> Tuple[List[str], int]:
    diffs: List[Tuple[float, str]] = []
    for feat in FEATURES:
        if feat not in row:
            continue
        cf_val = row[feat]
        fx_val = factual_x[feat]
        if feat in CATEGORICAL:
            if str(cf_val) != str(fx_val):
                diffs.append((1.0, f"{feat}: {fx_val} -> {cf_val}"))
        else:
            delta = float(cf_val) - float(fx_val)
            if abs(delta) > 1e-12:
                diffs.append((abs(delta), f"{feat}: {float(fx_val):.4g} -> {float(cf_val):.4g} ({delta:+.4g})"))
    diffs.sort(key=lambda x: x[0], reverse=True)
    return [txt for _, txt in diffs[:max_items]], len(diffs)


def build_alignment_records(cf_df: pd.DataFrame, query: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    if cf_df.empty:
        cols = ["cf_id", "score", "objective_match", "soft_match_rate", "violations", "changed_feature_count"]
        return [], pd.DataFrame(columns=cols)

    factual_x = query["factual"]["x"]
    factual_sur = float(query["factual"]["value_surrogate"])
    eval_source = cf_df.copy().reset_index(drop=True)
    eval_source.insert(0, "cf_id", range(len(eval_source)))

    records: List[Dict[str, Any]] = []
    for _, row in eval_source.iterrows():
        score = float(row[TARGET])
        obj_match = objective_ok(score, query["objective"], factual_sur)

        checks: Dict[str, Any] = {}
        violated: List[str] = []
        for feat, spec in query["soft_constraints"].items():
            ok, rule = soft_ok(row, feat, spec)
            checks[feat] = {"ok": bool(ok), "rule": rule, "value": to_py(row.get(feat))}
            if not ok:
                violated.append(feat)

        changes_top, n_changes = major_changes(row, factual_x=factual_x)
        n_soft = max(1, len(query["soft_constraints"]))
        soft_ok_count = sum(1 for v in checks.values() if v["ok"])

        records.append(
            {
                "cf_id": int(row["cf_id"]),
                "score": score,
                "objective_match": bool(obj_match),
                "soft_ok_count": int(soft_ok_count),
                "soft_total": int(n_soft),
                "soft_match_rate": float(soft_ok_count / n_soft),
                "violated_constraints": violated,
                "changed_feature_count": int(n_changes),
                "top_changes": changes_top,
                "checks": checks,
            }
        )

    eval_df = pd.DataFrame(
        [
            {
                "cf_id": r["cf_id"],
                "score": r["score"],
                "objective_match": r["objective_match"],
                "soft_match_rate": r["soft_match_rate"],
                "violations": ", ".join(r["violated_constraints"]) if r["violated_constraints"] else "none",
                "changed_feature_count": r["changed_feature_count"],
            }
            for r in records
        ]
    ).sort_values(["objective_match", "soft_match_rate", "score"], ascending=[False, False, False]).reset_index(drop=True)

    return records, eval_df


def _parse_json_obj(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", t, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        t = fence.group(1).strip()

    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(t[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON object found in model output.")


def _validate_llm_json(obj: Dict[str, Any], valid_ids: List[int]) -> Dict[str, Any]:
    valid_set = set(valid_ids)
    if not isinstance(obj, dict):
        raise ValueError("Root must be a JSON object.")

    per_cf = obj.get("per_cf")
    if not isinstance(per_cf, list):
        raise ValueError("`per_cf` must be a list.")

    seen: List[int] = []
    for item in per_cf:
        if not isinstance(item, dict):
            raise ValueError("Each `per_cf` item must be an object.")
        if "cf_id" not in item:
            raise ValueError("Each `per_cf` item must include `cf_id`.")
        cid = int(item["cf_id"])
        if cid not in valid_set:
            raise ValueError(f"Invalid cf_id in per_cf: {cid}")

        bullets = item.get("bullets")
        if not isinstance(bullets, list) or not (2 <= len(bullets) <= 3):
            raise ValueError(f"CF {cid}: `bullets` must be a list of length 2-3.")
        if any(not isinstance(b, str) or not b.strip() for b in bullets):
            raise ValueError(f"CF {cid}: each bullet must be a non-empty string.")
        seen.append(cid)

    if len(seen) != len(set(seen)):
        raise ValueError("Duplicate cf_id entries in per_cf.")
    if set(seen) != valid_set:
        raise ValueError(f"per_cf IDs mismatch. expected={sorted(valid_set)} got={sorted(set(seen))}")

    best = obj.get("best_choice")
    if not isinstance(best, dict):
        raise ValueError("`best_choice` must be an object.")
    if "cf_id" not in best:
        raise ValueError("`best_choice.cf_id` is required.")
    best_id = int(best["cf_id"])
    if best_id not in valid_set:
        raise ValueError(f"Invalid best_choice cf_id: {best_id}")

    runner = obj.get("runner_up")
    if not isinstance(runner, dict):
        raise ValueError("`runner_up` must be an object.")
    runner_id = runner.get("cf_id", None)

    if len(valid_ids) == 1:
        if runner_id is not None:
            raise ValueError("runner_up.cf_id must be null when only one CF exists.")
    else:
        if runner_id is None:
            raise ValueError("runner_up.cf_id must be non-null when >=2 CFs exist.")
        runner_id = int(runner_id)
        if runner_id not in valid_set:
            raise ValueError(f"Invalid runner_up cf_id: {runner_id}")
        if runner_id == best_id:
            raise ValueError("runner_up.cf_id must differ from best_choice.cf_id.")

    return obj


def llm_explain_json(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    payload: Dict[str, Any],
    valid_ids: List[int],
    max_retries: int = 8,
) -> Dict[str, Any]:
    base_prompt = (
        "You are helping a user choose among generated counterfactuals.\n"
        "Use only the provided structured data.\n"
        f"Valid CF IDs are exactly: {valid_ids}. Never mention any other ID.\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "per_cf": [\n'
        '    {"cf_id": <int>, "bullets": ["...", "...", "..."]}\n'
        "  ],\n"
        '  "best_choice": {"cf_id": <int>, "why": "<one concise paragraph>"},\n'
        '  "runner_up": {"cf_id": <int or null>, "why": "<one concise paragraph>"}\n'
        "}\n"
        "Rules:\n"
        "- Include every CF exactly once in per_cf.\n"
        "- Each CF must have 2-3 bullets.\n"
        "- If only one CF exists, runner_up.cf_id must be null.\n"
        "- Do not invent values.\n\n"
        f"DATA:\n{json.dumps(payload, indent=2)}\n\n"
        "JSON:\n"
    )

    last_err = ""
    last_raw = ""
    for _attempt in range(1, max_retries + 1):
        prompt = base_prompt
        if last_err:
            prompt += f"\nPrevious output failed validation: {last_err}\nRegenerate valid JSON only.\n"
            if last_raw:
                prompt += f"\nInvalid output was:\n{last_raw[:1200]}\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(_model_device(model))
        with torch.inference_mode():
            out = generate_greedy(tokenizer, model, inputs, max_new_tokens=700)

        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        last_raw = raw

        try:
            obj = _parse_json_obj(raw)
            return _validate_llm_json(obj, valid_ids)
        except Exception as exc:
            last_err = str(exc)

    raise RuntimeError(
        f"LLM failed to produce valid explanation JSON after {max_retries} attempts. Last error: {last_err}"
    )


def format_explanation(resp: Dict[str, Any], valid_ids: List[int]) -> str:
    order = {cid: i for i, cid in enumerate(valid_ids)}
    per_cf_sorted = sorted(resp["per_cf"], key=lambda x: order[int(x["cf_id"])])

    lines: List[str] = []
    for item in per_cf_sorted:
        cid = int(item["cf_id"])
        lines.append(f"CF {cid}:")
        for b in item["bullets"][:3]:
            lines.append(f"- {str(b).strip()}")

    best_id = int(resp["best_choice"]["cf_id"])
    lines.append(f"Best choice: CF {best_id}. {str(resp['best_choice'].get('why', '')).strip()}")

    runner_id = resp.get("runner_up", {}).get("cf_id", None)
    runner_why = str(resp.get("runner_up", {}).get("why", "")).strip()
    if runner_id is None:
        lines.append(f"Runner-up: none. {runner_why}")
    else:
        lines.append(f"Runner-up: CF {int(runner_id)}. {runner_why}")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


@st.cache_resource(show_spinner=False)
def load_surrogate(model_path: str) -> CatBoostRegressor:
    cb = CatBoostRegressor()
    cb.load_model(model_path)
    return cb


@st.cache_resource(show_spinner=False)
def load_llm(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.to("cpu")
    model.eval()
    return tokenizer, model


def build_factual_candidates(
    df: pd.DataFrame, cb: CatBoostRegressor, size: int = 3, seed: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    size = max(1, min(int(size), len(df)))
    rng = np.random.default_rng(int(seed))
    idxs = rng.choice(df.index.to_numpy(), size=size, replace=False)

    preview_rows: List[Dict[str, Any]] = []
    for j, idx in enumerate(idxs):
        x = df.loc[idx, FEATURES].to_dict()
        y = float(cb.predict(pd.DataFrame([x]))[0])
        row: Dict[str, Any] = {"Candidate": j, "Predicted quality (F1 score)": y}
        for f in FEATURES[:6]:
            row[f] = x[f]
        preview_rows.append(row)

    factual_candidates = pd.DataFrame(preview_rows)
    return idxs, factual_candidates


def run_pipeline(
    df: pd.DataFrame,
    cb: CatBoostRegressor,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    selected_idx: int,
    user_text: str,
    export_dir: Path,
    model_path: Path,
    explain_retries: int,
) -> Dict[str, Any]:
    x_factual = df.loc[selected_idx, FEATURES].to_dict()
    x_factual_df = pd.DataFrame([x_factual])
    y_factual_sur = float(cb.predict(x_factual_df)[0])

    core = build_query_core(
        user_text=user_text,
        x_factual=x_factual,
        y_factual_sur=y_factual_sur,
        llama_json_fn=lambda prompt: llama_generate_json(tokenizer, model, prompt),
    )

    query_draft = {
        "factual": {"index": selected_idx, "x": dict(x_factual), "value_surrogate": y_factual_sur},
        "objective": core["objective"],
        "soft_constraints": core["soft_constraints"],
        "selection": core["selection"],
        "_meta": core.get("_meta", {}),
    }
    query_final = repair_and_snap(query_draft, x_factual=x_factual, y_factual_sur=y_factual_sur)
    query_final["_meta"] = core.get("_meta", {})
    query_final["_user_request_text"] = user_text.strip()

    desired_range = objective_to_desired_range(query_final)
    cf_df, permitted_range, pool, before_filter_count, after_range_filter_count = generate_cf_table(
        df=df, cb=cb, query=query_final, desired_range=desired_range
    )

    cfs_out, cfs_path, meta_path = export_artifacts(
        cf_df=cf_df,
        query=query_final,
        desired_range=desired_range,
        permitted_range=permitted_range,
        model_path=model_path,
        export_dir=export_dir,
    )

    explanation = "No counterfactuals available to explain."
    explain_path = None
    
    if not cf_df.empty:
        # Build minimal "records" from cf_df for the explainer
        cf_for_llm = cf_df.copy().reset_index(drop=True)
        cf_for_llm.insert(0, "cf_id", range(len(cf_for_llm)))
    
        # minimal per-cf structure (only what's needed downstream)
        records = []
        factual_x = query_final["factual"]["x"]
        for _, row in cf_for_llm.iterrows():
            changed = []
            for feat in FEATURES:
                if feat in row and feat in factual_x and str(row[feat]) != str(factual_x[feat]):
                    changed.append(f"{feat}: {factual_x[feat]} -> {row[feat]}")
            records.append(
                {
                    "cf_id": int(row["cf_id"]),
                    "score": float(row[TARGET]),
                    "top_changes": changed[:4],
                    "changed_feature_count": int(len(changed)),
                    "violated_constraints": [],   # no alignment checks anymore
                }
            )
    
        payload = {
            "user_request_text": query_final.get("_user_request_text", ""),
            "parsed_query": {
                "objective": query_final["objective"],
                "soft_constraints": query_final["soft_constraints"],
                "selection": query_final["selection"],
            },
            "factual": {
                "index": int(query_final["factual"]["index"]),
                "value_surrogate": float(query_final["factual"]["value_surrogate"]),
            },
            "counterfactuals": records,
        }
        valid_ids = [int(r["cf_id"]) for r in records]
        resp = llm_explain_json(tokenizer, model, payload, valid_ids, max_retries=int(explain_retries))
        explanation = format_explanation(resp, valid_ids)
        explain_path = export_dir / "cf_explanations.txt"
        explain_path.write_text(explanation + "\n", encoding="utf-8")

    return {
        "query": query_final,
        "desired_range": desired_range,
        "permitted_range": permitted_range,
        "pool": pool,
        "before_filter_count": before_filter_count,
        "after_range_filter_count": after_range_filter_count,
        "cf_df": cf_df,
        "cfs_out": cfs_out,
        "explanation": explanation,
        "cfs_path": cfs_path,
        "meta_path": meta_path,
        "explain_path": explain_path,
    }


def render_result(result: Dict[str, Any]):
    # --- Query (final): hidden by default ---
    with st.expander("Query (final)", expanded=False):
        st.json(result["query"])

    # --- Generation summary: hidden by default ---
    with st.expander("Generation summary", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Pool requested from DiCE", int(result["pool"]))
        c2.metric("CFs before desired_range filter", int(result["before_filter_count"]))
        c3.metric("CFs after desired_range filter", int(result["after_range_filter_count"]))

        st.write("`desired_range`:", result["desired_range"])
        st.write("`permitted_range`:")
        st.json(result["permitted_range"])

    # --- Always visible (main outputs) ---
    st.subheader("Final CF table (top-k)")
    st.dataframe(rename_df_columns_for_ui(result["cf_df"]), use_container_width=True)
    st.caption(f"Kept {len(result['cf_df'])} CFs (requested {int(result['query']['selection']['k'])}).")

    st.subheader("LLM explanation")
    st.text(result["explanation"])

    st.subheader("Saved artifacts")
    st.write(str(result["cfs_path"]))
    st.write(str(result["meta_path"]))
    if result["explain_path"] is not None:
        st.write(str(result["explain_path"]))

    if result["cfs_path"].exists():
        st.download_button(
            "Download cfs.csv",
            data=result["cfs_path"].read_bytes(),
            file_name="cfs.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if result["meta_path"].exists():
        st.download_button(
            "Download metadata.json",
            data=result["meta_path"].read_bytes(),
            file_name="metadata.json",
            mime="application/json",
            use_container_width=True,
        )
    if result["explain_path"] is not None and result["explain_path"].exists():
        st.download_button(
            "Download cf_explanations.txt",
            data=result["explain_path"].read_bytes(),
            file_name="cf_explanations.txt",
            mime="text/plain",
            use_container_width=True,
        )
        

def main():
    st.set_page_config(page_title="Counterfactual Explorer", layout="wide")
    st.title("Counterfactual Explorer")
    st.caption("Streamlit server converted from `master.ipynb`.")

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("LLM model", LLM_MODEL_OPTIONS + ["Custom model id"]) 
        if model_choice == "Custom model id":
            llm_model_id = st.text_input("Custom LLM model id", DEFAULT_LLM_MODEL_ID)
        else:
            llm_model_id = model_choice
        st.header("Run settings")
        candidate_count = st.slider("Factual candidates", min_value=1, max_value=20, value=3, step=1)
        candidate_seed = st.number_input("Candidate seed", value=42, step=1)
        explain_retries = st.slider("Explanation max retries", min_value=1, max_value=12, value=6, step=1)

    try:
        with st.spinner("Loading dataset + surrogate model..."):
            df = load_dataframe(DEFAULT_CSV_PATH)
            cb = load_surrogate(DEFAULT_MODEL_PATH)
    except Exception as exc:
        st.error("Failed to load dataset/model. Check paths in the sidebar.")
        st.exception(exc)
        st.stop()

    if df.empty:
        st.error("Loaded dataset is empty.")
        st.stop()

    idxs, factual_candidates = build_factual_candidates(df, cb, size=candidate_count, seed=int(candidate_seed))
    st.subheader("Factual candidates")
    st.dataframe(rename_df_columns_for_ui(factual_candidates), use_container_width=True)
    options: Dict[str, int] = {}
    for _, row in factual_candidates.iterrows():
        label = (
            f"Candidate {int(row['Candidate'])} | "
            f"Predicted quality (F1 score)={float(row['Predicted quality (F1 score)']):.4f}"
        )
        options[label] = int(row["Candidate"])

    selected_label = st.selectbox("Factual", list(options.keys()), index=0)
    user_text = st.text_area("Preferences", value=DEFAULT_USER_TEXT, height=180)
    run_clicked = st.button("Run pipeline", type="primary")

    if run_clicked:
        try:
            with st.spinner("Loading LLM..."):
                tokenizer, model = load_llm(llm_model_id)

            selected_pick = options[selected_label]
            selected_idx = int(idxs[selected_pick])

            with st.spinner("Running query -> DiCE -> explanation..."):
                result = run_pipeline(
                    df=df,
                    cb=cb,
                    tokenizer=tokenizer,
                    model=model,
                    selected_idx=selected_idx,
                    user_text=user_text,
                    export_dir=Path(DEFAULT_EXPORT_DIR),
                    model_path=Path(DEFAULT_MODEL_PATH),
                    explain_retries=int(explain_retries),
                )
            st.session_state["cf_result"] = result
            st.success("Pipeline completed.")
        except Exception as exc:
            st.exception(exc)

    if "cf_result" in st.session_state:
        render_result(st.session_state["cf_result"])


if __name__ == "__main__":
    main()
