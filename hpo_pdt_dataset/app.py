"""
LLM Interface for CF-HPO  –  clean rewrite
==========================================
Changes vs original
-------------------
* Single normalization pass: LLM JSON → schema repair → snap → explicit-preference overlay
* Rule-based explanation fixed:
  - Tied scores reported correctly (no spurious "rank 2 of 3" when scores are equal)
  - Universal violations (shared by every option) called out once at the top, not repeated per option
  - Best/runner-up reasoning omits redundant re-statement of universal violations
* Dead code removed: commented-out format_explanation, unused llm_explain_json UI path
* Anchor diversity loop is guarded against pool exhaustion
* query_template no longer needs a "factual" key for objective_to_desired_range
* LLM query parsing kept but simplified: one repair pass, one validation pass
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import dice_ml
import numpy as np
import pandas as pd
import streamlit as st
import torch
from catboost import CatBoostRegressor
from dice_ml import Dice
from jsonschema import validate
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATH   = Path("surrogate_ready_dataset/patchcore_surrogate_dataset_xgb.csv")
DEFAULT_MODEL_PATH = Path("surrogate_pkl_cfs_metadata/surrogate_catboost.cbm")
DEFAULT_EXPORT_DIR = Path("surrogate_pkl_cfs_metadata")
DEFAULT_LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

LLM_MODEL_OPTIONS = [
    "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
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
SOFT_CORRUPTION_LEVELS    = ["none", "mild", "strong"]
SOFT_REVIEW_BUDGETS       = [round(i / 1000, 3) for i in range(5, 501, 5)]
SOFT_REVIEW_BUDGET_BOUNDS = (SOFT_REVIEW_BUDGETS[0], SOFT_REVIEW_BUDGETS[-1])

DEFAULT_EPS          = 0.01
DEFAULT_K            = 5
MAX_LLM_RETRIES      = 3
DEFAULT_ANCHOR_COUNT = 3

FEATURE_LABELS: dict[str, str] = {
    "params_backbone":              "Backbone",
    "params_batch_size":            "Batch size",
    "params_center_crop_key":       "Center crop",
    "params_coreset_sampling_ratio":"Coreset sampling ratio",
    "params_image_size_key":        "Image size",
    "params_layers_key":            "Layer set",
    "params_max_patches_per_image": "Max patches per image",
    "params_num_neighbors":         "Nearest neighbors",
    "params_pre_trained":           "Use pretrained weights",
    "params_reduction":             "Reduction method",
    "params_soft_corruption_level": "Corruption level",
    "params_soft_review_budget":    "Review budget",
    "params_soft_train_fraction":   "Training fraction",
    TARGET:                         "Quality (F1 score)",
}
TARGET_LABEL = FEATURE_LABELS[TARGET]

DEFAULT_USER_TEXT = (
    "I want high quality with a reasonable review budget.\n"
    "Quality should be at least 0.85.\n"
    "Keep train fraction between 0.30 and 0.60.\n"
    "Allow corruption level to vary.\n"
    "Return 5 options with a balanced strategy."
)

EXAMPLE_REQUESTS = [
    (
        "High quality, moderate budget",
        "I want high quality with a reasonable review budget.\nQuality should be at minimum 0.87.\nReturn 5 options.",
    ),
    (
        "Constrained train fraction",
        "Minimum quality should be 0.85.\nKeep train fraction between 0.40 and 0.60.\nReturn 5 options.",
    ),
    (
        "Broader exploration",
        "Keep quality between 0.70 and 1.\nAllow review budget to vary.\nAllow corruption level to vary.\nReturn 8 options.",
    ),
    (
        "Safe target band",
        "Keep quality between 0.80 and 0.92.\nKeep train fraction between 0.40 and 0.70.\nReturn 6 options.",
    ),
]

# ---------------------------------------------------------------------------
# JSON schema for the parsed query
# ---------------------------------------------------------------------------

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
                "type":  {"type": "string", "enum": ["target_min", "delta_improve", "target_band"]},
                "value": {"type": "number"},
                "delta": {"type": "number"},
                "lower": {"type": "number"},
                "upper": {"type": "number"},
                "eps":   {"type": "number"},
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
                    "type": "object", "additionalProperties": False,
                    "required": ["mode"],
                    "properties": {
                        "mode":  {"type": "string", "enum": ["free", "fixed", "range"]},
                        "value": {"type": "number"},
                        "lower": {"type": "number"},
                        "upper": {"type": "number"},
                    },
                },
                "params_soft_review_budget": {
                    "type": "object", "additionalProperties": False,
                    "required": ["mode"],
                    "properties": {
                        "mode":  {"type": "string", "enum": ["free", "fixed", "range"]},
                        "value": {"type": "number"},
                        "lower": {"type": "number"},
                        "upper": {"type": "number"},
                    },
                },
                "params_soft_corruption_level": {
                    "type": "object", "additionalProperties": False,
                    "required": ["mode"],
                    "properties": {
                        "mode":    {"type": "string", "enum": ["free", "fixed", "allowed"]},
                        "value":   {"type": "string", "enum": SOFT_CORRUPTION_LEVELS},
                        "allowed": {
                            "type": "array", "minItems": 1,
                            "items": {"type": "string", "enum": SOFT_CORRUPTION_LEVELS},
                        },
                    },
                },
            },
        },
        "selection": {
            "type": "object", "additionalProperties": False,
            "required": ["k"],
            "properties": {"k": {"type": "integer", "minimum": 1, "maximum": 50}},
        },
    },
}

LLM_CONTRACT = f"""
Return ONLY valid JSON. No markdown, no extra text.

Required keys: objective, soft_constraints, selection.

objective.type in: target_min | delta_improve | target_band
  target_min   → "value": <float>
  delta_improve → "delta": <float>
  target_band  → "lower": <float>, "upper": <float>
All objective types accept optional "eps": <float> (default 0.01).

soft_constraints must contain exactly:
  params_soft_train_fraction   → mode: free|fixed|range
  params_soft_review_budget    → mode: free|fixed|range
  params_soft_corruption_level → mode: free|fixed|allowed

selection: {{ "k": <int 1-50> }}

Constraints:
  if target_band: lower <= upper
  if range: lower <= upper
  Review budget values are multiples of 0.005 in [{SOFT_REVIEW_BUDGET_BOUNDS[0]}, {SOFT_REVIEW_BUDGET_BOUNDS[1]}]
  Train fraction in {SOFT_TRAIN_FRACTION_RANGE}
  Allowed corruption values: {SOFT_CORRUPTION_LEVELS}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def feat_label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat.replace("params_", "").replace("_", " ").title())


def option_name(cf_id: int) -> str:
    cid = int(cf_id)
    return f"Option {chr(ord('A') + cid)}" if 0 <= cid < 26 else f"Option {cid + 1}"


def snap_budget(x: float) -> float:
    return min(SOFT_REVIEW_BUDGETS, key=lambda v: abs(v - float(x)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def snap01(x: float, lo: float, hi: float) -> float:
    return round(clamp(x, lo, hi) * 100) / 100


def fmt3(x: Any) -> str:
    return f"{float(x):.3f}"


def human_join(items: list[str]) -> str:
    items = [s.strip() for s in items if s.strip()]
    if not items:       return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def feat_val_text(feat: str, value: Any) -> str:
    if value is None: return "-"
    if feat in CATEGORICAL: return str(value)
    try:
        v = float(value)
    except Exception:
        return str(value)
    if feat in {"params_soft_train_fraction", "params_soft_review_budget", TARGET}:
        return f"{v:.3f}"
    return f"{v:.4g}"


def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    return df.rename(columns=lambda c: FEATURE_LABELS.get(c, c))


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _model_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "device"): return model.device
    return next(model.parameters()).device


def _extract_first_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            esc = (not esc and ch == "\\")
            if not esc and ch == '"': in_str = False
            continue
        if ch == '"':   in_str = True
        elif ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0: return text[start:i + 1]
    raise ValueError("Unbalanced JSON braces in model output.")


@torch.inference_mode()
def llm_generate_json(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    user_text: str,
    max_new_tokens: int = 400,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "Output ONLY valid JSON. No extra text."},
            {"role": "user",   "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"SYSTEM: Output ONLY valid JSON.\nUSER:\n{user_text}\nASSISTANT:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(_model_device(model))
    gen_cfg = GenerationConfig(
        do_sample=False, temperature=None, top_p=None, top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out     = model.generate(**inputs, generation_config=gen_cfg)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _extract_first_json(decoded)


# ---------------------------------------------------------------------------
# Query parsing: LLM + regex overlay + repair/snap  (single pass each)
# ---------------------------------------------------------------------------

def _repair_objective(obj: dict, y_factual: float) -> dict:
    t = obj.get("type", "delta_improve")
    eps = float(obj.get("eps", DEFAULT_EPS))
    if t == "target_min":
        return {"type": "target_min",   "value": float(obj.get("value", y_factual)), "eps": eps}
    if t == "delta_improve":
        return {"type": "delta_improve", "delta": float(obj.get("delta", 0.02)),      "eps": eps}
    if t == "target_band":
        lo = float(obj.get("lower", y_factual))
        hi = float(obj.get("upper", y_factual))
        if lo > hi: lo, hi = hi, lo
        return {"type": "target_band", "lower": lo, "upper": hi, "eps": eps}
    # unknown type → safe fallback
    return {"type": "target_band", "lower": y_factual, "upper": y_factual, "eps": DEFAULT_EPS}


def _repair_soft_constraints(sc: dict, x_factual: dict) -> dict:
    out: dict = {}

    # --- review budget ---
    rb = sc.get("params_soft_review_budget", {"mode": "free"})
    mode = rb.get("mode", "free")
    if mode == "fixed":
        v = snap_budget(clamp(float(rb.get("value", x_factual.get("params_soft_review_budget", 0.1))),
                              *SOFT_REVIEW_BUDGET_BOUNDS))
        out["params_soft_review_budget"] = {"mode": "fixed", "value": v}
    elif mode == "range":
        lo = rb.get("lower"); hi = rb.get("upper")
        if lo is None or hi is None:
            out["params_soft_review_budget"] = {"mode": "free"}
        else:
            lo, hi = sorted([float(lo), float(hi)])
            lo = snap_budget(clamp(lo, *SOFT_REVIEW_BUDGET_BOUNDS))
            hi = snap_budget(clamp(hi, *SOFT_REVIEW_BUDGET_BOUNDS))
            out["params_soft_review_budget"] = {"mode": "range", "lower": lo, "upper": hi}
    else:
        out["params_soft_review_budget"] = {"mode": "free"}

    # --- train fraction ---
    tf = sc.get("params_soft_train_fraction", {"mode": "free"})
    mode = tf.get("mode", "free")
    lo_tf, hi_tf = SOFT_TRAIN_FRACTION_RANGE
    if mode == "fixed":
        v = snap01(float(tf.get("value", x_factual.get("params_soft_train_fraction", 0.5))), lo_tf, hi_tf)
        out["params_soft_train_fraction"] = {"mode": "fixed", "value": v}
    elif mode == "range":
        lo = tf.get("lower"); hi = tf.get("upper")
        if lo is None or hi is None:
            out["params_soft_train_fraction"] = {"mode": "free"}
        else:
            lo, hi = sorted([snap01(float(lo), lo_tf, hi_tf), snap01(float(hi), lo_tf, hi_tf)])
            out["params_soft_train_fraction"] = {"mode": "range", "lower": lo, "upper": hi}
    else:
        out["params_soft_train_fraction"] = {"mode": "free"}

    # --- corruption level ---
    cl   = sc.get("params_soft_corruption_level", {"mode": "free"})
    mode = cl.get("mode", "free")
    allowed_set = set(SOFT_CORRUPTION_LEVELS)
    if mode == "fixed":
        v = str(cl.get("value", x_factual.get("params_soft_corruption_level", "none")))
        if v not in allowed_set: v = "none"
        out["params_soft_corruption_level"] = {"mode": "fixed", "value": v}
    elif mode == "allowed":
        vals = sorted({str(a) for a in cl.get("allowed", []) if str(a) in allowed_set})
        if vals:
            out["params_soft_corruption_level"] = {"mode": "allowed", "allowed": vals}
        else:
            out["params_soft_corruption_level"] = {"mode": "free"}
    else:
        out["params_soft_corruption_level"] = {"mode": "free"}

    return out


def _regex_overlay(query: dict, user_text: str) -> dict:
    """Extract explicit numeric constraints from user text and override LLM interpretation."""
    q   = copy.deepcopy(query)
    txt = user_text.lower()

    # Objective
    m = re.search(
        r"(?:min(?:imum)?\s*(?:quality|performance)?\s*(?:should\s*be|is|:)?|"
        r"(?:quality|performance)\s*(?:should\s*be|is)?\s*at\s*least)\s*([0-9]*\.?[0-9]+)",
        txt,
    )
    if m:
        q["objective"] = {"type": "target_min", "value": float(m.group(1)), "eps": DEFAULT_EPS}

    m_band = re.search(
        r"(?:keep\s*quality|quality)\s*between\s*([0-9]*\.?[0-9]+)\s*(?:and|to)\s*([0-9]*\.?[0-9]+)",
        txt,
    )
    if m_band and not m:
        lo, hi = sorted([float(m_band.group(1)), float(m_band.group(2))])
        q["objective"] = {"type": "target_band", "lower": lo, "upper": hi, "eps": DEFAULT_EPS}

    # Train fraction
    for line in re.split(r"[\r\n;]+", txt):
        if "train fraction" in line or "training fraction" in line:
            mr = re.search(r"(?:between|from)\s*([0-9]*\.?[0-9]+)\s*(?:and|to|-)\s*([0-9]*\.?[0-9]+)", line)
            if mr:
                lo, hi = sorted([float(mr.group(1)), float(mr.group(2))])
                q["soft_constraints"]["params_soft_train_fraction"] = {"mode": "range", "lower": lo, "upper": hi}
            else:
                mf = re.search(r"([0-9]*\.?[0-9]+)", line)
                if mf:
                    q["soft_constraints"]["params_soft_train_fraction"] = {"mode": "fixed", "value": float(mf.group(1))}

        if "review budget" in line:
            mr = re.search(r"(?:between|from)\s*([0-9]*\.?[0-9]+)\s*(?:and|to|-)\s*([0-9]*\.?[0-9]+)", line)
            if mr:
                lo, hi = sorted([float(mr.group(1)), float(mr.group(2))])
                q["soft_constraints"]["params_soft_review_budget"] = {"mode": "range", "lower": lo, "upper": hi}
            elif any(w in line for w in ["free", "any", "vary", "variable"]):
                q["soft_constraints"]["params_soft_review_budget"] = {"mode": "free"}

        if "corruption" in line:
            if any(w in line for w in ["free", "any", "vary", "variable"]):
                q["soft_constraints"]["params_soft_corruption_level"] = {"mode": "free"}
            else:
                vals = sorted({v for v in SOFT_CORRUPTION_LEVELS if re.search(rf"\b{re.escape(v)}\b", line)})
                if len(vals) == 1:
                    q["soft_constraints"]["params_soft_corruption_level"] = {"mode": "fixed",   "value": vals[0]}
                elif len(vals) > 1:
                    q["soft_constraints"]["params_soft_corruption_level"] = {"mode": "allowed", "allowed": vals}

        mk = re.search(r"\b(\d+)\s*(?:options?|alternatives?|cfs?|counterfactuals?)\b", line)
        if mk:
            q["selection"]["k"] = int(mk.group(1))

    return q


def _fallback_query(y_factual: float, k: int = DEFAULT_K) -> dict:
    return {
        "objective": {"type": "delta_improve", "delta": 0.02, "eps": DEFAULT_EPS},
        "soft_constraints": {
            "params_soft_train_fraction":   {"mode": "free"},
            "params_soft_review_budget":    {"mode": "free"},
            "params_soft_corruption_level": {"mode": "free"},
        },
        "selection": {"k": k},
    }


def build_query(
    user_text: str,
    x_factual: dict,
    y_factual: float,
    llm_fn,                 # (prompt: str) -> str
    max_retries: int = MAX_LLM_RETRIES,
) -> dict:
    """
    Parse user intent into a structured query dict.
    Pipeline: LLM → JSON parse → repair/snap → regex overlay → repair/snap → validate
    Falls back to a safe default if all LLM attempts fail.
    """
    last_err: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            raw   = llm_fn(LLM_CONTRACT + "\n\nUSER REQUEST:\n" + user_text.strip())
            draft = json.loads(raw)

            # Single repair pass
            q: dict = {
                "objective":        _repair_objective(draft.get("objective", {}), y_factual),
                "soft_constraints": _repair_soft_constraints(draft.get("soft_constraints", {}), x_factual),
                "selection":        {"k": max(1, min(50, int(draft.get("selection", {}).get("k", DEFAULT_K))))},
            }

            # Regex overlay (honours explicit user numbers over LLM interpretation)
            q = _regex_overlay(q, user_text)

            # Second repair to snap anything the overlay may have put out-of-range
            q["soft_constraints"] = _repair_soft_constraints(q["soft_constraints"], x_factual)
            q["selection"]["k"]   = max(1, min(50, int(q["selection"]["k"])))

            validate(instance=q, schema=QUERY_SCHEMA)
            q["_meta"] = {"used_fallback": False, "attempts": attempt + 1}
            return q

        except Exception as exc:
            last_err = exc

    # All attempts failed → rule-based fallback
    q = _fallback_query(y_factual)
    q = _regex_overlay(q, user_text)
    q["soft_constraints"] = _repair_soft_constraints(q["soft_constraints"], x_factual)
    q["selection"]["k"]   = max(1, min(50, int(q["selection"]["k"])))
    q["_meta"] = {"used_fallback": True, "reason": str(last_err)[:250] if last_err else "unknown"}
    return q


# ---------------------------------------------------------------------------
# Desired range from objective
# ---------------------------------------------------------------------------

def objective_to_desired_range(objective: dict, y_factual: float) -> list[float]:
    t = objective["type"]
    if t == "target_min":
        lo, hi = float(objective["value"]), 1.0
    elif t == "delta_improve":
        lo, hi = y_factual + float(objective["delta"]), 1.0
    elif t == "target_band":
        lo, hi = float(objective["lower"]), float(objective["upper"])
    else:
        raise ValueError(f"Unknown objective type: {t}")
    lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
    if lo > hi: lo, hi = hi, lo
    return [float(lo), float(hi)]


# ---------------------------------------------------------------------------
# Counterfactual generation
# ---------------------------------------------------------------------------

def _domain(df: pd.DataFrame, feat: str):
    if feat in CATEGORICAL:
        return sorted(df[feat].dropna().astype(str).unique().tolist())
    col = pd.to_numeric(df[feat], errors="coerce")
    return [float(np.nanmin(col)), float(np.nanmax(col))]


def generate_cf_table(
    df: pd.DataFrame,
    cb: CatBoostRegressor,
    x_factual: dict,
    query: dict,
    y_factual: float,
) -> tuple[pd.DataFrame, int, int]:
    """Return (cf_df, before_range_filter_count, after_range_filter_count)."""
    df_dice     = df[FEATURES + [TARGET]].copy()
    continuous  = [c for c in FEATURES if c not in CATEGORICAL]
    data_dice   = dice_ml.Data(dataframe=df_dice, continuous_features=continuous, outcome_name=TARGET)
    model_dice  = dice_ml.Model(model=cb, backend="sklearn", model_type="regressor")
    exp_dice    = Dice(data_dice, model_dice, method="random")
    x0          = pd.DataFrame([x_factual])[FEATURES]
    desired_rng = objective_to_desired_range(query["objective"], y_factual)

    # Build permitted_range from soft_constraints
    sc = query["soft_constraints"]
    permitted: dict = {}
    for feat in FEATURES:
        dom = _domain(df_dice, feat)
        if feat == "params_soft_review_budget":
            spec = sc.get("params_soft_review_budget", {"mode": "free"})
        elif feat == "params_soft_train_fraction":
            spec = sc.get("params_soft_train_fraction", {"mode": "free"})
        elif feat == "params_soft_corruption_level":
            spec = sc.get("params_soft_corruption_level", {"mode": "free"})
        else:
            permitted[feat] = dom
            continue

        mode = spec.get("mode", "free")
        if mode == "free":
            permitted[feat] = dom
        elif mode == "fixed":
            v = spec.get("value")
            if feat in CATEGORICAL:
                permitted[feat] = [str(v)] if v is not None else dom
            else:
                v = float(v) if v is not None else float(x_factual.get(feat, dom[0]))
                permitted[feat] = [v, v]
        elif mode == "range":
            lo = float(spec.get("lower", dom[0]))
            hi = float(spec.get("upper", dom[1] if isinstance(dom[1], float) else dom[0]))
            lo, hi = max(dom[0], lo), min(dom[1], hi)
            permitted[feat] = [lo, hi] if lo <= hi else dom
        elif mode == "allowed":
            vals = [str(a) for a in spec.get("allowed", []) if str(a) in set(map(str, dom))]
            permitted[feat] = sorted(set(vals)) if vals else dom
        else:
            permitted[feat] = dom

    k        = int(query["selection"]["k"])
    base_pool = max(20, 5 * k)

    lo0, hi0 = desired_rng
    attempts = [
        (desired_rng,   permitted,                                                     base_pool),
        ([max(0.0, lo0 - 0.02), hi0], permitted,                                       max(base_pool, 60)),
        ([0.0, 1.0],   {f: _domain(df_dice, f) for f in FEATURES},                    max(base_pool, 120)),
    ]

    for rng, perm, pool in attempts:
        try:
            dice_res = exp_dice.generate_counterfactuals(
                x0, total_CFs=pool, desired_range=rng,
                features_to_vary=FEATURES, permitted_range=perm,
            )
        except Exception as exc:
            if "no counterfactuals found" in str(exc).lower():
                continue
            raise

        ex     = dice_res.cf_examples_list[0]
        cf_raw = (ex.final_cfs_df_sparse if hasattr(ex, "final_cfs_df_sparse") else ex.final_cfs_df).copy()
        cf_df  = cf_raw[FEATURES + [TARGET]].copy()
        for c in CATEGORICAL:
            if c in cf_df.columns:
                cf_df[c] = cf_df[c].astype(str)

        before = len(cf_df)
        lo, hi = rng
        cf_df  = cf_df[(cf_df[TARGET] >= lo) & (cf_df[TARGET] <= hi)].copy()
        after  = len(cf_df)
        if after == 0:
            continue

        cf_df = cf_df.sort_values(TARGET, ascending=False).reset_index(drop=True).head(k)
        return cf_df, before, after

    return pd.DataFrame(columns=FEATURES + [TARGET]), 0, 0


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------

def _norm_numeric(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in FEATURES if c not in CATEGORICAL]
    x    = df[cols].copy()
    for c in cols:
        vals = pd.to_numeric(x[c], errors="coerce")
        lo, hi = float(vals.min()), float(vals.max())
        x[c] = (vals - lo) / (hi - lo) if hi > lo else 0.0
    return x.fillna(0.0).to_numpy(dtype=float)


def select_anchors(df: pd.DataFrame, cb: CatBoostRegressor, n: int = DEFAULT_ANCHOR_COUNT) -> pd.DataFrame:
    scored = df.copy()
    scored["_pred"] = cb.predict(df[FEATURES])
    scored = scored.sort_values("_pred", ascending=False).reset_index().rename(columns={"index": "_row_idx"})

    n = max(1, int(n))
    if len(scored) <= n:
        return scored.head(n).copy()

    chosen: list[int] = [int(scored.iloc[0]["_row_idx"])]

    # Anchor 2: balanced (near median budget, decent quality)
    budget_col  = "params_soft_review_budget"
    quality_cut = float(scored["_pred"].quantile(0.60))
    mid_budget  = float(scored[budget_col].astype(float).median())
    pool2 = scored[scored["_pred"] >= quality_cut].copy()
    pool2["_bd"] = (pool2[budget_col].astype(float) - mid_budget).abs()
    pool2 = pool2.sort_values(["_bd", "_pred"], ascending=[True, False])
    for _, r in pool2.iterrows():
        ridx = int(r["_row_idx"])
        if ridx not in chosen:
            chosen.append(ridx)
            break

    # Remaining: max-distance diversity in numeric feature space
    cand_pool = scored.head(min(max(100, 20 * n), len(scored))).copy()
    x_num     = _norm_numeric(cand_pool)
    chosen_set = set(chosen)

    while len(chosen) < n:
        mask         = cand_pool["_row_idx"].isin(chosen_set).to_numpy()
        chosen_pos   = np.where(mask)[0]
        if len(chosen_pos) == 0:
            break

        scores = []
        for i in range(len(cand_pool)):
            if mask[i]:
                scores.append(-1.0)
                continue
            d = float(np.min(np.linalg.norm(x_num[i] - x_num[chosen_pos], axis=1)))
            scores.append(d + 0.15 * float(cand_pool.iloc[i]["_pred"]))

        best_i = int(np.argmax(scores))
        ridx   = int(cand_pool.iloc[best_i]["_row_idx"])
        if ridx not in chosen_set:
            chosen.append(ridx)
            chosen_set.add(ridx)
        else:
            # safety: pick next unchosen row
            found = False
            for ridx2 in cand_pool["_row_idx"].astype(int).tolist():
                if ridx2 not in chosen_set:
                    chosen.append(ridx2)
                    chosen_set.add(ridx2)
                    found = True
                    break
            if not found:
                break

    anchors = scored[scored["_row_idx"].isin(chosen_set)].copy()
    order   = {r: i for i, r in enumerate(chosen)}
    anchors["_anchor_order"] = anchors["_row_idx"].map(order)
    return anchors.sort_values("_anchor_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deduplication + candidate assembly
# ---------------------------------------------------------------------------

def _signature(row: pd.Series) -> str:
    parts = []
    for feat in FEATURES:
        v = row.get(feat)
        parts.append(f"{feat}={str(v) if feat in CATEGORICAL else f'{float(v):.6f}'}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Soft-constraint compliance check
# ---------------------------------------------------------------------------

def _soft_ok(row: pd.Series, feat: str, spec: dict) -> bool:
    mode = spec.get("mode", "free")
    val  = row.get(feat)
    if mode == "free":    return True
    if mode == "fixed":
        t = spec.get("value")
        return str(val) == str(t) if feat in CATEGORICAL else abs(float(val) - float(t)) <= 1e-9
    if mode == "range":
        lo, hi = spec.get("lower"), spec.get("upper")
        return lo is not None and hi is not None and float(lo) <= float(val) <= float(hi)
    if mode == "allowed":
        return str(val) in {str(a) for a in spec.get("allowed", [])}
    return False


def _objective_ok(score: float, objective: dict, y_factual: float) -> bool:
    t = objective.get("type")
    if t == "target_min":    return score >= float(objective["value"])
    if t == "delta_improve": return score >= y_factual + float(objective["delta"])
    if t == "target_band":   return float(objective["lower"]) <= score <= float(objective["upper"])
    return False


# ---------------------------------------------------------------------------
# Multi-anchor pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    cb: CatBoostRegressor,
    tokenizer: AutoTokenizer,
    llm_model: AutoModelForCausalLM,
    user_text: str,
    export_dir: Path,
    model_path: Path,
    anchor_count: int,
) -> dict:

    anchors        = select_anchors(df, cb, n=anchor_count)
    all_parts:     list[pd.DataFrame] = []
    anchor_runs:   list[dict]         = []
    query_template: dict | None       = None

    for _, anchor_row in anchors.iterrows():
        idx       = int(anchor_row["_row_idx"])
        x_factual = df.loc[idx, FEATURES].to_dict()
        y_factual = float(cb.predict(pd.DataFrame([x_factual]))[0])

        q = build_query(
            user_text=user_text,
            x_factual=x_factual,
            y_factual=y_factual,
            llm_fn=lambda prompt: llm_generate_json(tokenizer, llm_model, prompt),
        )

        if query_template is None:
            query_template = {
                "objective":        copy.deepcopy(q["objective"]),
                "soft_constraints": copy.deepcopy(q["soft_constraints"]),
                "selection":        copy.deepcopy(q["selection"]),
                "_meta":            q.get("_meta", {}),
            }

        cf_df, before, after = generate_cf_table(df, cb, x_factual, q, y_factual)

        anchor_runs.append({
            "anchor_idx":   idx,
            "anchor_score": y_factual,
            "desired_range": objective_to_desired_range(q["objective"], y_factual),
            "n_cf": len(cf_df),
            "before_filter": before,
            "after_filter":  after,
        })

        if not cf_df.empty:
            part = cf_df.copy().reset_index(drop=True)
            part["_anchor_idx"]   = idx
            part["_anchor_score"] = y_factual
            part["_signature"]    = part.apply(_signature, axis=1)
            all_parts.append(part)

    # Merge, dedupe, top-k
    if not all_parts:
        merged = pd.DataFrame(columns=["cf_id"] + FEATURES + [TARGET])
    else:
        merged = (
            pd.concat(all_parts, ignore_index=True)
            .sort_values([TARGET, "_anchor_score"], ascending=[False, False])
            .drop_duplicates(subset=["_signature"], keep="first")
            .reset_index(drop=True)
        )
        k      = int(query_template["selection"]["k"])
        merged = merged.head(k).copy().reset_index(drop=True)
        merged.insert(0, "cf_id", range(len(merged)))

    # Build summary records
    summary_records: list[dict] = []
    if not merged.empty and query_template:
        obj = query_template["objective"]
        sc  = query_template["soft_constraints"]
        for _, row in merged.iterrows():
            score    = float(row[TARGET])
            y_anchor = float(row.get("_anchor_score", score))
            violated = [f for f in sc if not _soft_ok(row, f, sc[f])]
            summary_records.append({
                "cf_id":              int(row["cf_id"]),
                "score":              score,
                "objective_match":    _objective_ok(score, obj, y_anchor),
                "soft_match_rate":    1.0 - len(violated) / max(1, len(sc)),
                "violated_constraints": violated,
                "changed_feature_count": len(row.get("_changes", [])),
                "top_changes":        (row.get("_changes") or [])[:4],
            })

    # Export
    export_dir.mkdir(parents=True, exist_ok=True)
    keep_cols  = ["cf_id"] + FEATURES + [TARGET]
    cfs_out    = merged[[c for c in keep_cols if c in merged.columns]].copy()
    cfs_path   = export_dir / "alternatives.csv"
    cfs_out.to_csv(cfs_path, index=False)

    meta = {
        "surrogate_model_path": str(model_path),
        "target": TARGET, "features": FEATURES, "categorical": CATEGORICAL,
        "query_template": query_template,
        "anchor_runs":    anchor_runs,
        "n_options":      len(cfs_out),
    }
    meta_path = export_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    explanation  = "No alternatives available."
    explain_path = None
    if summary_records and query_template:
        explanation  = build_explanation(summary_records, query_template)
        explain_path = export_dir / "option_explanations.txt"
        explain_path.write_text(explanation + "\n", encoding="utf-8")

    return {
        "query_template":  query_template,
        "anchor_runs":     anchor_runs,
        "cf_df":           merged,
        "summary_records": summary_records,
        "explanation":     explanation,
        "cfs_out":         cfs_out,
        "cfs_path":        cfs_path,
        "meta_path":       meta_path,
        "explain_path":    explain_path,
    }


# ---------------------------------------------------------------------------
# Explanation  (fixed ranking + universal-violation detection)
# ---------------------------------------------------------------------------

def _rank_label(cf_id: int, score_map: dict[int, float]) -> str:
    """Return a human rank string that handles ties correctly."""
    score = score_map[cf_id]
    n     = len(score_map)
    rank  = sum(1 for s in score_map.values() if s > score + 1e-9) + 1
    tied  = [option_name(i) for i, s in score_map.items() if i != cf_id and abs(s - score) <= 1e-9]

    if n == 1:
        return "It is the only option."
    if tied and rank == 1:
        return f"It is tied for the highest score with {human_join(tied)}."
    if tied:
        return f"It is tied at rank {rank} of {n}."
    if rank == 1:
        return f"It is the highest-scoring option out of {n}."
    if rank == n:
        return f"It is the lowest-scoring option out of {n}."
    return f"It ranks {rank} of {n} by score."


def _objective_text(obj: dict) -> str:
    t = obj.get("type")
    if t == "target_min":    return f"Required quality ≥ {fmt3(obj['value'])}."
    if t == "target_band":   return f"Required quality in [{fmt3(obj['lower'])}, {fmt3(obj['upper'])}]."
    if t == "delta_improve": return "Required quality: satisfy the requested improvement target."
    return "Required quality: not specified."


def _requirement_text(score: float, obj: dict, match: bool) -> str:
    t = obj.get("type")
    s = "meets" if match else "misses"
    if t == "target_min":
        return f"Quality {fmt3(score)} — {s} the minimum {fmt3(obj['value'])}."
    if t == "target_band":
        inside = "inside" if match else "outside"
        return f"Quality {fmt3(score)} — {inside} [{fmt3(obj['lower'])}, {fmt3(obj['upper'])}]."
    if t == "delta_improve":
        return f"Quality {fmt3(score)} — {'meets' if match else 'does not meet'} the improvement target."
    return f"Quality {fmt3(score)}."


def _rank_records(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: (
        -int(r["objective_match"]),
        -r["soft_match_rate"],
        -r["score"],
        r["changed_feature_count"],
        r["cf_id"],
    ))


def build_explanation(summary_records: list[dict], query_template: dict) -> str:
    if not summary_records:
        return "No alternatives available to explain."

    obj       = query_template.get("objective", {})
    sc        = query_template.get("soft_constraints", {})
    score_map = {int(r["cf_id"]): float(r["score"]) for r in summary_records}
    ranked    = _rank_records(summary_records)
    best      = ranked[0]
    runner    = ranked[1] if len(ranked) > 1 else None

    # Detect violations shared by ALL options (so we don't repeat them per-option)
    all_violations = [set(r["violated_constraints"]) for r in summary_records]
    universal_violations = set.intersection(*all_violations) if all_violations else set()

    lines: list[str] = []
    lines.append(_objective_text(obj))

    # Score overview — compact
    score_parts = [f"{option_name(r['cf_id'])} {fmt3(r['score'])}" for r in
                   sorted(summary_records, key=lambda r: (-r["score"], r["cf_id"]))]
    lines.append("Scores: " + "; ".join(score_parts) + ".")

    # Universal violation note (shown once)
    if universal_violations:
        uv_labels = human_join([feat_label(v) for v in sorted(universal_violations)])
        lines.append(f"Note: all options violate the constraint on {uv_labels}.")

    lines.append("")

    for r in sorted(summary_records, key=lambda x: x["cf_id"]):
        cid        = int(r["cf_id"])
        per_option = [v for v in r["violated_constraints"] if v not in universal_violations]
        lines.append(f"{option_name(cid)}:")
        lines.append(f"  • {_requirement_text(r['score'], obj, r['objective_match'])} {_rank_label(cid, score_map)}")
        if per_option:
            lines.append(f"  • Violates: {human_join([feat_label(v) for v in per_option])}.")
        else:
            lines.append("  • Satisfies all per-option constraints.")

    lines.append("")

    # Best choice
    best_specific = [v for v in best["violated_constraints"] if v not in universal_violations]
    constraint_clause = (
        "all per-option constraints are satisfied"
        if not best_specific
        else f"it violates {human_join([feat_label(v) for v in best_specific])}"
    )
    lines.append(
        f"Best choice: {option_name(int(best['cf_id']))}. "
        f"Quality {fmt3(best['score'])}, {constraint_clause}. "
        f"{_rank_label(int(best['cf_id']), score_map)}"
    )

    if runner is not None:
        runner_specific = [v for v in runner["violated_constraints"] if v not in universal_violations]
        r_constraint = (
            "all per-option constraints are satisfied"
            if not runner_specific
            else f"it violates {human_join([feat_label(v) for v in runner_specific])}"
        )
        lines.append(
            f"Runner-up: {option_name(int(runner['cf_id']))}. "
            f"Quality {fmt3(runner['score'])}, {r_constraint}. "
            f"{_rank_label(int(runner['cf_id']), score_map)}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Display table
# ---------------------------------------------------------------------------

def build_display_table(cf_df: pd.DataFrame, summary_records: list[dict]) -> pd.DataFrame:
    if cf_df.empty: return pd.DataFrame()
    violations_map = {
        int(r["cf_id"]): (
            human_join([feat_label(v) for v in r["violated_constraints"]])
            if r["violated_constraints"] else "none"
        )
        for r in (summary_records or [])
    }
    rows = []
    for _, row in cf_df.iterrows():
        cid = int(row["cf_id"])
        rows.append({
            "Option":           option_name(cid),
            TARGET_LABEL:       feat_val_text(TARGET, row[TARGET]),
            "Backbone":         row["params_backbone"],
            "Image size":       row["params_image_size_key"],
            "Layer set":        row["params_layers_key"],
            "Review budget":    feat_val_text("params_soft_review_budget",  row["params_soft_review_budget"]),
            "Training fraction":feat_val_text("params_soft_train_fraction", row["params_soft_train_fraction"]),
            "Corruption level": row["params_soft_corruption_level"],
            "Violations":       violations_map.get(cid, "none"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in CATEGORICAL:
        if c in df.columns: df[c] = df[c].astype(str)
    return df


@st.cache_resource(show_spinner=False)
def load_surrogate(model_path: str) -> CatBoostRegressor:
    cb = CatBoostRegressor()
    cb.load_model(model_path)
    return cb


@st.cache_resource(show_spinner=False)
def load_llm(model_id: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
    model.eval()
    return tokenizer, model


def render_result(result: dict) -> None:
    st.subheader("Recommended alternatives")
    st.dataframe(
        build_display_table(result["cf_df"], result["summary_records"]),
        use_container_width=True,
    )

    st.subheader("Why these options")
    st.text(result["explanation"])

    with st.expander("Parsed request", expanded=False):
        st.json(result["query_template"])

    with st.expander("Full configurations", expanded=False):
        st.dataframe(rename_cols(result["cf_df"]), use_container_width=True)

    st.subheader("Downloads")
    col1, col2, col3 = st.columns(3)
    if result["cfs_path"].exists():
        col1.download_button(
            "alternatives.csv", data=result["cfs_path"].read_bytes(),
            file_name="alternatives.csv", mime="text/csv", use_container_width=True,
        )
    if result["meta_path"].exists():
        col2.download_button(
            "metadata.json", data=result["meta_path"].read_bytes(),
            file_name="metadata.json", mime="application/json", use_container_width=True,
        )
    if result["explain_path"] and result["explain_path"].exists():
        col3.download_button(
            "explanation.txt", data=result["explain_path"].read_bytes(),
            file_name="option_explanations.txt", mime="text/plain", use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="LLM Interface for CF-HPO", layout="wide")
    st.title("LLM Interface for CF-HPO")
    st.caption("Describe your goal. The app finds matching hyper-parameter configurations and explains the trade-offs.")

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("LLM model", LLM_MODEL_OPTIONS + ["Custom model id"])
        llm_model_id = (
            st.text_input("Custom LLM model id", DEFAULT_LLM_MODEL_ID)
            if model_choice == "Custom model id"
            else model_choice
        )
        st.header("Advanced")
        anchor_count = st.slider("Search breadth (anchors)", 1, 5, DEFAULT_ANCHOR_COUNT)

    try:
        with st.spinner("Loading dataset and surrogate model…"):
            df = load_dataframe(str(DEFAULT_CSV_PATH))
            cb = load_surrogate(str(DEFAULT_MODEL_PATH))
    except Exception as exc:
        st.error("Failed to load dataset or surrogate model.")
        st.exception(exc)
        st.stop()

    if df.empty:
        st.error("Loaded dataset is empty.")
        st.stop()

    st.subheader("What do you want?")
    cols = st.columns(len(EXAMPLE_REQUESTS))
    for i, (title, text) in enumerate(EXAMPLE_REQUESTS):
        if cols[i].button(title, use_container_width=True):
            st.session_state["user_text"] = text

    user_text = st.text_area(
        "Describe your goal and constraints",
        value=st.session_state.get("user_text", DEFAULT_USER_TEXT),
        height=160,
        placeholder="Example: I want quality ≥ 0.87, train fraction 0.3–0.6, and 5 options.",
    )

    if st.button("Generate alternatives", type="primary"):
        try:
            with st.spinner("Loading LLM…"):
                tokenizer, llm_model = load_llm(llm_model_id)

            with st.spinner("Generating alternatives…"):
                result = run_pipeline(
                    df=df, cb=cb,
                    tokenizer=tokenizer, llm_model=llm_model,
                    user_text=user_text,
                    export_dir=Path(DEFAULT_EXPORT_DIR),
                    model_path=Path(DEFAULT_MODEL_PATH),
                    anchor_count=int(anchor_count),
                )
            st.session_state["cf_result"] = result
            st.success("Alternatives generated.")
        except Exception as exc:
            st.exception(exc)

    if "cf_result" in st.session_state:
        render_result(st.session_state["cf_result"])


if __name__ == "__main__":
    main()