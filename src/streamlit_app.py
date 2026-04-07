from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import json
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MAX_LENGTH = 128
LABELS_PATH = MODEL_DIR / "labels.json"


@st.cache_resource(show_spinner=False)
def load_classifier():
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model.eval()
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
    )
    return clf, model.config.num_labels


def normalize_text(title: str, abstract: str) -> str:
    title = title.strip()
    abstract = abstract.strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def top_95(scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(scores, key=lambda item: item["score"], reverse=True)
    picked = []
    total = 0.0
    for item in ordered:
        picked.append(item)
        total += float(item["score"])
        if total >= 0.95:
            break
    return picked


def format_results(scores: List[Dict[str, Any]], label_map: Dict[str, str]) -> pd.DataFrame:
    rows = [
        {
            "topic": label_map.get(str(item["label"]), str(item["label"])),
            "probability": round(float(item["score"]) * 100, 2),
        }
        for item in scores
    ]
    return pd.DataFrame(rows)


def get_top1(scores: List[Dict[str, Any]], label_map: Dict[str, str]) -> Tuple[str, float]:
    best = max(scores, key=lambda item: item["score"])
    label = label_map.get(str(best["label"]), str(best["label"]))
    return label, float(best["score"])


st.set_page_config(page_title="PaperScope", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.1rem; }
    .subtitle { color: #5b5b5b; margin-bottom: 1.2rem; }
    .metric-box { padding: 0.5rem 0.75rem; border-radius: 0.6rem; background: #f6f7fb; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">PaperScope</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Классификация статей по тематикам arXiv</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1], gap="large")

with left:
    st.write(
        "Введите название и/или аннотацию статьи. "
        "Модель вернет темы по убыванию вероятности, пока сумма не превысит 95%."
    )
    with st.form("paper_form"):
        title_input = st.text_input(
            "Название статьи", placeholder="Attention Is All You Need"
        )
        abstract_input = st.text_area(
            "Аннотация",
            placeholder="We propose a new simple network architecture, the Transformer...",
            height=180,
        )
        submitted = st.form_submit_button("Классифицировать")

label_map = {}
if LABELS_PATH.exists():
    with LABELS_PATH.open("r", encoding="utf-8") as handle:
        names = json.load(handle)
    label_map = {str(i): name for i, name in enumerate(names)}

if submitted:
    text = normalize_text(title_input, abstract_input)
    if not text:
        st.warning("Введите название статьи и/или аннотацию.")
        st.stop()

    with st.spinner("Считаем вероятности тем..."):
        classifier, num_labels = load_classifier()
        scores = classifier(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            top_k=None,
        )


    if isinstance(scores, dict):
        scores = [scores]
    elif isinstance(scores, list) and scores and isinstance(scores[0], list):
        scores = scores[0]

    top_scores = top_95(scores)
    result_df = format_results(top_scores, label_map)
    top_label, top_score = get_top1(scores, label_map)

    st.subheader("Тематики (top-95%)")
    st.dataframe(result_df, use_container_width=True, hide_index=True)
    st.bar_chart(result_df.set_index("topic"))
    st.markdown(
        f'<div class="metric-box">Топ‑1: <b>{top_label}</b> '
        f'({top_score * 100:.2f}%)</div>',
        unsafe_allow_html=True,
    )

with st.expander("Как считается top-95%"):
    st.write(
        "Темы сортируются по убыванию вероятности, затем добавляются в ответ до тех пор, "
        "пока сумма вероятностей не превысит 95%."
    )