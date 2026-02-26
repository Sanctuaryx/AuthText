import os
import requests
import streamlit as st
from textwrap import dedent
from typing import Optional
import json
from pathlib import Path

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8001")

st.set_page_config(
    page_title="AuthText",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=60)
def fetch_models(base_url: str) -> dict:
    r = requests.get(f"{base_url}/models", timeout=10)
    r.raise_for_status()
    return r.json()

def badge(label: str, kind: str = "neutral") -> str:
    styles = {
        "good":    "background:#E7F7EE;color:#0F5132;border:1px solid #B7E4C7;",
        "bad":     "background:#FDECEC;color:#842029;border:1px solid #F5C2C7;",
        "neutral": "background:#EEF2FF;color:#1E3A8A;border:1px solid #C7D2FE;",
        "warn":    "background:#FFF7E6;color:#7A4D00;border:1px solid #FFE0A3;",
    }
    style = styles.get(kind, styles["neutral"])
    return f"""
    <span style="
        padding:6px 10px;border-radius:999px;
        font-size:0.85rem;font-weight:600;
        display:inline-block; {style}
    ">{label}</span>
    """

def kind_for_decision(decision: Optional[str]) -> str:
    d = (decision or "").lower()
    if d in {"humano", "human"}:
        return "good"
    if d in {"ia", "ai", "generated"}:
        return "bad"
    if d in {"indeterminado", "unknown"}:
        return "warn"
    return "neutral"

def kind_for_confidence(conf: Optional[str]) -> str:
    c = (conf or "").lower()
    if c in {"high", "alta"}:
        return "good"
    if c in {"medium", "media"}:
        return "neutral"
    if c in {"low", "baja"}:
        return "warn"
    return "neutral"

def render_result_card(data: dict) -> None:
    decision = data.get("decision")
    confidence = data.get("confidence")

    decision_badge = badge(f"Decisión: {decision}", kind_for_decision(decision))
    conf_badge = badge(f"Confianza: {confidence}", kind_for_confidence(confidence))

    card_html = dedent(f"""
        <div style="width:100%;
            border:1px solid rgba(0,0,0,.08);
            border-radius:14px;
            padding:14px;
            background:#fff;
            box-shadow:0 4px 16px rgba(0,0,0,.06);">
            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px;">
                {decision_badge}
                {conf_badge}
    """).strip()

    st.markdown(card_html, unsafe_allow_html=True)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    with st.expander("Ver respuesta completa"):
        st.json(data)

def run_inference(base_url: str, model: str, text: str) -> dict:
    r = requests.post(
        f"{base_url}/predict",
        json={"text": text, "model": model},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"Error {r.status_code} en /predict\n\n{r.text}")
    try:
        return r.json()
    except ValueError:
        raise RuntimeError("La respuesta no es JSON válido.\n\n" + r.text)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_text" not in st.session_state:
    st.session_state.pending_text = None
if "reset_id" not in st.session_state:
    st.session_state.reset_id = 0

SUGGESTIONS_PATH = Path(__file__).parent / "suggestions.json"

@st.cache_data
def load_suggestions(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SUGGESTIONS = load_suggestions(SUGGESTIONS_PATH)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Configuración")

    base_url = st.text_input("BASE_URL", value=BASE, help="URL base del servicio REST")

    try:
        models = fetch_models(base_url)
        available = models.get("available", [])
        default = models.get("default", available[0] if available else None)
        model = st.selectbox(
            "Modelo",
            available,
            index=available.index(default) if default in available else 0,
            disabled=not bool(available),
        )
    except Exception as e:
        st.error("No se pudo obtener /models. Revisa BASE_URL y que el servicio esté levantado.")
        st.caption(str(e))
        model = None

    st.divider()

# -----------------------------
# Header row (tipo assistant)
# -----------------------------
header = st.container()
with header:
    st.markdown("<div style='font-size:52px; line-height:1;'>❉</div>", unsafe_allow_html=True)

    title_row = st.container()
    with title_row:
        left, right = st.columns([8, 2], vertical_alignment="bottom")
        with left:
            st.title("AuthText Detector", anchor=False)
            st.caption("Detección IA vs humano en español.")
        with right:
            def clear_conversation():
                st.session_state.messages = []
                st.session_state.pending_text = None
                st.session_state.reset_id += 1

            st.button("Reiniciar", icon=":material/refresh:", on_click=clear_conversation, use_container_width=True)

st.divider()

# -----------------------------
# Empty state (antes de primera interacción)
# -----------------------------
if len(st.session_state.messages) == 0 and not st.session_state.get("pending_text"):
    with st.container():
        initial = st.chat_input(
        "Pega aquí el texto a analizar…",
        key=f"initial_question_{st.session_state.reset_id}",
        )

    selected = st.pills(
        label="Ejemplos",
        label_visibility="collapsed",
        options=list(SUGGESTIONS.keys()),
        key=f"selected_example_{st.session_state.reset_id}",
        )

    if initial:
        st.session_state.pending_text = initial
        st.rerun()

    if selected:
        st.session_state.pending_text = SUGGESTIONS[selected]
        st.rerun()

    st.stop()

# -----------------------------
# Chat input (follow-ups)
# -----------------------------
user_message = st.chat_input(
    "Pega otro texto para analizar…",
    key=f"user_message_{st.session_state.reset_id}",
)

if (not user_message) and st.session_state.get("pending_text"):
    user_message = st.session_state.pending_text
    st.session_state.pending_text = None

if user_message:
    user_message = user_message.replace("$", r"\$")

# -----------------------------
# Render history
# -----------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        if role == "user":
            st.text(msg["content"])
        else:
            if "data" in msg and isinstance(msg["data"], dict):
                render_result_card(msg["data"])
            else:
                st.markdown(msg.get("content", ""))

# -----------------------------
# New message + inference
# -----------------------------
if user_message:
    if model is None:
        with st.chat_message("assistant"):
            st.error("No hay modelo seleccionado (no se pudo cargar /models).")
        st.stop()

    if not user_message.strip():
        with st.chat_message("assistant"):
            st.error("El texto está vacío.")
        st.stop()

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Ejecutando inferencia..."):
                data = run_inference(base_url=base_url, model=model, text=user_message)

            render_result_card(data)

            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": "", "data": data})

        except requests.exceptions.RequestException as e:
            st.error("No se pudo conectar con el servicio de inferencia.")
            st.caption(str(e))
        except Exception as e:
            st.error("Se produjo un error durante la inferencia.")
            st.code(str(e))