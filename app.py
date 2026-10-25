"""Existing retirement_sim Streamlit entry point, now with live model inference.
Run: python -m streamlit run app.py
"""
from pathlib import Path
import streamlit as st
from backend.inference import predict_retirement, MODEL_PATH

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
MODEL_PAGE = FRONTEND / "retirement_model_explained.html"

st.set_page_config(
    page_title="Retire-Proof | Retirement Simulator",
    page_icon="🌅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def on_predict():
    """Triggered only when Generate prediction is clicked."""
    event = st.session_state["retirement_simulator_ui"].submit
    if not event:
        return
    st.session_state["retirement_ui_last_input"] = event.get("raw", {})
    try:
        p = predict_retirement(event["payload"])
        st.session_state["retirement_ui_reply"] = {
            "prediction": p,
            "model_version": "v1.0",
        }
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        st.session_state["retirement_ui_reply"] = {"error": str(exc)}


def simulator_page():
    # Works with the uploaded self-contained index(1).html markup,
    # converted into a Streamlit v2 component without changing index.html.
    assets = {
        "html": FRONTEND / "simulator_component.html",
        "css": FRONTEND / "simulator_component.css",
        "js": FRONTEND / "simulator_component.js",
    }
    missing = [path.name for path in assets.values() if not path.is_file()]
    if missing:
        st.error(f"Missing simulator integration files in frontend/: {', '.join(missing)}")
        return
    if not MODEL_PATH.is_file():
        st.warning("Missing backend/retire_model_v12.npz — the form will show an error until the model file is present.")

    # Component V2 sends inputs to the Python app via a trigger, no separate API.
    component = st.components.v2.component(
        "retire_proof_original_ui",
        html=assets["html"].read_text(encoding="utf-8"),
        css=assets["css"].read_text(encoding="utf-8"),
        js=assets["js"].read_text(encoding="utf-8"),
        isolate_styles=True,
    )
    component(
        key="retirement_simulator_ui",
        data={
            "reply": st.session_state.get("retirement_ui_reply"),
            "last_input": st.session_state.get("retirement_ui_last_input"),
        },
        on_submit_change=on_predict,
    )


def model_explained_page():
    if not MODEL_PAGE.is_file():
        st.error("Missing frontend/retirement_model_explained.html")
        return
    st.iframe(
        MODEL_PAGE,
        width="stretch",
        height="content",
        alt="Inside the Model: explanation of the retirement neural network",
    )


nav = st.navigation(
    [
        st.Page(simulator_page, title="Simulator", icon=":material/calculate:", default=True),
        st.Page(model_explained_page, title="Inside the Model", icon=":material/psychology:"),
    ],
    position="top",
)
nav.run()
