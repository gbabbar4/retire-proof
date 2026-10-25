"""Streamlit entrypoint for the existing retirement_sim project.

Run from the project root with:
    python -m streamlit run app.py
"""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PAGE = ROOT / "frontend" / "retirement_model_explained.html"

st.set_page_config(
    page_title="Retirement Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def simulator_page():
    st.title("Retirement Simulator")
    st.info(
        "The current HTML simulator in frontend/index.html and Python backend "
        "are unchanged. This page will be connected after we inspect their "
        "input fields and prediction API."
    )
    st.caption("No demo or fabricated predictions are shown here.")


def model_explained_page():
    if not MODEL_PAGE.is_file():
        st.error(
            "Could not find frontend/retirement_model_explained.html. "
            "Check the exact filename in VS Code."
        )
        return

    st.iframe(
        MODEL_PAGE,
        width="stretch",
        height="content",
        alt="Inside the Model: explanation of the retirement neural network",
    )


navigation = st.navigation(
    [
        st.Page(simulator_page, title="Simulator", icon=":material/calculate:", default=True),
        st.Page(model_explained_page, title="Inside the Model", icon=":material/psychology:"),
    ],
    position="top",
)
navigation.run()