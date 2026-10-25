# Retire-Proof — Replace the Simulator placeholder (existing project)

This patch targets **your current `retirement_sim` folder**, not a new project.
Your original frontend/index.html, script.js, style.css, backend/main.py and
backend/neural_network.py stay unchanged.

## 1. Back up your current `app.py`

Duplicate it as `app_backup.py` or use Git before copying files.

## 2. Copy patch files into your existing project

Copy the files from this ZIP **into the existing retirement_sim folder**, retaining paths:

```
retirement_sim/
  app.py                    <-- REPLACE placeholder
  requirements.txt          <-- REPLACE previous 1-line requirements
  backend/
    main.py                 <-- KEEP original
    neural_network.py       <-- KEEP original
    retire_model_v12.npz    <-- KEEP original weights (not in ZIP)
    inference.py            <-- ADD
  frontend/
    index.html              <-- KEEP original
    script.js               <-- KEEP original
    style.css               <-- KEEP original
    retirement_model_explained.html  <-- KEEP working page
    simulator_component.html         <-- ADD (converted uploaded index(1).html)
    simulator_component.css          <-- ADD
    simulator_component.js           <-- ADD
```

## 3. Install and run (VS Code terminal in project root)

```
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Visit http://localhost:8501. Both tabs should render. On Simulator, click Generate
prediction. Python loads backend/retire_model_v12.npz on the server and returns
result to the component, with **NO** uvicorn/FastAPI server needed.

## Important distinctions

- `index(1).html` was the newer self-contained HTML you uploaded. The patch uses
  its original markup and styling but replaces its localhost fetch with a Streamlit
  bidirectional component. `frontend/index.html` stays untouched.
- Don't copy old `script.js` into simulator_component.html: uploaded HTML already had
  different inlined JS, and we replaced that with simulator_component.js.
- The old FastAPI backend/main.py is left for experiments; the Streamlit page calls
  backend/inference.py directly using your saved weights/scaler.
- This ZIP deliberately does not contain retire_model_v12.npz. Keep your real file.
- This approach uses trusted Streamlit components v2, available in modern Streamlit.
- We have NOT confirmed results against your original FastAPI since the .npz
  model file was not uploaded. Before going live, compare several identical inputs.
- Financial estimates are experimental. Do not present as guaranteed outcomes.
- The Inside the Model page's internal Back to Simulator link remains a separate
  polish task; top Streamlit navigation works.

## Deployment checklist

Add/confirm .gitignore containing `.venv/`, `__pycache__/`, `.DS_Store`, `.env`.
Include backend/retire_model_v12.npz if non-sensitive and permitted to distribute,
then commit project to a GitHub repo. Streamlit Community Cloud entrypoint is app.py.
No extra FastAPI service necessary for the integrated simulator.
