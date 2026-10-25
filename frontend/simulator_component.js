// Streamlit Component v2 communication bridge for your original simulator UI.
// No localhost:8000 fetch: submissions go to Streamlit Python via setTriggerValue.
export default function (component) {
  const { parentElement, data, setTriggerValue } = component;
  const root = parentElement.querySelector('.retire-sim-root');
  if (!root) return;
  const find = (id) => root.querySelector(`#${id}`);
  const form = find('retirement-form');
  const output = find('prediction-output');
  const button = find('predict-button');
  const gauge = find('gauge');
  const number = find('prediction-number');
  const percentSign = find('percent-sign');
  const themeToggle = find('theme-toggle');

  // Theme preference is independent of Streamlit's theme.
  function setTheme(isDark) {
    root.classList.toggle('dark-mode', isDark);
    themeToggle.textContent = isDark ? '☀️ Light mode' : '🌙 Dark mode';
    themeToggle.setAttribute('aria-pressed', String(isDark));
    themeToggle.setAttribute('aria-label', isDark ? 'Enable light mode' : 'Enable dark mode');
  }
  let savedTheme = 'light';
  try { savedTheme = localStorage.getItem('retire-proof-theme') || 'light'; } catch (_) {}
  setTheme(savedTheme === 'dark');
  themeToggle.onclick = () => {
    const dark = !root.classList.contains('dark-mode');
    setTheme(dark);
    try { localStorage.setItem('retire-proof-theme', dark ? 'dark' : 'light'); } catch (_) {}
  };

  function updateYears() {
    const ageValue = find('age').value;
    const retirementValue = find('retirement').value;
    const age = Number(ageValue), retirement = Number(retirementValue);
    if (ageValue === '' || retirementValue === '' || retirement <= age) {
      find('years-to-retirement').textContent = 'Retirement age must be greater than current age';
      return;
    }
    const years = retirement - age;
    find('years-to-retirement').textContent = `${Number.isInteger(years) ? years : years.toFixed(1)} years until your planned retirement`;
  }
  find('age').oninput = updateYears;
  find('retirement').oninput = updateYears;

  // Retain submitted inputs when a Streamlit rerun remounts the UI.
  if (data?.last_input) {
    for (const [key, val] of Object.entries(data.last_input)) {
      const el = find(key);
      if (el && val !== undefined && val !== null) {
        // equity_alloc and infl are saved as percentages (0-100) in last_input.
        el.value = val;
      }
    }
  }
  updateYears();

  if (data?.reply) {
    const reply = data.reply;
    if (reply.error) {
      output.textContent = 'Unable to get prediction: ' + reply.error;
      output.style.color = '#f38b89';
    } else if (typeof reply.prediction === 'number' && Number.isFinite(reply.prediction) && reply.prediction >= 0 && reply.prediction <= 1) {
      const percentage = reply.prediction * 100;
      number.textContent = percentage.toFixed(2);
      percentSign.hidden = false;
      gauge.style.setProperty('--angle', `${reply.prediction * 360}deg`);
      gauge.setAttribute('aria-label', `Experimental NN estimate: ${percentage.toFixed(2)} percent`);
      find('model-version').textContent = reply.model_version || 'v1.0';
      output.textContent = 'Prediction calculated. This is a model estimate, not a guarantee.';
      output.style.color = '';
    }
    button.disabled = false;
    button.innerHTML = 'Generate prediction <span aria-hidden="true">↗</span>';
  }

  form.onsubmit = (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const fieldNames = ['age', 'retirement', 'corpus', 'monthly_savings', 'monthly_expense_today', 'equity_alloc', 'infl'];
    const raw = {};
    for (const field of fieldNames) {
      if (find(field).value.trim() === '') {
        output.textContent = 'Please fill in all fields.';
        return;
      }
      raw[field] = Number(find(field).value);
    }
    if (!Object.values(raw).every(Number.isFinite)) {
      output.textContent = 'Please enter valid numbers for all fields.';
      return;
    }
    if (raw.retirement <= raw.age) {
      output.textContent = 'Retirement age must be greater than current age.';
      return;
    }
    if (raw.equity_alloc < 0 || raw.equity_alloc > 100 || raw.infl < 0 || raw.infl > 100) {
      output.textContent = 'Equity allocation and inflation must be between 0 and 100%.';
      return;
    }
    // Python expects fractions for these features, matching the FastAPI backend.
    const payload = { ...raw, equity_alloc: raw.equity_alloc / 100, infl: raw.infl / 100 };
    button.disabled = true;
    button.textContent = 'Calculating…';
    output.textContent = 'Calculating your estimate…';
    output.style.color = '';
    setTriggerValue('submit', { payload, raw });
  };
}
