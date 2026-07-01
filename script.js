// ============================================================
// Spam Mail Detector — frontend logic
// Talks to the FastAPI backend at API_BASE.
// ============================================================

// If you run backend.py with `uvicorn backend:app` on the default port,
// this will work whether you open index.html directly OR visit
// http://localhost:8000 (where FastAPI also serves this frontend).
const API_BASE = "http://localhost:8000";

const messageInput = document.getElementById("messageInput");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText = document.getElementById("btnText");
const btnSpinner = document.getElementById("btnSpinner");
const errorMsg = document.getElementById("errorMsg");

const resultSection = document.getElementById("resultSection");
const resultIcon = document.getElementById("resultIcon");
const resultLabel = document.getElementById("resultLabel");
const resultSub = document.getElementById("resultSub");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBarFill = document.getElementById("confidenceBarFill");
const meterDot = document.getElementById("meterDot");
const signalWords = document.getElementById("signalWords");
const cleanedText = document.getElementById("cleanedText");
const metricsGrid = document.getElementById("metricsGrid");

const SAMPLES = {
  spam: "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now: bit.ly/claim-prize before it expires!!!",
  ham: "Hey, are we still on for lunch tomorrow at 1pm? Let me know if you want to change the spot.",
};

// ---------------- Char counter ----------------
messageInput.addEventListener("input", () => {
  charCount.textContent = messageInput.value.length;
});

// ---------------- Sample chips ----------------
document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.sample;
    messageInput.value = SAMPLES[key];
    charCount.textContent = messageInput.value.length;
    messageInput.focus();
  });
});

// ---------------- Analyze ----------------
analyzeBtn.addEventListener("click", analyzeMessage);
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    analyzeMessage();
  }
});

async function analyzeMessage() {
  const text = messageInput.value.trim();
  errorMsg.classList.add("hidden");

  if (!text) {
    showError("Type or paste a message first.");
    return;
  }

  setLoading(true);
  resultSection.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) throw new Error(`Server responded with ${res.status}`);

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(
      "Couldn't reach the model backend. Make sure `uvicorn backend:app --reload` is running on localhost:8000."
    );
    console.error(err);
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  btnText.textContent = isLoading ? "Analyzing..." : "Analyze Message";
  btnSpinner.classList.toggle("hidden", !isLoading);
}

function showError(text) {
  errorMsg.textContent = text;
  errorMsg.classList.remove("hidden");
}

function renderResult(data) {
  const { label, is_spam, confidence, spam_probability, cleaned_text, top_signal_words } = data;

  resultSection.classList.remove("hidden");
  resultSection.classList.toggle("is-spam", is_spam);
  resultSection.classList.toggle("is-ham", !is_spam);

  resultIcon.textContent = is_spam ? "🚫" : "✅";
  resultLabel.textContent = is_spam ? "SPAM" : "HAM";
  resultSub.textContent = is_spam
    ? "This message shows strong spam signals."
    : "This message looks like a normal, legitimate message.";

  const confPct = Math.round(confidence * 100);
  confidenceValue.textContent = `${confPct}%`;
  requestAnimationFrame(() => {
    confidenceBarFill.style.width = `${confPct}%`;
  });

  const meterPct = Math.round(spam_probability * 100);
  requestAnimationFrame(() => {
    meterDot.style.left = `${meterPct}%`;
  });

  cleanedText.textContent = cleaned_text || "(nothing left after cleaning)";

  signalWords.innerHTML = "";
  if (top_signal_words && top_signal_words.length) {
    const heading = document.createElement("div");
    heading.style.width = "100%";
    heading.style.fontSize = "0.85rem";
    heading.style.fontWeight = "700";
    heading.style.marginBottom = "4px";
    heading.textContent = "Top signal words:";
    signalWords.appendChild(heading);

    top_signal_words.forEach((word) => {
      const chip = document.createElement("span");
      chip.className = "signal-word";
      chip.textContent = word;
      signalWords.appendChild(chip);
    });
  }

  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------- Load model metrics on page load ----------------
async function loadMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) throw new Error("metrics fetch failed");
    const data = await res.json();
    renderMetrics(data);
  } catch (err) {
    metricsGrid.innerHTML = `<p style="color:#b98d63; font-size:0.85rem;">Model metrics unavailable — start the backend to see them.</p>`;
  }
}

function renderMetrics(data) {
  const { best_model_name, all_results } = data;
  metricsGrid.innerHTML = "";

  Object.entries(all_results).forEach(([name, r]) => {
    const tile = document.createElement("div");
    tile.className = "metric-tile" + (name === best_model_name ? " best" : "");

    const nameEl = document.createElement("div");
    nameEl.className = "model-name";
    nameEl.textContent = name === best_model_name ? `${name} ★` : name;
    tile.appendChild(nameEl);

    const rows = [
      ["Accuracy", r.accuracy],
      ["Precision", r.precision],
      ["Recall", r.recall],
      ["F1", r.f1],
    ];
    rows.forEach(([label, val]) => {
      const row = document.createElement("div");
      row.className = "metric-row";
      row.innerHTML = `<span>${label}</span><b>${(val * 100).toFixed(1)}%</b>`;
      tile.appendChild(row);
    });

    metricsGrid.appendChild(tile);
  });
}

loadMetrics();
