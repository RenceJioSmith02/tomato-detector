const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const dropText = document.getElementById("drop-text");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const spinner = document.getElementById("spinner");
const resultSection = document.getElementById("resultSection");

let probChartInstance = null;
let droppedFile = null; // ← track drag-and-dropped file separately

// ── Drag & drop ─────────────────────────────────────────────────────────────
dropArea.addEventListener("click", () => fileInput.click());
dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("drag-over");
});
dropArea.addEventListener("dragleave", () =>
  dropArea.classList.remove("drag-over"),
);
dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    droppedFile = file; // ← save the dropped file
    showPreview(file);
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    droppedFile = null; // ← file input takes priority, clear dropped
    showPreview(fileInput.files[0]);
  }
});

// ── Get whichever file is active ─────────────────────────────────────────────
function getActiveFile() {
  return fileInput.files[0] || droppedFile;
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = "block";
    dropText.style.display = "none";
  };
  reader.readAsDataURL(file);
}

// ── Clear ────────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  fileInput.value = "";
  droppedFile = null; // ← also clear dropped file
  preview.src = "";
  preview.style.display = "none";
  dropText.style.display = "";
  resultSection.style.display = "none";
  if (probChartInstance) {
    probChartInstance.destroy();
    probChartInstance = null;
  }
});

// ── Predict ──────────────────────────────────────────────────────────────────
predictBtn.addEventListener("click", () => {
  const file = getActiveFile(); // ← works for both browse & drag-drop
  if (!file) {
    alert("Please select or drop an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  spinner.style.display = "block";
  resultSection.style.display = "none";

  fetch("/predict", { method: "POST", body: formData })
    .then((r) => r.json())
    .then((data) => {
      spinner.style.display = "none";
      if (data.error) {
        alert(data.error);
        return;
      }
      displayResult(data);
    })
    .catch((err) => {
      spinner.style.display = "none";
      alert("Prediction failed. Check the server logs.");
      console.error(err);
    });
});

// ── Display Result ────────────────────────────────────────────────────────────
function displayResult(data) {
  const badge = document.getElementById("resultBadge");
  const result = document.getElementById("result");
  const confText = document.getElementById("confidence-text");
  const treatment = document.getElementById("treatment");

  badge.className = "result-badge badge-" + data.prediction;
  const labelMap = {
    healthy: "Healthy",
    late_blight: "Late Blight",
    other_diseases: "Unknown / Other",
  };
  badge.textContent = labelMap[data.prediction] || data.prediction;

  result.textContent = labelMap[data.prediction] || data.prediction;
  confText.textContent = `Model confidence: ${data.confidence}%`;
  treatment.textContent = data.treatment;

  resultSection.style.display = "flex";

  // ── Chart ──
  if (probChartInstance) probChartInstance.destroy();

  const ctx = document.getElementById("probChart").getContext("2d");
  const labels = data.classes.map((c) => labelMap[c] || c);
  const values = data.probabilities.map((v) => +(v * 100).toFixed(2));

  const colorMap = {
    late_blight: "rgba(192, 57, 43, 0.75)",
    healthy: "rgba(45, 122, 69, 0.75)",
    other_diseases: "rgba(230, 126, 34, 0.75)",
  };
  const bgColors = data.classes.map(
    (c) => colorMap[c] || "rgba(100,100,100,0.5)",
  );
  const bdColors = bgColors.map((c) => c.replace("0.75", "1"));

  probChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Confidence (%)",
          data: values,
          backgroundColor: bgColors,
          borderColor: bdColors,
          borderWidth: 2,
          borderRadius: 8,
        },
      ],
    },
    options: {
      indexAxis: "y",
      scales: {
        x: {
          min: 0,
          max: 100,
          grid: { color: "rgba(0,0,0,.05)" },
          ticks: { callback: (v) => v + "%" },
        },
        y: { grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: (ctx) => ` ${ctx.parsed.x.toFixed(2)}%` },
        },
      },
      animation: { duration: 600, easing: "easeOutQuart" },
    },
  });
}
