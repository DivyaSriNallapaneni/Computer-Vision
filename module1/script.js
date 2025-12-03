const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");

const imageInput = document.getElementById("imageInput");
const distanceInput = document.getElementById("distanceInput");
const focalInput = document.getElementById("focalInput");
const sensorWidthInput = document.getElementById("sensorWidthInput");
const trueSizeInput = document.getElementById("trueSizeInput");
const statusDiv = document.getElementById("status");
const resultDiv = document.getElementById("result");
const resetBtn = document.getElementById("resetPointsBtn");
const addEvalBtn = document.getElementById("addEvalBtn");
const evalTableBody = document.getElementById("evalTableBody");

let img = new Image();
let imgLoaded = false;
let clickPoints = [];   // [{x,y}, {x,y}]
let lastEval = null;    // {Z_cm, trueSize_cm, predicted_cm, errorPct}
let evalCount = 0;

function setStatus(msg) {
  statusDiv.textContent = "Status: " + msg;
}

// Load image
imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  img.onload = () => {
    imgLoaded = true;

    // Resize canvas to image size (or scaled down if huge)
    const maxWidth = 800;
    let scale = 1.0;
    if (img.width > maxWidth) {
      scale = maxWidth / img.width;
    }
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;

    // Make sure CSS size matches canvas size exactly
    canvas.style.width = canvas.width + "px";
    canvas.style.height = canvas.height + "px";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    clickPoints = [];
    lastEval = null;
    addEvalBtn.disabled = true;
    resultDiv.innerHTML = "";
    setStatus("Image loaded. Click TWO points on the object.");
  };
  img.src = url;
});

// Click handling – now using offsetX/offsetY directly (since CSS == canvas size)
canvas.addEventListener("click", (e) => {
  if (!imgLoaded) return;

  const x = e.offsetX;
  const y = e.offsetY;

  clickPoints.push({ x, y });
  drawCanvas();

  if (clickPoints.length === 2) {
    setStatus("Two points selected. Computing real-world dimension...");
    computeRealWorldDimension();
  } else {
    setStatus(`Point ${clickPoints.length} selected. Click one more point.`);
  }
});

resetBtn.addEventListener("click", () => {
  clickPoints = [];
  lastEval = null;
  addEvalBtn.disabled = true;

  if (imgLoaded) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }
  resultDiv.innerHTML = "";
  setStatus("Points reset. Click two points on the object.");
});

function drawCanvas() {
  if (!imgLoaded) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "red";
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;

  clickPoints.forEach((p) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
    ctx.fill();
  });

  if (clickPoints.length === 2) {
    const p1 = clickPoints[0];
    const p2 = clickPoints[1];
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
}

function computeRealWorldDimension() {
  if (!imgLoaded || clickPoints.length < 2) return;

  const p1 = clickPoints[0];
  const p2 = clickPoints[1];

  // Pixel distance directly on the (already scaled) canvas
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const pixelLengthCanvas = Math.sqrt(dx * dx + dy * dy);

  // Treat canvas pixels as image pixels (simpler behaviour like before)
  const pixelLengthImage = pixelLengthCanvas;

  const Z_cm = parseFloat(distanceInput.value);
  const f_mm = parseFloat(focalInput.value);
  const sensorWidth_mm = parseFloat(sensorWidthInput.value);
  const trueSize_cm = parseFloat(trueSizeInput.value);

  if (
    isNaN(Z_cm) || isNaN(f_mm) || isNaN(sensorWidth_mm) ||
    Z_cm <= 0 || f_mm <= 0 || sensorWidth_mm <= 0
  ) {
    alert("Please enter valid positive camera parameters.");
    lastEval = null;
    addEvalBtn.disabled = true;
    return;
  }

  // Focal length in pixels based on displayed image width
  const imageWidth_px = canvas.width; // use the same width we draw with
  const f_px = f_mm * (imageWidth_px / sensorWidth_mm);

  const Z_mm = Z_cm * 10.0;

  // Perspective projection: X = (Z * x) / f
  const X_mm = (Z_mm * pixelLengthImage) / f_px;
  const X_cm = X_mm / 10.0;

  let errorPct = null;
  let errorText = "N/A";

  if (!isNaN(trueSize_cm) && trueSize_cm > 0) {
    const err = Math.abs(X_cm - trueSize_cm) / trueSize_cm * 100.0;
    errorPct = err;
    errorText = err.toFixed(2) + " %";
  }

  resultDiv.innerHTML = `
    <b>Calculation details:</b><br>
    Pixel length (between clicks): ${pixelLengthImage.toFixed(2)} px<br>
    Image width used: ${imageWidth_px.toFixed(0)} px<br>
    Focal length in pixels: f = ${f_px.toFixed(2)} px<br>
    Distance: Z = ${Z_cm.toFixed(2)} cm = ${Z_mm.toFixed(2)} mm<br>
    <br>
    Using perspective projection: X = (Z · x) / f<br>
    X (mm) = (${Z_mm.toFixed(2)} × ${pixelLengthImage.toFixed(2)}) / ${f_px.toFixed(2)} = ${X_mm.toFixed(2)} mm<br>
    X (cm) = ${X_cm.toFixed(2)} cm<br>
    <br>
    True size (for evaluation): ${
      !isNaN(trueSize_cm) && trueSize_cm > 0 ? trueSize_cm.toFixed(2) + " cm" : "N/A"
    }<br>
    Error: ${errorText}
  `;

  if (!isNaN(trueSize_cm) && trueSize_cm > 0 && errorPct !== null) {
    lastEval = {
      Z_cm,
      trueSize_cm,
      predicted_cm: X_cm,
      errorPct
    };
    addEvalBtn.disabled = false;
  } else {
    lastEval = null;
    addEvalBtn.disabled = true;
  }

  setStatus("Done. You can reset points or add this result to the evaluation table.");
}

// Add current result to evaluation table
addEvalBtn.addEventListener("click", () => {
  if (!lastEval) return;

  // Remove placeholder row if present
  if (
    evalTableBody.children.length === 1 &&
    evalTableBody.children[0].dataset.placeholder === "true"
  ) {
    evalTableBody.innerHTML = "";
  }

  evalCount += 1;
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${evalCount}</td>
    <td>${lastEval.Z_cm.toFixed(1)}</td>
    <td>${lastEval.trueSize_cm.toFixed(2)}</td>
    <td>${lastEval.predicted_cm.toFixed(2)}</td>
    <td>${lastEval.errorPct.toFixed(2)}%</td>
    <td></td> <!-- You can manually type a note later if you want -->
  `;
  evalTableBody.appendChild(row);

  addEvalBtn.disabled = true;
  setStatus("Result added to evaluation table. Capture another distance/image to add more.");
});


