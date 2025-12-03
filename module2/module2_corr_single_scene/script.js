let cvReady = false;
let sceneLoaded = false;

function setStatus(msg) {
  const statusMsg = document.getElementById("statusMsg");
  if (statusMsg) statusMsg.textContent = msg;
}

function initOpenCV() {
  if (typeof cv === "undefined") {
    // OpenCV.js not yet loaded, try again shortly
    setTimeout(initOpenCV, 100);
    return;
  }

  if (cvReady) return;

  if (cv.getBuildInformation) {
    // Runtime already initialized
    cvReady = true;
    setStatus("Status: OpenCV.js loaded. Please upload a scene image.");
  } else {
    // Normal async init path
    cv.onRuntimeInitialized = () => {
      cvReady = true;
      setStatus("Status: OpenCV.js loaded. Please upload a scene image.");
    };
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("sceneFile");
  const sceneImg = document.getElementById("sceneImg");

  if (fileInput && sceneImg) {
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const url = URL.createObjectURL(file);
      sceneImg.onload = () => {
        sceneLoaded = true;
        setStatus("Status: Scene image loaded. Ready to run matching.");
      };
      sceneImg.src = url;
    });
  }

  initOpenCV();
});

function runTemplateMatching() {
  if (!cvReady) {
    alert("OpenCV is still loading. Please wait a moment and try again.");
    return;
  }
  if (!sceneLoaded) {
    alert("Please upload a scene image first.");
    return;
  }

  setStatus("Running template matching…");

  setTimeout(() => {
    // ---- 1. LOAD SCENE ----
    const sceneImg = document.getElementById("sceneImg");
    let sceneMatFull = cv.imread(sceneImg);

    // ⭐ SCALE SCENE DOWN (this makes blur visually strong)
    let sceneMat = new cv.Mat();
    const TARGET_WIDTH = 800; // use 600 if you want even stronger blur

    if (sceneMatFull.cols > TARGET_WIDTH) {
      let scale = TARGET_WIDTH / sceneMatFull.cols;
      let newSize = new cv.Size(
        sceneMatFull.cols * scale,
        sceneMatFull.rows * scale
      );
      cv.resize(sceneMatFull, sceneMat, newSize, 0, 0, cv.INTER_AREA);
    } else {
      sceneMat = sceneMatFull.clone();
    }

    let sceneGray = new cv.Mat();
    cv.cvtColor(sceneMat, sceneGray, cv.COLOR_RGBA2GRAY);

    // ---- 2. TEMPLATE MATCHING ----
    const tplIds = [
      "tpl1","tpl2","tpl3","tpl4","tpl5",
      "tpl6","tpl7","tpl8","tpl9","tpl10"
    ];
    const detections = [];
    const matchThreshold = 0.4;

    tplIds.forEach((id, idx) => {
      const elem = document.getElementById(id);
      if (!elem || !elem.complete) return;

      let tpl = cv.imread(elem);
      let tplGray = new cv.Mat();
      cv.cvtColor(tpl, tplGray, cv.COLOR_RGBA2GRAY);

      if (sceneGray.cols < tplGray.cols || sceneGray.rows < tplGray.rows) {
        tpl.delete();
        tplGray.delete();
        return;
      }

      let result = new cv.Mat();
      let resultCols = sceneGray.cols - tplGray.cols + 1;
      let resultRows = sceneGray.rows - tplGray.rows + 1;
      result.create(resultRows, resultCols, cv.CV_32FC1);

      cv.matchTemplate(sceneGray, tplGray, result, cv.TM_CCORR_NORMED);
      let mm = cv.minMaxLoc(result);

      if (mm.maxVal >= matchThreshold) {
        detections.push({
          x: mm.maxLoc.x,
          y: mm.maxLoc.y,
          w: tplGray.cols,
          h: tplGray.rows,
          label: "Template " + (idx + 1)
        });
      }

      tpl.delete();
      tplGray.delete();
      result.delete();
    });

    // ---- 3. EXTREME BLUR ON DETECTED REGIONS (NO BOXES) ----
    detections.forEach(det => {
      let rect = new cv.Rect(det.x, det.y, det.w, det.h);
      let roi = sceneMat.roi(rect);

      let blurred = new cv.Mat();
      let extraBlur = new cv.Mat();

      // Massive double Gaussian blur
      cv.GaussianBlur(roi, blurred, new cv.Size(151, 151), 0, 0);
      cv.GaussianBlur(blurred, extraBlur, new cv.Size(151, 151), 0, 0);

      // Slight darkening so region stands out as "censored"
      let finalRegion = new cv.Mat();
      cv.addWeighted(extraBlur, 0.9, roi, 0.1, 0, finalRegion);

      // Put blurred region back
      finalRegion.copyTo(roi);

      // Cleanup
      blurred.delete();
      extraBlur.delete();
      finalRegion.delete();
      roi.delete();

  
    });

    // ---- 4. SHOW OUTPUT ----
    cv.imshow("resultCanvas", sceneMat);

    // Cleanup
    sceneMatFull.delete();
    sceneMat.delete();
    sceneGray.delete();

    setStatus(`${detections.length} region(s) blurred (no boxes, blur only).`);
  }, 150);
}
