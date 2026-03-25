const elements = {
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    previewContainer: document.getElementById("preview-container"),
    imagePreview: document.getElementById("image-preview"),
    fileName: document.getElementById("file-name"),
    fileSize: document.getElementById("file-size"),
    replaceBtn: document.getElementById("replace-btn"),
    clearBtn: document.getElementById("clear-btn"),
    analyzeBtn: document.getElementById("analyze-btn"),
    analyzeCanvasBtn: document.getElementById("analyze-canvas"),
    clearCanvasBtn: document.getElementById("clear-canvas"),
    canvas: document.getElementById("drawing-canvas"),
    tabs: Array.from(document.querySelectorAll(".tab")),
    uploadPanel: document.getElementById("upload-panel"),
    drawPanel: document.getElementById("draw-panel"),
    emptyState: document.getElementById("empty-state"),
    loading: document.getElementById("loading"),
    resultContent: document.getElementById("result-content"),
    expression: document.getElementById("res-expr"),
    answer: document.getElementById("res-ans"),
    lineResults: document.getElementById("line-results"),
    error: document.getElementById("res-error"),
    note: document.getElementById("res-note"),
    bboxImage: document.getElementById("bbox-img"),
    predictionSummary: document.getElementById("prediction-summary"),
};

const state = {
    mode: "upload",
    selectedFile: null,
    sourcePreviewUrl: "",
    drawing: false,
    lastX: 0,
    lastY: 0,
};

const canvasContext = elements.canvas.getContext("2d");
const CANVAS_BRUSH_SIZE = 12;

function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return "-";
    }

    const units = ["B", "KB", "MB"];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }

    return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function showState(name) {
    elements.emptyState.classList.toggle("hidden", name !== "empty");
    elements.loading.classList.toggle("hidden", name !== "loading");
    elements.resultContent.classList.toggle("hidden", name !== "result");
}

function resetMessages() {
    elements.error.textContent = "";
    elements.error.classList.add("hidden");
    elements.note.textContent = "";
    elements.note.classList.add("hidden");
}

function resetResults() {
    showState("empty");
    resetMessages();
    elements.expression.textContent = "N/A";
    elements.answer.textContent = "N/A";
    elements.lineResults.innerHTML = "";
    elements.lineResults.classList.add("hidden");
    elements.bboxImage.src = "";
    elements.predictionSummary.textContent = "Chua co du lieu";
}

function setMode(mode) {
    state.mode = mode;
    elements.tabs.forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.mode === mode);
    });
    elements.uploadPanel.classList.toggle("hidden", mode !== "upload");
    elements.drawPanel.classList.toggle("hidden", mode !== "draw");
}

function updatePreview(file, previewUrl) {
    state.selectedFile = file;
    state.sourcePreviewUrl = previewUrl;
    elements.imagePreview.src = previewUrl;
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatBytes(file.size);
    elements.dropZone.classList.add("hidden");
    elements.previewContainer.classList.remove("hidden");
}

function clearSelectedFile() {
    state.selectedFile = null;
    state.sourcePreviewUrl = "";
    elements.fileInput.value = "";
    elements.imagePreview.src = "";
    elements.fileName.textContent = "Chua chon anh";
    elements.fileSize.textContent = "-";
    elements.previewContainer.classList.add("hidden");
    elements.dropZone.classList.remove("hidden");
    resetResults();
}

function clearCanvas() {
    canvasContext.fillStyle = "#ffffff";
    canvasContext.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
    canvasContext.beginPath();
}

function getCanvasPoint(event) {
    const rect = elements.canvas.getBoundingClientRect();
    const scaleX = elements.canvas.width / rect.width;
    const scaleY = elements.canvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
    };
}

function startDrawing(event) {
    state.drawing = true;
    const point = getCanvasPoint(event);
    state.lastX = point.x;
    state.lastY = point.y;
}

function draw(event) {
    if (!state.drawing) {
        return;
    }

    event.preventDefault();
    const point = getCanvasPoint(event);
    canvasContext.strokeStyle = "#111111";
    canvasContext.lineWidth = CANVAS_BRUSH_SIZE;
    canvasContext.lineCap = "round";
    canvasContext.lineJoin = "round";
    canvasContext.beginPath();
    canvasContext.moveTo(state.lastX, state.lastY);
    canvasContext.lineTo(point.x, point.y);
    canvasContext.stroke();
    state.lastX = point.x;
    state.lastY = point.y;
}

function stopDrawing() {
    state.drawing = false;
    canvasContext.beginPath();
}

function buildPredictionSummary(characters = []) {
    if (!characters.length) {
        return "Khong co du lieu confidence";
    }

    const average = Math.round(
        (characters.reduce((sum, item) => sum + Number(item.conf || 0), 0) / characters.length) * 100
    );
    const lowConfidence = characters.filter((item) => Number(item.conf || 0) < 0.8).length;

    if (lowConfidence > 0) {
        return `${characters.length} ky tu - TB ${average}% - ${lowConfidence} ky tu do tin cay thap`;
    }

    return `${characters.length} ky tu - TB ${average}%`;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function renderLineResults(lines = []) {
    if (!Array.isArray(lines) || lines.length <= 1) {
        elements.lineResults.innerHTML = "";
        elements.lineResults.classList.add("hidden");
        return;
    }

    elements.lineResults.innerHTML = lines.map((line, index) => {
        const expression = escapeHtml(line.expression || "N/A");
        const result = escapeHtml(line.result || "N/A");
        const confidence = escapeHtml(buildPredictionSummary(line.characters || []));
        const errorBlock = line.error
            ? `<div class="line-card-error">${escapeHtml(line.error)}</div>`
            : "";

        return `
            <article class="line-card">
                <div class="line-card-head">
                    <strong>Dong ${index + 1}</strong>
                    <span>${confidence}</span>
                </div>
                <div class="line-card-row">
                    <span>Bieu thuc</span>
                    <strong>${expression}</strong>
                </div>
                <div class="line-card-row">
                    <span>Ket qua</span>
                    <strong>${result}</strong>
                </div>
                ${errorBlock}
            </article>
        `;
    }).join("");

    elements.lineResults.classList.remove("hidden");
}

function exportTrimmedCanvas(callback) {
    const { width, height } = elements.canvas;
    const imageData = canvasContext.getImageData(0, 0, width, height);
    const { data } = imageData;

    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            const index = (y * width + x) * 4;
            const r = data[index];
            const g = data[index + 1];
            const b = data[index + 2];
            const isInk = r < 245 || g < 245 || b < 245;

            if (!isInk) {
                continue;
            }

            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }
    }

    if (maxX < 0 || maxY < 0) {
        callback(null, null);
        return;
    }

    const padding = 24;
    const cropX = Math.max(0, minX - padding);
    const cropY = Math.max(0, minY - padding);
    const cropWidth = Math.min(width - cropX, maxX - minX + 1 + padding * 2);
    const cropHeight = Math.min(height - cropY, maxY - minY + 1 + padding * 2);

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = cropWidth;
    tempCanvas.height = cropHeight;
    const tempContext = tempCanvas.getContext("2d");
    tempContext.fillStyle = "#ffffff";
    tempContext.fillRect(0, 0, cropWidth, cropHeight);
    tempContext.drawImage(
        elements.canvas,
        cropX,
        cropY,
        cropWidth,
        cropHeight,
        0,
        0,
        cropWidth,
        cropHeight
    );

    tempCanvas.toBlob(
        (blob) => callback(blob, tempCanvas.toDataURL("image/png")),
        "image/png",
        1
    );
}

async function executeAnalysis(fileOrBlob, filename) {
    showState("loading");
    resetMessages();
    elements.analyzeBtn.disabled = true;
    elements.analyzeCanvasBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", fileOrBlob, filename);

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        showState("result");
        elements.expression.textContent = data.expression || "N/A";
        elements.answer.textContent = data.result || "N/A";
        renderLineResults(data.lines || []);
        elements.bboxImage.src = data.display_image || "";
        elements.predictionSummary.textContent = buildPredictionSummary(data.characters || []);

        if (!response.ok || data.error) {
            elements.error.textContent = data.error || "Khong the xu ly anh dau vao.";
            elements.error.classList.remove("hidden");
        }

        const lowConfidence = Array.isArray(data.characters)
            ? data.characters.filter((item) => Number(item.conf || 0) < 0.8).length
            : 0;

        if (lowConfidence > 0) {
            elements.note.textContent = `Model chua chac o ${lowConfidence} ky tu. Neu ket qua sai, hay viet tach ky tu hon hoac dung anh ro hon.`;
            elements.note.classList.remove("hidden");
        } else if (Array.isArray(data.lines) && data.lines.length > 1) {
            elements.note.textContent = `Anh co ${data.lines.length} dong, he thong dang phan tich tung dong rieng de giam lan ky tu giua cac dong.`;
            elements.note.classList.remove("hidden");
        } else if (!data.error) {
            elements.note.textContent = "Ket qua duoc tao tu pipeline segmentation -> classifier -> parser.";
            elements.note.classList.remove("hidden");
        }
    } catch (error) {
        showState("result");
        elements.error.textContent = "Khong ket noi duoc toi server Flask.";
        elements.error.classList.remove("hidden");
    } finally {
        elements.analyzeBtn.disabled = false;
        elements.analyzeCanvasBtn.disabled = false;
    }
}

function handleFiles(fileList) {
    const files = Array.from(fileList || []);
    const file = files.find((item) => item.type && item.type.startsWith("image/"));
    if (!file) {
        return;
    }

    setMode("upload");
    const reader = new FileReader();
    reader.onload = (event) => {
        updatePreview(file, event.target.result);
        resetResults();
    };
    reader.readAsDataURL(file);
}

elements.tabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
    });
});

["dragenter", "dragover"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, () => elements.dropZone.classList.add("dragover"));
});

["dragleave", "drop"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, () => elements.dropZone.classList.remove("dragover"));
});

elements.dropZone.addEventListener("drop", (event) => {
    handleFiles(event.dataTransfer.files);
});

elements.fileInput.addEventListener("change", (event) => {
    handleFiles(event.target.files);
});

elements.replaceBtn.addEventListener("click", () => {
    elements.fileInput.click();
});

elements.clearBtn.addEventListener("click", clearSelectedFile);

elements.analyzeBtn.addEventListener("click", () => {
    if (!state.selectedFile) {
        return;
    }

    executeAnalysis(state.selectedFile, state.selectedFile.name);
});

document.addEventListener("paste", (event) => {
    const items = Array.from(event.clipboardData?.items || []);
    const imageItem = items.find((item) => item.type.startsWith("image/"));
    if (!imageItem) {
        return;
    }

    const blob = imageItem.getAsFile();
    if (!blob) {
        return;
    }

    const pastedFile = new File([blob], "clipboard-image.png", { type: blob.type || "image/png" });
    handleFiles([pastedFile]);
});

elements.clearCanvasBtn.addEventListener("click", () => {
    clearCanvas();
    resetResults();
});

elements.canvas.addEventListener("pointerdown", (event) => {
    elements.canvas.setPointerCapture(event.pointerId);
    startDrawing(event);
});
elements.canvas.addEventListener("pointermove", draw);
elements.canvas.addEventListener("pointerup", stopDrawing);
elements.canvas.addEventListener("pointerleave", stopDrawing);
elements.canvas.addEventListener("pointercancel", stopDrawing);

elements.analyzeCanvasBtn.addEventListener("click", () => {
    exportTrimmedCanvas((blob) => {
        if (!blob) {
            showState("result");
            elements.error.textContent = "Canvas dang trong. Hay ve bieu thuc truoc khi phan tich.";
            elements.error.classList.remove("hidden");
            return;
        }

        executeAnalysis(blob, "canvas-expression.png");
    });
});

clearCanvas();
resetResults();
