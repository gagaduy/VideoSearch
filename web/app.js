const defaultApiBase = `${window.location.protocol}//${window.location.hostname}:8000`;
const apiBase = window.VIDEO_API_BASE || defaultApiBase;
const uploadForm = document.getElementById("upload-form");
const videoFileInput = document.getElementById("video-file");
const videoFilenameInput = document.getElementById("video-filename");
const searchForm = document.getElementById("search-form");
const questionSearchForm = document.getElementById("question-search-form");
const questionQueryInput = document.getElementById("question-query");
const videoQueryForm = document.getElementById("video-query-form");
const queryVideoInput = document.getElementById("query-video");
const queryVideoPreview = document.getElementById("query-video-preview");
const jobStatus = document.getElementById("job-status");
const jobStatusLabel = document.getElementById("job-status-label");
const jobStageLabel = document.getElementById("job-stage-label");
const jobProgressCount = document.getElementById("job-progress-count");
const jobProgressPercent = document.getElementById("job-progress-percent");
const jobProgressBar = document.getElementById("job-progress-bar");
const jobStatusNote = document.getElementById("job-status-note");
const jobOutput = document.getElementById("job-output");
const results = document.getElementById("results");
const timelinePreview = document.getElementById("timeline-preview");
const jobStatusPanel = document.getElementById("job-status-panel");
let textSearchInFlight = false;
let questionSearchInFlight = false;
let videoQueryInFlight = false;
let activeQueryVideoUrl = "";

function splitLabels(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatStage(stage) {
  if (!stage) {
    return "queued";
  }
  if (stage === "extracting_frames") {
    return "Extracting frames";
  }
  if (stage === "building_segments") {
    return "Building segments";
  }
  if (stage === "queued") {
    return "Queued";
  }
  if (stage === "processing") {
    return "Preparing job";
  }
  if (stage === "done") {
    return "Completed";
  }
  if (stage === "error") {
    return "Failed";
  }
  if (stage.startsWith("indexing:")) {
    const [, progress] = stage.split(":");
    return `Indexing ${progress}`;
  }
  if (stage.startsWith("embedding_frames:")) {
    const [, progress] = stage.split(":");
    return `Embedding frames ${progress}`;
  }
  if (stage.startsWith("enriching_segments:")) {
    const [, progress] = stage.split(":");
    return `Enriching segments ${progress}`;
  }
  return stage;
}

function progressForJob(job) {
  if (!job) {
    return 0;
  }
  if (job.status === "completed") {
    return 100;
  }
  if (job.status === "failed") {
    return 100;
  }
  if (job.stage === "queued") {
    return 8;
  }
  if (job.stage === "processing") {
    return 15;
  }
  if (job.stage === "extracting_frames") {
    return 30;
  }
  if (job.stage === "building_segments") {
    return 78;
  }
  if (job.stage && job.stage.startsWith("indexing:")) {
    const [, progress] = job.stage.split(":");
    const [current, total] = progress.split("/").map((value) => Number.parseInt(value, 10));
    if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
      return Math.max(35, Math.min(95, 35 + Math.round((current / total) * 60)));
    }
  }
  if (job.stage && job.stage.startsWith("embedding_frames:")) {
    const [, progress] = job.stage.split(":");
    const [current, total] = progress.split("/").map((value) => Number.parseInt(value, 10));
    if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
      return Math.max(18, Math.min(76, 18 + Math.round((current / total) * 58)));
    }
  }
  if (job.stage && job.stage.startsWith("enriching_segments:")) {
    const [, progress] = job.stage.split(":");
    const [current, total] = progress.split("/").map((value) => Number.parseInt(value, 10));
    if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
      return Math.max(82, Math.min(96, 82 + Math.round((current / total) * 14)));
    }
  }
  return 20;
}

function parseJobProgress(job) {
  const stage = String(job?.stage || "queued");
  const quantifiedStage = stage.match(/^(embedding_frames|enriching_segments|indexing):(\d+)\/(\d+)$/);
  const percent = progressForJob(job);
  if (!quantifiedStage) {
    return {
      stage,
      current: null,
      total: null,
      percent,
    };
  }
  return {
    stage: quantifiedStage[1],
    current: Number.parseInt(quantifiedStage[2], 10),
    total: Number.parseInt(quantifiedStage[3], 10),
    percent,
  };
}

function describeJobNote(job) {
  const stage = String(job?.stage || "queued");
  if (job?.status === "completed" || stage === "done") {
    return "Indexing finished. Search is ready to use.";
  }
  if (job?.status === "failed" || stage === "error") {
    return "Indexing failed. Check the latest job state and worker logs.";
  }
  if (stage === "queued") {
    return "Job is queued and waiting for the worker.";
  }
  if (stage === "processing") {
    return "Upload completed. Worker is preparing the indexing job.";
  }
  if (stage === "extracting_frames") {
    return "Worker is extracting candidate frames from the uploaded video.";
  }
  if (stage.startsWith("embedding_frames:")) {
    return "Worker is embedding extracted frames for segment building.";
  }
  if (stage === "building_segments") {
    return "Worker is grouping nearby frames into retrieval segments.";
  }
  if (stage.startsWith("enriching_segments:")) {
    return "Worker is enriching segment keyframes with InternVL, OCR, and object detection.";
  }
  return "Worker is updating the indexing job.";
}

function formatProgressCount(progress) {
  if (!Number.isFinite(progress.current) || !Number.isFinite(progress.total)) {
    return "stage-based";
  }
  return `${progress.current}/${progress.total}`;
}

function formatRetrievalScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  return numeric.toFixed(3);
}

function formatTimestamp(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function formatObjectCounts(counts) {
  if (!counts || typeof counts !== "object") {
    return "";
  }
  const entries = Object.entries(counts)
    .filter(([, value]) => Number.isFinite(value) && value > 0)
    .sort((left, right) => Number(right[1]) - Number(left[1]));
  if (!entries.length) {
    return "";
  }
  return entries.map(([label, count]) => `${label}×${count}`).join(", ");
}

function revokeActiveQueryVideoUrl() {
  if (activeQueryVideoUrl) {
    URL.revokeObjectURL(activeQueryVideoUrl);
    activeQueryVideoUrl = "";
  }
}

function renderEmptyState(message) {
  results.className = "results-pane";
  results.innerHTML = `<p class="empty-state">${message}</p>`;
  timelinePreview.textContent = "No result selected.";
}

function renderSearchResults(payload, emptyMessage) {
  renderResults(payload.results || [], emptyMessage);
}

function syncIdleJobStatus() {
  if (!jobStatusPanel || !jobStatus) {
    return;
  }
  if (jobStatus.hidden) {
    jobStatusPanel.classList.add("panel-idle");
  } else {
    jobStatusPanel.classList.remove("panel-idle");
  }
}

function renderJobStatus(job, message) {
  if (!jobStatus || !jobStatusLabel || !jobStageLabel || !jobProgressBar || !jobProgressCount || !jobProgressPercent || !jobStatusNote) {
    return;
  }
  const progress = parseJobProgress(job);
  jobStatus.hidden = false;
  jobStatusLabel.textContent = message;
  jobStageLabel.textContent = formatStage(job?.stage);
  jobProgressCount.textContent = formatProgressCount(progress);
  jobProgressPercent.textContent = `${progress.percent}%`;
  jobProgressBar.style.width = `${progress.percent}%`;
  jobStatusNote.textContent = describeJobNote(job);
  syncIdleJobStatus();
}

async function pollJob(jobId) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const response = await fetch(`${apiBase}/jobs/${jobId}`);
    const job = await response.json();
    renderJobStatus(job, job.status === "failed" ? "Indexing failed" : "Indexing in progress");
    if (job.status === "completed" || job.status === "failed") {
      return job;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }

  renderJobStatus({ status: "running", stage: "processing" }, "Polling timed out");
  if (jobStatusNote) {
    jobStatusNote.textContent = "Polling timed out while the worker may still be running. Refresh the page or query the job again.";
  }
  return null;
}

syncIdleJobStatus();

if (videoFileInput && videoFilenameInput) {
  videoFileInput.addEventListener("change", () => {
    const [file] = videoFileInput.files || [];
    videoFilenameInput.value = file ? file.name : "";
  });
}

if (uploadForm) {
  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submitButton = uploadForm.querySelector("button[type='submit']");
    const formData = new FormData(uploadForm);
    const file = formData.get("file");
    if (!(file instanceof File) || !file.size) {
      jobOutput.textContent = "Please choose a video file.";
      return;
    }
    if (submitButton instanceof HTMLButtonElement) {
      submitButton.disabled = true;
      submitButton.textContent = "Indexing...";
    }
    renderJobStatus({ status: "running", stage: "queued" }, "Uploading video");
    jobOutput.textContent = "";
    try {
      const response = await fetch(`${apiBase}/videos/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Upload failed.");
      }

      jobOutput.textContent = JSON.stringify(
        {
          ...data,
          message: "Upload complete. Indexing started.",
        },
        null,
        2,
      );
      renderJobStatus(data.job, "Indexing started");

      const finalJob = await pollJob(data.job.id);
      if (finalJob) {
        jobOutput.textContent = JSON.stringify(
        {
          ...data,
          job: finalJob,
          message: finalJob.status === "completed" ? "Indexing completed." : "Indexing failed.",
          },
          null,
          2,
        );
        renderJobStatus(finalJob, finalJob.status === "completed" ? "Indexing completed" : "Indexing failed");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed.";
      jobOutput.textContent = message;
      renderJobStatus({ status: "failed", stage: "error" }, message);
    } finally {
      if (submitButton instanceof HTMLButtonElement) {
        submitButton.disabled = false;
        submitButton.textContent = "Create Index Job";
      }
    }
  });
}

if (searchForm) {
  searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (textSearchInFlight) {
      return;
    }
    const formData = new FormData(searchForm);
    const payload = {
      query: formData.get("query"),
      object_labels: splitLabels(String(formData.get("object_labels") || "")),
    };
    const submitButton = searchForm.querySelector("button[type='submit']");
    textSearchInFlight = true;
    if (submitButton instanceof HTMLButtonElement) {
      submitButton.disabled = true;
      submitButton.textContent = "Searching...";
    }

    try {
      const response = await fetch(`${apiBase}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Search failed.");
      }
      renderSearchResults(data, "No strong matches found for this query.");
    } catch (error) {
      results.textContent = error instanceof Error ? error.message : "Search failed.";
      timelinePreview.textContent = "No result selected.";
    } finally {
      textSearchInFlight = false;
      if (submitButton instanceof HTMLButtonElement) {
        submitButton.disabled = false;
        submitButton.textContent = "Search";
      }
    }
  });
}

if (questionSearchForm) {
  questionSearchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (questionSearchInFlight) {
      return;
    }

    const question = String(questionQueryInput?.value || "").trim();
    if (!question) {
      renderEmptyState("Enter a question before searching for evidence frames.");
      return;
    }

    const submitButton = questionSearchForm.querySelector("button[type='submit']");
    questionSearchInFlight = true;
    if (submitButton instanceof HTMLButtonElement) {
      submitButton.disabled = true;
      submitButton.textContent = "Searching...";
    }

    try {
      const response = await fetch(`${apiBase}/search/question`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Question search failed.");
      }
      renderSearchResults(data, "No strong evidence frames found for this question.");
    } catch (error) {
      results.textContent = error instanceof Error ? error.message : "Question search failed.";
      timelinePreview.textContent = "No result selected.";
    } finally {
      questionSearchInFlight = false;
      if (submitButton instanceof HTMLButtonElement) {
        submitButton.disabled = false;
        submitButton.textContent = "Find Evidence Frames";
      }
    }
  });
}

if (queryVideoInput && queryVideoPreview) {
  queryVideoInput.addEventListener("change", () => {
    const [file] = queryVideoInput.files || [];
    revokeActiveQueryVideoUrl();
    if (!file) {
      queryVideoPreview.textContent = "No query clip selected.";
      return;
    }
    activeQueryVideoUrl = URL.createObjectURL(file);
    queryVideoPreview.innerHTML = `
      <video class="preview-video" controls muted playsinline preload="metadata">
        <source src="${activeQueryVideoUrl}" type="${file.type || "video/mp4"}">
      </video>
    `;
  });
}

if (videoQueryForm) {
  videoQueryForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (videoQueryInFlight) {
      return;
    }

    const formData = new FormData(videoQueryForm);
    const file = formData.get("file");
    if (!(file instanceof File) || !file.size) {
      renderEmptyState("Choose a short video clip before searching.");
      return;
    }

    const submitButton = videoQueryForm.querySelector("button[type='submit']");
    videoQueryInFlight = true;
    if (submitButton instanceof HTMLButtonElement) {
      submitButton.disabled = true;
      submitButton.textContent = "Searching...";
    }

    try {
      const response = await fetch(`${apiBase}/search/video-query`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Video clip search failed.");
      }
      renderSearchResults(data, "No strong matches found for this clip.");
    } catch (error) {
      results.textContent = error instanceof Error ? error.message : "Video clip search failed.";
      timelinePreview.textContent = "No result selected.";
    } finally {
      videoQueryInFlight = false;
      if (submitButton instanceof HTMLButtonElement) {
        submitButton.disabled = false;
        submitButton.textContent = "Find Similar Frames";
      }
    }
  });
}

function renderResults(items, emptyMessage = "No strong matches found for this query.") {
  results.innerHTML = "";
  if (!items.length) {
    renderEmptyState(emptyMessage);
    return;
  }

  results.className = "results-grid";
  for (const item of items) {
    const article = document.createElement("article");
    article.className = "result-card";
    article.innerHTML = `
      <img class="result-thumb" src="${apiBase}${item.thumb_url}" alt="${item.caption || `Frame ${item.frame_id}`}">
      <div class="result-body">
        <strong>Frame #${item.frame_id}</strong>
        <p>${item.caption || "No caption"}</p>
        <small>${formatTimestamp(item.start_timestamp_sec)}-${formatTimestamp(item.end_timestamp_sec)} | relative match=${formatRetrievalScore(item.score)}</small>
        ${formatObjectCounts(item.object_counts) ? `<div class="object-counts">${formatObjectCounts(item.object_counts)}</div>` : ""}
      </div>
    `;
    article.addEventListener("click", () => {
      timelinePreview.innerHTML = `
        <article class="preview-card">
          <video class="preview-video" controls autoplay muted playsinline preload="metadata" poster="${apiBase}${item.thumb_url}">
            <source src="${apiBase}${item.preview_url}" type="video/mp4">
          </video>
          <div class="preview-body">
            <strong>Frame #${item.frame_id}</strong>
            <p>${item.caption || "No caption"}</p>
            <small>${formatTimestamp(item.start_timestamp_sec)}-${formatTimestamp(item.end_timestamp_sec)}</small>
            ${formatObjectCounts(item.object_counts) ? `<div class="object-counts">${formatObjectCounts(item.object_counts)}</div>` : ""}
          </div>
        </article>
      `;
    });
    results.appendChild(article);
  }
}
