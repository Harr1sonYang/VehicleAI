const labelNames = {
  charger_target: "Charger 目标",
  explorer_target: "Explorer / PIU 目标",
  durango_target: "Durango 目标",
  other: "其他车辆",
  unknown: "不确定",
  unusable: "不可用",
};

let state = null;
let currentIndex = Number(localStorage.getItem("vehicleai-review-index") || 0);
let actionPending = false;
const filters = {
  status: document.querySelector("#status-filter"),
  class_hint: document.querySelector("#hint-filter"),
  label: document.querySelector("#label-filter"),
  provider: document.querySelector("#provider-filter"),
  candidate_status: document.querySelector("#candidate-status-filter"),
  sort: document.querySelector("#sort-filter"),
};

function params(candidateId = null) {
  const query = new URLSearchParams({index: String(currentIndex)});
  for (const [key, element] of Object.entries(filters)) {
    if (element.value) query.set(key, element.value);
  }
  if (candidateId) query.set("candidate_id", candidateId);
  return query;
}

function fillOptions(select, values, formatter = value => value) {
  const oldValue = select.value;
  for (const value of values) {
    if (![...select.options].some(option => option.value === value)) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = formatter(value);
      select.append(option);
    }
  }
  select.value = oldValue;
}

function showValue(field, value) {
  document.querySelector(`[data-field="${field}"]`).textContent = value ?? "—";
}

function render() {
  const candidate = state.candidate;
  document.querySelector("#progress").textContent =
    `浏览 ${state.position}/${state.navigation_count} · 当前筛选 ${state.filtered_count} 张 · 已审核 ${state.reviewed_count}/${state.total_count} · 未审核 ${state.unreviewed_count}`;
  fillOptions(filters.class_hint, state.filters.class_hints);
  fillOptions(filters.provider, state.filters.providers);
  fillOptions(filters.label, state.filters.labels, value => labelNames[value] || value);

  const image = document.querySelector("#candidate-image");
  const imageLink = document.querySelector("#full-image-link");
  const empty = document.querySelector("#empty");
  const controls = document.querySelectorAll(".actions button, #previous, #skip, #next, #confirm-group, #split-track");
  controls.forEach(button => { button.disabled = !candidate; });
  if (!candidate) {
    image.removeAttribute("src");
    imageLink.removeAttribute("href");
    imageLink.hidden = true;
    empty.hidden = false;
    document.querySelector("#filename").textContent = "—";
    return;
  }

  empty.hidden = true;
  imageLink.hidden = false;
  image.src = candidate.image_url;
  imageLink.href = candidate.image_url;
  document.querySelector("#filename").textContent = candidate.filename;
  showValue("class_hint", candidate.class_hint);
  showValue("label", candidate.label ? (labelNames[candidate.label] || candidate.label) : "未审核");
  showValue("provider", candidate.provider);
  showValue("query", candidate.query);
  showValue("dimensions", candidate.width && candidate.height ? `${candidate.width} × ${candidate.height}` : null);
  showValue("source", candidate.source);
  showValue("source_id", candidate.source_id);
  showValue("source_video", candidate.source_video);
  showValue("frame_number", candidate.frame_number);
  showValue("track_id", candidate.track_id);
  showValue("vehicle_group_id", candidate.vehicle_group_id);
  showValue("group_track_ids", candidate.group_track_ids?.join(", "));
  showValue("association_status", candidate.association_status);
  showValue("suspected_duplicate_track_ids", candidate.suspected_duplicate_track_ids?.join(", "));
  showValue("candidate_status", candidate.candidate_status);
  showValue("candidate_filter_reasons", candidate.candidate_filter_reasons?.join(", "));
  showValue("ai_recommendation", candidate.ai_recommendation ? (labelNames[candidate.ai_recommendation] || candidate.ai_recommendation) : "无推荐");
  showValue("ai_similarity", candidate.ai_similarity == null ? null : `${candidate.ai_similarity.toFixed(3)}（非概率）`);
  showValue("ai_top3", candidate.ai_top3?.map(item => `${labelNames[item.label] || item.label}: ${item.score.toFixed(3)}`).join(" · "));
  const acceptAi = document.querySelector("#accept-ai");
  acceptAi.disabled = candidate.reviewed || !candidate.ai_recommendation || candidate.ai_recommendation === "unknown";
  const aiReferenceLink = document.querySelector("#ai-reference-link");
  const aiReferenceImage = document.querySelector("#ai-reference-image");
  aiReferenceLink.hidden = !candidate.ai_nearest_reference_url;
  if (candidate.ai_nearest_reference_url) {
    aiReferenceLink.href = candidate.ai_nearest_reference_url;
    aiReferenceImage.src = candidate.ai_nearest_reference_url;
  } else {
    aiReferenceLink.removeAttribute("href");
    aiReferenceImage.removeAttribute("src");
  }
  document.querySelector("#group-track-ids").value = candidate.group_track_ids?.join(", ") || candidate.track_id || "";
  const sourceLink = document.querySelector("#source-link");
  sourceLink.hidden = !candidate.source_page_url;
  if (candidate.source_page_url) sourceLink.href = candidate.source_page_url;
  else sourceLink.removeAttribute("href");
  if (state.next_image_url) {
    const preload = new Image();
    preload.src = state.next_image_url;
  }
}

async function updateTrackGroup(action) {
  if (!state?.candidate || actionPending) return;
  actionPending = true;
  try {
    const trackIds = document.querySelector("#group-track-ids").value
      .split(",").map(value => Number(value.trim())).filter(Number.isInteger);
    const response = await fetch("/api/track-groups", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({candidate_id: state.candidate.id, action, track_ids: trackIds}),
    });
    if (!response.ok) throw new Error(await response.text());
    document.querySelector("#message").textContent = action === "merge" ? "已确认车辆组合并。" : "已拆分当前轨迹。";
    await loadState(state.candidate.id);
  } finally {
    actionPending = false;
  }
}

async function loadState(candidateId = null) {
  const response = await fetch(`/api/state?${params(candidateId)}`);
  if (!response.ok) throw new Error(await response.text());
  state = await response.json();
  currentIndex = state.index;
  localStorage.setItem("vehicleai-review-index", String(currentIndex));
  render();
}

async function annotate(label, annotationMethod = "manual") {
  if (!state?.candidate || actionPending) return;
  actionPending = true;
  try {
    const currentCandidateId = state.candidate.id;
    let nextCandidateId = state.navigation.next_unreviewed_id || state.navigation.next_id;
    let broadenStatus = false;
    if (!nextCandidateId && state.navigation.fallback_next_id) {
      nextCandidateId = state.navigation.fallback_next_id;
      broadenStatus = true;
    }
    const response = await fetch("/api/annotations", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({candidate_id: state.candidate.id, label, annotation_method: annotationMethod}),
    });
    if (!response.ok) throw new Error(await response.text());
    document.querySelector("#message").textContent = `已保存：${labelNames[label]}`;
    if (broadenStatus || !nextCandidateId) filters.status.value = "all";
    await loadState(nextCandidateId || currentCandidateId);
  } finally {
    actionPending = false;
  }
}

function move(direction) {
  if (!state?.candidate) return;
  let candidateId = null;
  let fallbackId = null;
  if (direction === "previous") {
    candidateId = state.navigation.previous_id;
    fallbackId = state.navigation.fallback_previous_id;
  } else if (direction === "next") {
    candidateId = state.navigation.next_unreviewed_id || state.navigation.next_id;
    fallbackId = state.navigation.fallback_next_id;
    if (!state.navigation.next_unreviewed_id && candidateId) {
      document.querySelector("#message").textContent = "没有其他未审核图片，显示下一张候选。";
    }
  } else if (direction === "skip") {
    candidateId = state.navigation.next_id;
    fallbackId = state.navigation.fallback_next_id;
  }
  if (!candidateId && fallbackId) {
    filters.status.value = "all";
    candidateId = fallbackId;
    document.querySelector("#message").textContent = "已切换到“全部”，继续浏览当前筛选范围。";
  }
  if (!candidateId) {
    document.querySelector("#message").textContent = "没有其他可浏览的候选图片。";
    return;
  }
  loadState(candidateId).catch(showError);
}

function showError(error) {
  document.querySelector("#message").textContent = `错误：${error.message}`;
}

document.querySelectorAll(".actions button").forEach(button => {
  button.addEventListener("click", () => annotate(button.dataset.label).catch(showError));
});
document.querySelector("#previous").addEventListener("click", () => move("previous"));
document.querySelector("#next").addEventListener("click", () => move("next"));
document.querySelector("#skip").addEventListener("click", () => move("skip"));
document.querySelector("#confirm-group").addEventListener("click", () => updateTrackGroup("merge").catch(showError));
document.querySelector("#split-track").addEventListener("click", () => updateTrackGroup("split").catch(showError));
document.querySelector("#accept-ai").addEventListener("click", () => {
  if (state?.candidate?.ai_recommendation) annotate(state.candidate.ai_recommendation, "ai_accepted").catch(showError);
});
filters.label.addEventListener("change", () => {
  if (filters.label.value && filters.status.value === "unreviewed") {
    filters.status.value = "reviewed";
  }
  currentIndex = 0;
  loadState().catch(showError);
});
filters.status.addEventListener("change", () => {
  if (filters.status.value === "unreviewed" && filters.label.value) {
    filters.label.value = "";
  }
  currentIndex = 0;
  loadState().catch(showError);
});
[filters.class_hint, filters.provider, filters.candidate_status, filters.sort].forEach(element => element.addEventListener("change", () => {
  currentIndex = 0;
  loadState().catch(showError);
}));

const labelByCode = {
  Digit1: "charger_target", Numpad1: "charger_target",
  Digit2: "explorer_target", Numpad2: "explorer_target",
  Digit3: "durango_target", Numpad3: "durango_target",
  Digit4: "other", Numpad4: "other",
  Digit5: "unknown", Numpad5: "unknown",
  Digit6: "unusable", Numpad6: "unusable",
};

window.addEventListener("keydown", event => {
  if (event.ctrlKey || event.altKey || event.metaKey || event.repeat) return;
  const label = labelByCode[event.code];
  if (label) {
    event.preventDefault();
    annotate(label).catch(showError);
    return;
  }
  if (event.code === "KeyS") {
    event.preventDefault();
    move("skip");
  } else if (event.code === "ArrowLeft") {
    event.preventDefault();
    move("previous");
  } else if (event.code === "ArrowRight") {
    event.preventDefault();
    move("next");
  }
}, true);

loadState().catch(showError);
