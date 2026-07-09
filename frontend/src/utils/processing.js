export function formatElapsed(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function formatDurationSeconds(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "-";
  const value = Number(seconds);
  if (value < 60) return `${value.toFixed(value < 10 ? 1 : 0)} s`;
  return formatElapsed(Math.round(value));
}

export function formatCost(amount) {
  if (amount == null || Number.isNaN(Number(amount))) return "-";
  const value = Number(amount);
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(4)}`;
}

export function formatStepMetrics(durationSeconds, estimatedCost) {
  const parts = [];
  if (durationSeconds != null) {
    parts.push(formatDurationSeconds(durationSeconds));
  }
  if (estimatedCost != null) {
    parts.push(formatCost(estimatedCost));
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

export function stageIndexFromStages(stages, progressStage) {
  if (!stages.length) return 0;
  if (!progressStage || progressStage === "queued") return 0;

  const index = stages.findIndex((stage) => stage.id === progressStage);
  if (index >= 0) return index;

  if (progressStage.startsWith("text:")) {
    return stages.findIndex((stage) => stage.id === "text:pymupdf");
  }
  if (progressStage.startsWith("llm:")) {
    if (progressStage === "llm:escalation") {
      return stages.findIndex((stage) => stage.id === "llm:escalation");
    }
    if (progressStage === "llm:vision") {
      return stages.findIndex((stage) => stage.id === "llm:vision");
    }
    return stages.findIndex((stage) => stage.id === "llm:deepseek");
  }
  if (progressStage === "validation") {
    return stages.findIndex((stage) => stage.id === "validation");
  }

  return 0;
}
