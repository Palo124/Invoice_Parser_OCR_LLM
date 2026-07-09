export function formatElapsed(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function stageIndexFromStages(stages, progressStage) {
  if (!stages.length) return 0;
  if (!progressStage || progressStage === "queued") return 0;

  const index = stages.findIndex((stage) => stage.id === progressStage);
  if (index >= 0) return index;

  if (progressStage.startsWith("ocr:")) {
    return stages.findIndex((stage) => stage.id === "ocr:tesseract");
  }
  if (progressStage.startsWith("text:")) {
    return stages.findIndex((stage) => stage.id === "text:pymupdf");
  }
  if (progressStage.startsWith("llm:")) {
    return stages.findIndex((stage) => stage.id === "llm:deepseek");
  }

  return 0;
}
