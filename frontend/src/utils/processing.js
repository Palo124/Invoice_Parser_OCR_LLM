export const PROCESSING_STAGES = [
  "Uploading file to server",
  "Converting PDF and preparing pages",
  "Running OCR (Tesseract, PaddleOCR, EasyOCR)",
  "Extracting structured data with LLMs",
  "Merging results and saving to database",
];

export function formatElapsed(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function stageIndexForElapsed(seconds, uploadDone) {
  if (!uploadDone) return 0;
  if (seconds < 10) return 1;
  if (seconds < 40) return 2;
  if (seconds < 90) return 3;
  return 4;
}
