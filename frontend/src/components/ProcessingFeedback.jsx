import {
  formatCost,
  formatDurationSeconds,
  formatElapsed,
  stageIndexFromStages,
} from "../utils/processing.js";
import { usePipelineStages } from "../hooks/usePipelineStages.js";

export default function ProcessingFeedback({
  active,
  mode = "upload",
  uploadProgress = 0,
  uploadDone = false,
  elapsedSeconds = 0,
  filename,
  progressStage,
  progressLabel,
  metadata,
}) {
  const pipelineStages = usePipelineStages();

  if (!active) return null;

  const isUpload = mode === "upload";
  const isServer = mode === "server";
  const stageIndex = isServer
    ? stageIndexFromStages(pipelineStages, progressStage)
    : 0;
  const currentLabel = progressLabel || "Processing on server…";

  return (
    <div className="processing-panel" role="status" aria-live="polite">
      <div className="processing-header">
        <div className="spinner" aria-hidden="true" />
        <div>
          <strong>{isUpload ? "Uploading invoice" : "Processing invoice"}</strong>
          {filename && <div className="processing-filename">{filename}</div>}
        </div>
      </div>

      {isUpload && !uploadDone ? (
        <>
          <p className="processing-lead">Uploading file…</p>
          <div className="progress-track">
            <div
              className="progress-bar determinate"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p className="processing-meta">{uploadProgress}% uploaded</p>
        </>
      ) : (
        <>
          <p className="processing-lead">
            {isUpload
              ? "Upload complete — opening invoice detail…"
              : `Current step: ${currentLabel}`}
          </p>
          {isServer && (
            <div className="progress-track">
              <div className="progress-bar indeterminate" />
            </div>
          )}
          <p className="processing-meta">
            {isServer && (
              <>
                Elapsed: {formatElapsed(elapsedSeconds)}
                {metadata?.total_duration_seconds != null && (
                  <>
                    {" · "}
                    Server time: {formatDurationSeconds(metadata.total_duration_seconds)}
                  </>
                )}
                {metadata?.estimated_cost != null && (
                  <>
                    {" · "}
                    Cost so far: {formatCost(metadata.estimated_cost)}
                  </>
                )}
                {" · "}
                Step outputs appear below as each stage finishes.
              </>
            )}
          </p>
        </>
      )}

      {isServer && pipelineStages.length > 0 && (
        <ol className="processing-steps">
          {pipelineStages.map((stage, index) => {
            let state = "pending";
            if (index < stageIndex) state = "done";
            if (index === stageIndex) state = "active";

            return (
              <li key={stage.id} className={`processing-step ${state}`}>
                <span className="processing-step-icon" aria-hidden="true">
                  {state === "done" ? "✓" : state === "active" ? "…" : "○"}
                </span>
                {stage.label}
              </li>
            );
          })}
        </ol>
      )}

      <p className="processing-note">
        {isUpload
          ? "Processing continues on the server after upload finishes."
          : "This page refreshes automatically while extraction runs."}
      </p>
    </div>
  );
}
