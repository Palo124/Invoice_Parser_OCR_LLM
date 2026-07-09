import {
  PROCESSING_STAGES,
  formatElapsed,
  stageIndexForElapsed,
} from "../utils/processing.js";

export default function ProcessingFeedback({
  active,
  uploadProgress,
  uploadDone,
  elapsedSeconds,
  filename,
}) {
  if (!active) return null;

  const stageIndex = stageIndexForElapsed(elapsedSeconds, uploadDone);

  return (
    <div className="processing-panel" role="status" aria-live="polite">
      <div className="processing-header">
        <div className="spinner" aria-hidden="true" />
        <div>
          <strong>Processing invoice</strong>
          {filename && <div className="processing-filename">{filename}</div>}
        </div>
      </div>

      {!uploadDone ? (
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
            Server is working — this usually takes 2–5 minutes.
          </p>
          <div className="progress-track">
            <div className="progress-bar indeterminate" />
          </div>
          <p className="processing-meta">
            Elapsed: {formatElapsed(elapsedSeconds)}
            {" · "}
            First run may take longer while OCR models load.
          </p>
        </>
      )}

      <ol className="processing-steps">
        {PROCESSING_STAGES.map((label, index) => {
          let state = "pending";
          if (index < stageIndex) state = "done";
          if (index === stageIndex) state = "active";

          return (
            <li key={label} className={`processing-step ${state}`}>
              <span className="processing-step-icon" aria-hidden="true">
                {state === "done" ? "✓" : state === "active" ? "…" : "○"}
              </span>
              {label}
            </li>
          );
        })}
      </ol>

      <p className="processing-note">
        Please keep this tab open. The page will redirect when processing
        finishes.
      </p>
    </div>
  );
}
