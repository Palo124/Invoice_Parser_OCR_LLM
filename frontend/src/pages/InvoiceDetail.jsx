import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { cancelInvoice, fetchInvoice, redoInvoice } from "../api/client.js";
import PipelineStepsView from "../components/PipelineStepsView.jsx";
import ProcessingFeedback from "../components/ProcessingFeedback.jsx";
import { useElapsedSeconds, usePollWhen } from "../hooks/useProcessingTimers.js";

export default function InvoiceDetail() {
  const { id } = useParams();
  const [invoice, setInvoice] = useState(null);
  const [error, setError] = useState("");
  const [redoing, setRedoing] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await fetchInvoice(id);
        if (!cancelled) {
          setInvoice(data);
          setError("");
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [id]);

  const isProcessing = invoice?.status === "processing";
  const elapsedSeconds = useElapsedSeconds(isProcessing);
  const htmlPreviewUrl = invoice?.data ? `/api/invoices/${id}/html` : null;
  const progressStage = invoice?.metadata?.progress?.stage;
  const progressLabel = invoice?.metadata?.progress?.label;
  const liveSteps = invoice?.metadata?.steps;

  const pollInvoice = useCallback(async () => {
    try {
      const data = await fetchInvoice(id);
      setInvoice(data);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }, [id]);

  usePollWhen(isProcessing, pollInvoice);

  async function handleCancel() {
    if (cancelling || !isProcessing) return;

    setCancelling(true);
    setError("");

    try {
      const data = await cancelInvoice(id);
      setInvoice(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setCancelling(false);
    }
  }

  async function handleRedo() {
    if (redoing || isProcessing) return;

    setRedoing(true);
    setError("");

    try {
      const data = await redoInvoice(id);
      setInvoice(data);
    } catch (err) {
      setError(err.message);
      try {
        const data = await fetchInvoice(id);
        setInvoice(data);
      } catch {
        // ignore secondary fetch failure
      }
    } finally {
      setRedoing(false);
    }
  }

  if (error && !invoice) {
    return (
      <section className="card">
        <p className="error">{error}</p>
        <Link to="/">Back to list</Link>
      </section>
    );
  }

  if (!invoice) {
    return (
      <section className="card">
        <p>Loading...</p>
      </section>
    );
  }

  return (
    <section className="card">
      <p>
        <Link to="/">← Back to list</Link>
      </p>
      <div className="detail-header">
        <h2>{invoice.original_filename}</h2>
        <div className="detail-actions">
          {isProcessing ? (
            <button
              type="button"
              className="button-danger"
              onClick={handleCancel}
              disabled={cancelling}
            >
              {cancelling ? "Cancelling…" : "Cancel processing"}
            </button>
          ) : (
            <button
              type="button"
              className="button-secondary"
              onClick={handleRedo}
              disabled={redoing}
            >
              {redoing ? "Starting redo…" : "Redo extraction"}
            </button>
          )}
        </div>
      </div>
      <p>
        Status: <span className={`status ${invoice.status}`}>{invoice.status}</span>
      </p>

      {isProcessing && (
        <ProcessingFeedback
          active
          mode="server"
          elapsedSeconds={elapsedSeconds}
          filename={invoice.original_filename}
          progressStage={progressStage}
          progressLabel={progressLabel}
        />
      )}

      {!isProcessing && (
        <>
          <p>Extraction path: {invoice.extraction_path || "-"}</p>
          <p>
            Confidence:{" "}
            <span className={`confidence ${invoice.confidence || "unknown"}`}>
              {invoice.confidence || "-"}
            </span>
          </p>
          <p>Needs review: {invoice.needs_review ? "yes" : "no"}</p>
          {invoice.review_status && <p>Review status: {invoice.review_status}</p>}
          {invoice.metadata?.vision_used && <p>Vision fallback: used</p>}
          {invoice.metadata?.escalation_used && <p>Escalation model: used</p>}
        </>
      )}

      {!isProcessing && (invoice.flags?.length > 0 || invoice.flagged_fields?.length > 0) && (
        <section className="validation-panel">
          <h3>Validation flags</h3>
          {invoice.flags?.length > 0 && (
            <div className="flag-list">
              {invoice.flags.map((flag) => (
                <span key={flag} className="flag-chip">
                  {flag}
                </span>
              ))}
            </div>
          )}
          {invoice.flagged_fields?.length > 0 && (
            <table className="flagged-fields-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Flag</th>
                  <th>Message</th>
                </tr>
              </thead>
              <tbody>
                {invoice.flagged_fields.map((item) => (
                  <tr key={`${item.field}-${item.flag}`}>
                    <td>{item.field}</td>
                    <td>{item.flag}</td>
                    <td>{item.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {invoice.validation_errors?.length > 0 && (
            <>
              <h4>Validation errors</h4>
              <pre>{JSON.stringify(invoice.validation_errors, null, 2)}</pre>
            </>
          )}
        </section>
      )}

      {htmlPreviewUrl && !isProcessing && (
        <section className="html-preview-section">
          <div className="html-preview-header">
            <h3>Generated invoice (HTML)</h3>
            <a href={htmlPreviewUrl} target="_blank" rel="noreferrer">
              Open in new tab
            </a>
          </div>
          <iframe
            title="Invoice HTML preview"
            className="html-preview-frame"
            src={htmlPreviewUrl}
          />
        </section>
      )}

      <PipelineStepsView steps={liveSteps} live={isProcessing} />

      {invoice.metadata && !isProcessing && (
        <details className="step-block">
          <summary>
            <span className="step-title">Run metadata</span>
            <span className="step-subtitle">tokens, cost, models</span>
          </summary>
          <div className="step-body">
            <pre>{JSON.stringify(invoice.metadata, null, 2)}</pre>
          </div>
        </details>
      )}

      {error && <p className="error">{error}</p>}
      {invoice.error_message && <p className="error">{invoice.error_message}</p>}

      {invoice.data && !isProcessing && (
        <section className="final-result">
          <h3>Final invoice data (JSON)</h3>
          <pre>{JSON.stringify(invoice.data, null, 2)}</pre>
        </section>
      )}

      {!invoice.data && !isProcessing && <p>No extracted data.</p>}
    </section>
  );
}
