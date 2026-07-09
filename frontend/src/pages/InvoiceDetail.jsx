import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  approveInvoice,
  cancelInvoice,
  deleteInvoice,
  fetchInvoice,
  redoInvoice,
  updateInvoice,
} from "../api/client.js";
import InvoiceReviewForm from "../components/InvoiceReviewForm.jsx";
import SourceDocumentPreview from "../components/SourceDocumentPreview.jsx";
import PipelineStepsView from "../components/PipelineStepsView.jsx";
import ProcessingCostSummary from "../components/ProcessingCostSummary.jsx";
import ProcessingFeedback from "../components/ProcessingFeedback.jsx";
import { useElapsedSeconds, usePollWhen } from "../hooks/useProcessingTimers.js";
import { formatCost, formatDurationSeconds } from "../utils/processing.js";

export default function InvoiceDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [invoice, setInvoice] = useState(null);
  const [error, setError] = useState("");
  const [redoing, setRedoing] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [approving, setApproving] = useState(false);

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
  const canReview = invoice?.status === "completed" && invoice?.data;
  const elapsedSeconds = useElapsedSeconds(isProcessing);
  const sourcePreviewUrl = canReview ? `/api/invoices/${id}/file` : null;
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

  async function handleSave(formData) {
    if (saving || approving || !canReview) return;

    setSaving(true);
    setError("");

    try {
      const data = await updateInvoice(id, formData);
      setInvoice(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleApprove() {
    if (saving || approving || !canReview) return;

    setApproving(true);
    setError("");

    try {
      const data = await approveInvoice(id);
      setInvoice(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setApproving(false);
    }
  }

  async function handleDelete() {
    if (deleting || !invoice) return;

    const label = invoice.invoice_number || invoice.original_filename || `#${id}`;
    const confirmed = window.confirm(
      `Delete invoice ${label}? This removes the record and uploaded file permanently.`,
    );
    if (!confirmed) return;

    setDeleting(true);
    setError("");

    try {
      await deleteInvoice(id);
      navigate("/");
    } catch (err) {
      setError(err.message);
      setDeleting(false);
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
    <div className={canReview ? "detail-page wide" : "detail-page"}>
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
                disabled={redoing || invoice.review_status === "approved"}
              >
                {redoing ? "Starting redo…" : "Redo extraction"}
              </button>
            )}
            <button
              type="button"
              className="button-danger"
              onClick={handleDelete}
              disabled={deleting}
            >
              {deleting ? "Deleting…" : "Delete invoice"}
            </button>
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
            metadata={invoice.metadata}
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
            <p>
              Review:{" "}
              {invoice.review_status === "approved"
                ? "approved"
                : invoice.needs_review
                  ? "needs review"
                  : "ok"}
            </p>
            {invoice.metadata?.vision_used && <p>Vision fallback: used</p>}
            {invoice.metadata?.escalation_used && <p>Escalation model: used</p>}
            {invoice.metadata?.estimated_cost != null && (
              <p>
                Processing cost: <strong>{formatCost(invoice.metadata.estimated_cost)}</strong>
                {invoice.metadata.total_duration_seconds != null && (
                  <> · {formatDurationSeconds(invoice.metadata.total_duration_seconds)} total</>
                )}
              </p>
            )}
          </>
        )}

        {error && <p className="error">{error}</p>}
        {invoice.error_message && <p className="error">{invoice.error_message}</p>}
      </section>

      {canReview && (
        <section className="card review-layout">
          <div className="review-source">
            <div className="review-source-header">
              <h3>Source document</h3>
              <a href={sourcePreviewUrl} target="_blank" rel="noreferrer">
                Open in new tab
              </a>
            </div>
            <SourceDocumentPreview
              invoiceId={id}
              filename={invoice.original_filename}
            />
          </div>

          <div className="review-editor">
            {!isProcessing && (invoice.flags?.length > 0 || invoice.flagged_fields?.length > 0) && (
              <section className="validation-panel compact">
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
              </section>
            )}

            <InvoiceReviewForm
              data={invoice.data}
              validationErrors={invoice.validation_errors}
              correctedFields={invoice.corrected_fields}
              reviewedAt={invoice.reviewed_at}
              reviewStatus={invoice.review_status}
              saving={saving}
              approving={approving}
              onSave={handleSave}
              onApprove={handleApprove}
            />
          </div>
        </section>
      )}

      <section className="card">
        <ProcessingCostSummary metadata={invoice.metadata} live={isProcessing} />
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

        {!canReview && !invoice.data && !isProcessing && <p>No extracted data.</p>}
      </section>
    </div>
  );
}
