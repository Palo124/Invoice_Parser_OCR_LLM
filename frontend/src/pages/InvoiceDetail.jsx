import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchInvoice } from "../api/client.js";
import PipelineStepsView from "../components/PipelineStepsView.jsx";
import { formatElapsed } from "../utils/processing.js";

export default function InvoiceDetail() {
  const { id } = useParams();
  const [invoice, setInvoice] = useState(null);
  const [error, setError] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

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
  const htmlPreviewUrl = invoice?.data ? `/api/invoices/${id}/html` : null;

  useEffect(() => {
    if (!isProcessing) return undefined;

    const startedAt = Date.now();
    const elapsedTimer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);

    const pollTimer = window.setInterval(async () => {
      try {
        const data = await fetchInvoice(id);
        setInvoice(data);
      } catch (err) {
        setError(err.message);
      }
    }, 3000);

    return () => {
      window.clearInterval(elapsedTimer);
      window.clearInterval(pollTimer);
    };
  }, [id, isProcessing]);

  if (error) {
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
      <h2>{invoice.original_filename}</h2>
      <p>
        Status: <span className={`status ${invoice.status}`}>{invoice.status}</span>
      </p>

      {isProcessing && (
        <div className="processing-panel" role="status" aria-live="polite">
          <div className="processing-header">
            <div className="spinner" aria-hidden="true" />
            <div>
              <strong>Still processing on the server</strong>
              <div className="processing-meta">
                Elapsed: {formatElapsed(elapsedSeconds)} · checking for updates…
              </div>
            </div>
          </div>
          <p className="processing-note">
            OCR and LLM extraction can take several minutes. This page refreshes
            automatically.
          </p>
        </div>
      )}

      <p>Extraction path: {invoice.extraction_path || "-"}</p>
      <p>Confidence: {invoice.confidence || "-"}</p>
      <p>Needs review: {invoice.needs_review ? "yes" : "no"}</p>

      {htmlPreviewUrl && (
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

      <PipelineStepsView steps={invoice.metadata?.steps} />

      {invoice.metadata && (
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

      {invoice.error_message && <p className="error">{invoice.error_message}</p>}

      {invoice.data && (
        <section className="final-result">
          <h3>Final invoice data (JSON)</h3>
          <pre>{JSON.stringify(invoice.data, null, 2)}</pre>
        </section>
      )}

      {!invoice.data && !isProcessing && <p>No extracted data.</p>}
    </section>
  );
}
