import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchInvoice } from "../api/client.js";

export default function InvoiceDetail() {
  const { id } = useParams();
  const [invoice, setInvoice] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchInvoice(id)
      .then(setInvoice)
      .catch((err) => setError(err.message));
  }, [id]);

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
      <p>Extraction path: {invoice.extraction_path || "-"}</p>
      <p>Confidence: {invoice.confidence || "-"}</p>
      <p>Needs review: {invoice.needs_review ? "yes" : "no"}</p>
      {invoice.metadata && (
        <div>
          <h3>Pipeline metadata</h3>
          <pre>{JSON.stringify(invoice.metadata, null, 2)}</pre>
        </div>
      )}
      {invoice.error_message && <p className="error">{invoice.error_message}</p>}
      {invoice.data ? (
        <pre>{JSON.stringify(invoice.data, null, 2)}</pre>
      ) : (
        <p>No extracted data.</p>
      )}
    </section>
  );
}
