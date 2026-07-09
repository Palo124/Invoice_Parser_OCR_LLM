import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { deleteInvoice, fetchInvoices } from "../api/client.js";

const FILTERS = [
  { id: "all", label: "All" },
  { id: "needs_review", label: "Needs review" },
  { id: "approved", label: "Approved" },
];

function reviewLabel(invoice) {
  if (invoice.review_status === "approved") return "approved";
  if (invoice.review_status === "corrected") return "corrected";
  if (invoice.needs_review) return "needs review";
  return "ok";
}

export default function InvoiceList() {
  const [invoices, setInvoices] = useState([]);
  const [filter, setFilter] = useState("all");
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [deletingId, setDeletingId] = useState(null);

  async function loadInvoices(showRefresh = false) {
    if (showRefresh) setRefreshing(true);
    try {
      const data = await fetchInvoices(filter);
      setInvoices(data);
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadInvoices();
  }, [filter]);

  const hasProcessing = invoices.some((invoice) => invoice.status === "processing");

  useEffect(() => {
    if (!hasProcessing) return undefined;

    const timer = window.setInterval(() => {
      loadInvoices();
    }, 3000);

    return () => window.clearInterval(timer);
  }, [hasProcessing, filter]);

  async function handleDelete(invoice) {
    const label = invoice.invoice_number || invoice.original_filename || `#${invoice.id}`;
    const confirmed = window.confirm(
      `Delete invoice ${label}? This removes the record and uploaded file permanently.`,
    );
    if (!confirmed) return;

    setDeletingId(invoice.id);
    setError("");

    try {
      await deleteInvoice(invoice.id);
      setInvoices((current) => current.filter((item) => item.id !== invoice.id));
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <section className="card">
      <div className="list-header">
        <h2>Invoices</h2>
        <button type="button" onClick={() => loadInvoices(true)} disabled={refreshing}>
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      <div className="filter-tabs" role="tablist" aria-label="Invoice filters">
        {FILTERS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={filter === item.id}
            className={`filter-tab ${filter === item.id ? "active" : ""}`}
            onClick={() => setFilter(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {hasProcessing && (
        <p className="processing-banner" role="status">
          An invoice is still processing on the server. This list refreshes
          automatically every few seconds.
        </p>
      )}

      {error && <p className="error">{error}</p>}
      {!error && invoices.length === 0 && <p>No invoices in this view.</p>}
      {invoices.length > 0 && (
        <div className="table-wrap">
          <table className="invoice-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>File</th>
                <th>Invoice #</th>
                <th>Supplier</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Flags</th>
                <th>Review</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {invoices.map((invoice) => (
                <tr key={invoice.id}>
                  <td>
                    <Link to={`/invoices/${invoice.id}`}>{invoice.id}</Link>
                  </td>
                  <td className="cell-truncate" title={invoice.original_filename}>
                    {invoice.original_filename}
                  </td>
                  <td className="cell-truncate" title={invoice.invoice_number || undefined}>
                    {invoice.invoice_number || "-"}
                  </td>
                  <td className="cell-truncate" title={invoice.supplier_name || undefined}>
                    {invoice.supplier_name || "-"}
                  </td>
                  <td>
                    <span className={`status ${invoice.status}`}>{invoice.status}</span>
                  </td>
                  <td>{invoice.confidence || "-"}</td>
                  <td className="cell-wrap">
                    {invoice.flags?.length ? invoice.flags.join(", ") : "-"}
                  </td>
                  <td>
                    <span className={`review-badge ${reviewLabel(invoice).replace(" ", "-")}`}>
                      {reviewLabel(invoice)}
                    </span>
                  </td>
                  <td>
                    <button
                      type="button"
                      className="button-danger button-compact"
                      onClick={() => handleDelete(invoice)}
                      disabled={deletingId === invoice.id}
                    >
                      {deletingId === invoice.id ? "Deleting…" : "Delete"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
