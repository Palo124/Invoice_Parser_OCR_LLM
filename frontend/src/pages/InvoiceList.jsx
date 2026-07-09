import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchInvoices } from "../api/client.js";

export default function InvoiceList() {
  const [invoices, setInvoices] = useState([]);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  async function loadInvoices(showRefresh = false) {
    if (showRefresh) setRefreshing(true);
    try {
      const data = await fetchInvoices();
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
  }, []);

  const hasProcessing = invoices.some((invoice) => invoice.status === "processing");

  useEffect(() => {
    if (!hasProcessing) return undefined;

    const timer = window.setInterval(() => {
      loadInvoices();
    }, 3000);

    return () => window.clearInterval(timer);
  }, [hasProcessing]);

  return (
    <section className="card">
      <div className="list-header">
        <h2>Invoices</h2>
        <button type="button" onClick={() => loadInvoices(true)} disabled={refreshing}>
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {hasProcessing && (
        <p className="processing-banner" role="status">
          An invoice is still processing on the server. This list refreshes
          automatically every few seconds.
        </p>
      )}

      {error && <p className="error">{error}</p>}
      {!error && invoices.length === 0 && <p>No invoices yet.</p>}
      {invoices.length > 0 && (
        <table>
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
            </tr>
          </thead>
          <tbody>
            {invoices.map((invoice) => (
              <tr key={invoice.id}>
                <td>
                  <Link to={`/invoices/${invoice.id}`}>{invoice.id}</Link>
                </td>
                <td>{invoice.original_filename}</td>
                <td>{invoice.invoice_number || "-"}</td>
                <td>{invoice.supplier_name || "-"}</td>
                <td>
                  <span className={`status ${invoice.status}`}>{invoice.status}</span>
                </td>
                <td>{invoice.confidence || "-"}</td>
                <td>{invoice.flags?.length ? invoice.flags.join(", ") : "-"}</td>
                <td>{invoice.needs_review ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
