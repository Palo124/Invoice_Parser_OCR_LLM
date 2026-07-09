import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchInvoices } from "../api/client.js";

export default function InvoiceList() {
  const [invoices, setInvoices] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchInvoices()
      .then(setInvoices)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <section className="card">
      <h2>Invoices</h2>
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
                <td>{invoice.needs_review ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
