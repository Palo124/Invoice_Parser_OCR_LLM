import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadInvoice } from "../api/client.js";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    setError("");
    try {
      const invoice = await uploadInvoice(file);
      navigate(`/invoices/${invoice.id}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Upload Invoice</h2>
      <p>Supported formats: PDF, PNG, JPEG</p>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".pdf,.png,.jpg,.jpeg"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
        <div style={{ marginTop: "1rem" }}>
          <button type="submit" disabled={!file || loading}>
            {loading ? "Processing..." : "Upload and Process"}
          </button>
        </div>
      </form>
      {error && <p className="error">{error}</p>}
    </section>
  );
}
