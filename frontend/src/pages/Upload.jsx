import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadInvoice } from "../api/client.js";
import ProcessingFeedback from "../components/ProcessingFeedback.jsx";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadDone, setUploadDone] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading) return undefined;

    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);

    return () => window.clearInterval(timer);
  }, [loading]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    setUploadProgress(0);
    setUploadDone(false);
    setElapsedSeconds(0);
    setError("");

    try {
      const invoice = await uploadInvoice(file, {
        onUploadProgress: setUploadProgress,
        onUploadComplete: () => {
          setUploadDone(true);
          setUploadProgress(100);
        },
      });
      navigate(`/invoices/${invoice.id}`);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Upload Invoice</h2>
      <p>Supported formats: PDF, PNG, JPEG</p>

      {!loading && (
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept=".pdf,.png,.jpg,.jpeg"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
          <div style={{ marginTop: "1rem" }}>
            <button type="submit" disabled={!file}>
              Upload and Process
            </button>
          </div>
        </form>
      )}

      <ProcessingFeedback
        active={loading}
        uploadProgress={uploadProgress}
        uploadDone={uploadDone}
        elapsedSeconds={elapsedSeconds}
        filename={file?.name}
      />

      {error && <p className="error">{error}</p>}
    </section>
  );
}
