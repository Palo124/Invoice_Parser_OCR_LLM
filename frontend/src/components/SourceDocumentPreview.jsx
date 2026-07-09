function isPdf(filename = "") {
  return filename.toLowerCase().endsWith(".pdf");
}

export default function SourceDocumentPreview({ invoiceId, filename }) {
  const sourcePreviewUrl = `/api/invoices/${invoiceId}/file`;

  return isPdf(filename) ? (
    <iframe
      title="Source PDF preview"
      className="source-preview-frame"
      src={sourcePreviewUrl}
    />
  ) : (
    <img
      className="source-preview-image"
      src={sourcePreviewUrl}
      alt={filename}
    />
  );
}
