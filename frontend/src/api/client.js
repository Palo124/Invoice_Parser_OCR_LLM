const API_BASE = "/api";

export async function fetchInvoices() {
  const res = await fetch(`${API_BASE}/invoices`);
  if (!res.ok) throw new Error("Failed to load invoices");
  return res.json();
}

export async function fetchInvoice(id) {
  const res = await fetch(`${API_BASE}/invoices/${id}`);
  if (!res.ok) throw new Error("Failed to load invoice");
  return res.json();
}

export async function uploadInvoice(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/invoices/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}
