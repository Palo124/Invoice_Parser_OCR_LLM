const API_BASE = "/api";

export async function fetchInvoices(filter) {
  const params = filter && filter !== "all" ? `?filter=${encodeURIComponent(filter)}` : "";
  const res = await fetch(`${API_BASE}/invoices${params}`);
  if (!res.ok) throw new Error("Failed to load invoices");
  return res.json();
}

export async function fetchInvoice(id) {
  const res = await fetch(`${API_BASE}/invoices/${id}`);
  if (!res.ok) throw new Error("Failed to load invoice");
  return res.json();
}

export async function cancelInvoice(id) {
  const res = await fetch(`${API_BASE}/invoices/${id}/cancel`, { method: "POST" });
  if (!res.ok) {
    let message = "Failed to cancel processing";
    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      message = `Failed to cancel processing (${res.status})`;
    }
    throw new Error(message);
  }
  return res.json();
}

export async function redoInvoice(id) {
  const res = await fetch(`${API_BASE}/invoices/${id}/redo`, { method: "POST" });
  if (!res.ok) {
    let message = "Failed to redo extraction";
    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      message = `Failed to redo extraction (${res.status})`;
    }
    throw new Error(message);
  }
  return res.json();
}

export async function updateInvoice(id, data) {
  const res = await fetch(`${API_BASE}/invoices/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  });
  if (!res.ok) {
    let message = "Failed to save corrections";
    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      message = `Failed to save corrections (${res.status})`;
    }
    throw new Error(message);
  }
  return res.json();
}

export async function approveInvoice(id) {
  const res = await fetch(`${API_BASE}/invoices/${id}/approve`, { method: "POST" });
  if (!res.ok) {
    let message = "Failed to approve invoice";
    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      message = `Failed to approve invoice (${res.status})`;
    }
    throw new Error(message);
  }
  return res.json();
}

export async function deleteInvoice(id) {
  const res = await fetch(`${API_BASE}/invoices/${id}`, { method: "DELETE" });
  if (!res.ok) {
    let message = "Failed to delete invoice";
    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      message = `Failed to delete invoice (${res.status})`;
    }
    throw new Error(message);
  }
}

let cachedPipelineStages = null;

export async function fetchPipelineStages() {
  if (cachedPipelineStages) {
    return cachedPipelineStages;
  }

  const res = await fetch(`${API_BASE}/pipeline/stages`);
  if (!res.ok) throw new Error("Failed to load pipeline stages");
  cachedPipelineStages = await res.json();
  return cachedPipelineStages;
}

export function uploadInvoice(file, callbacks = {}) {
  const { onUploadProgress, onUploadComplete } = callbacks;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    xhr.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable || !onUploadProgress) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      onUploadProgress(percent);
    });

    xhr.upload.addEventListener("loadend", () => {
      onUploadComplete?.();
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Invalid response from server"));
        }
        return;
      }

      try {
        const err = JSON.parse(xhr.responseText);
        reject(new Error(err.detail || "Upload failed"));
      } catch {
        reject(new Error(`Upload failed (${xhr.status})`));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Network error — is the backend running?"));
    });

    xhr.addEventListener("timeout", () => {
      reject(new Error("Request timed out — processing may still be running on the server"));
    });

    xhr.open("POST", `${API_BASE}/invoices/upload`);
    xhr.timeout = 0;
    xhr.send(form);
  });
}
