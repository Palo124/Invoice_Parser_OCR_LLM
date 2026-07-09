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
