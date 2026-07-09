import { useEffect, useState } from "react";
import { buildFieldSeverityMap, fieldClassName, fieldHint } from "../utils/fieldFlags.js";

function emptyParty() {
  return { name: "", ico: "", dic: "", address: "" };
}

function normalizeData(data) {
  const base = {
    invoice_number: "",
    invoice_date: "",
    tax_date: "",
    due_date: "",
    variable_symbol: "",
    specific_symbol: "",
    iban: "",
    swift: "",
    notes: "",
    contact: "",
    payment_method: "",
    supplier: emptyParty(),
    customer: emptyParty(),
    items: [],
    net_total: "",
    tax_total: "",
    gross_total: "",
    ...data,
  };

  return {
    ...base,
    supplier: { ...emptyParty(), ...(data?.supplier || {}) },
    customer: { ...emptyParty(), ...(data?.customer || {}) },
    items: Array.isArray(data?.items) ? data.items.map((item) => ({ ...item })) : [],
  };
}

function Field({ label, path, type = "text", value, onChange, severityMap, validationErrors }) {
  const hint = fieldHint(path, validationErrors);
  return (
    <label className="form-field">
      <span className="form-label">{label}</span>
      <input
        className={fieldClassName(path, severityMap)}
        type={type}
        value={value ?? ""}
        onChange={(event) => onChange(path, event.target.value)}
      />
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

function updateParty(form, partyKey, field, value) {
  return {
    ...form,
    [partyKey]: {
      ...form[partyKey],
      [field]: value,
    },
  };
}

function updateItem(form, index, field, value) {
  const items = [...form.items];
  const item = { ...items[index], [field]: value };
  if (field !== "index" && item.index == null) {
    item.index = index + 1;
  }
  items[index] = item;
  return { ...form, items };
}

function updateScalar(form, path, value) {
  if (path.includes(".")) {
    const [partyKey, field] = path.split(".");
    return updateParty(form, partyKey, field, value);
  }

  const numericFields = new Set(["net_total", "tax_total", "gross_total"]);
  const parsed =
    numericFields.has(path) && value !== "" ? Number(value) : value === "" ? null : value;
  return { ...form, [path]: parsed };
}

export default function InvoiceReviewForm({
  data,
  validationErrors = [],
  correctedFields = {},
  reviewedAt,
  reviewStatus,
  saving,
  approving,
  onSave,
  onApprove,
}) {
  const [form, setForm] = useState(() => normalizeData(data));

  useEffect(() => {
    setForm(normalizeData(data));
  }, [data]);

  const severityMap = buildFieldSeverityMap(validationErrors);
  const isApproved = reviewStatus === "approved";

  function handleChange(path, value) {
    setForm((current) => updateScalar(current, path, value));
  }

  function handleItemChange(index, field, value) {
    const numeric = new Set([
      "quantity",
      "unit_price",
      "tax_rate",
      "net_amount",
      "tax_amount",
      "gross_amount",
    ]);
    const parsed =
      numeric.has(field) && value !== "" ? Number(value) : value === "" ? null : value;
    setForm((current) => updateItem(current, index, field, parsed));
  }

  const correctionCount = Object.keys(correctedFields).length;

  return (
    <form
      className="review-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSave(form);
      }}
    >
      <div className="review-form-header">
        <h3>Review extracted data</h3>
        <p className="review-form-subtitle">
          Flagged fields are highlighted. Save corrections, then approve when ready.
        </p>
      </div>

      <section className="form-section">
        <h4>Invoice</h4>
        <div className="form-grid">
          <Field
            label="Invoice number"
            path="invoice_number"
            value={form.invoice_number}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Invoice date"
            path="invoice_date"
            value={form.invoice_date}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Tax date"
            path="tax_date"
            value={form.tax_date}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Due date"
            path="due_date"
            value={form.due_date}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Variable symbol"
            path="variable_symbol"
            value={form.variable_symbol}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Specific symbol"
            path="specific_symbol"
            value={form.specific_symbol}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="IBAN"
            path="iban"
            value={form.iban}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="SWIFT"
            path="swift"
            value={form.swift}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
        </div>
      </section>

      <section className="form-section">
        <h4>Supplier</h4>
        <div className="form-grid">
          <Field
            label="Name"
            path="supplier.name"
            value={form.supplier.name}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="IČO"
            path="supplier.ico"
            value={form.supplier.ico}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="DIČ"
            path="supplier.dic"
            value={form.supplier.dic}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Address"
            path="supplier.address"
            value={form.supplier.address}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
        </div>
      </section>

      <section className="form-section">
        <h4>Customer</h4>
        <div className="form-grid">
          <Field
            label="Name"
            path="customer.name"
            value={form.customer.name}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="IČO"
            path="customer.ico"
            value={form.customer.ico}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="DIČ"
            path="customer.dic"
            value={form.customer.dic}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Address"
            path="customer.address"
            value={form.customer.address}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
        </div>
      </section>

      {form.items.length > 0 && (
        <section className="form-section">
          <h4>Line items</h4>
          <div className="items-form-table-wrap">
            <table className="items-form-table">
              <thead>
                <tr>
                  <th>Description</th>
                  <th>Qty</th>
                  <th>Unit</th>
                  <th>Unit price</th>
                  <th>Net</th>
                  <th>Tax</th>
                  <th>Gross</th>
                </tr>
              </thead>
              <tbody>
                {form.items.map((item, index) => (
                  <tr key={index}>
                    {["description", "quantity", "unit", "unit_price", "net_amount", "tax_amount", "gross_amount"].map(
                      (field) => {
                        const path = `items.${index}.${field}`;
                        const hint = fieldHint(path, validationErrors);
                        const inputType = ["quantity", "unit_price", "net_amount", "tax_amount", "gross_amount"].includes(
                          field,
                        )
                          ? "number"
                          : "text";
                        return (
                          <td key={field}>
                            <input
                              className={fieldClassName(path, severityMap)}
                              type={inputType}
                              value={item[field] ?? ""}
                              onChange={(event) => handleItemChange(index, field, event.target.value)}
                            />
                            {hint && <span className="field-hint">{hint}</span>}
                          </td>
                        );
                      },
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="form-section">
        <h4>Totals</h4>
        <div className="form-grid">
          <Field
            label="Net total"
            path="net_total"
            type="number"
            value={form.net_total ?? ""}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Tax total"
            path="tax_total"
            type="number"
            value={form.tax_total ?? ""}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
          <Field
            label="Gross total"
            path="gross_total"
            type="number"
            value={form.gross_total ?? ""}
            onChange={handleChange}
            severityMap={severityMap}
            validationErrors={validationErrors}
          />
        </div>
      </section>

      {(correctionCount > 0 || reviewedAt) && (
        <section className="audit-panel">
          <h4>Audit trail</h4>
          {correctionCount > 0 && (
            <p>
              {correctionCount} field{correctionCount === 1 ? "" : "s"} corrected
              {reviewStatus === "corrected" ? " (saved)" : ""}.
            </p>
          )}
          {reviewedAt && <p>Approved at: {new Date(reviewedAt).toLocaleString()}</p>}
          {correctionCount > 0 && (
            <ul className="corrections-list">
              {Object.entries(correctedFields).map(([field, change]) => (
                <li key={field}>
                  <strong>{field}</strong>: {String(change.from)} → {String(change.to)}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {!isApproved && (
        <div className="review-form-actions">
          <button type="submit" disabled={saving || approving}>
            {saving ? "Saving…" : "Save corrections"}
          </button>
          <button
            type="button"
            className="button-secondary"
            onClick={onApprove}
            disabled={saving || approving}
          >
            {approving ? "Approving…" : "Approve invoice"}
          </button>
        </div>
      )}

      {isApproved && <p className="approved-banner">This invoice has been approved.</p>}
    </form>
  );
}
