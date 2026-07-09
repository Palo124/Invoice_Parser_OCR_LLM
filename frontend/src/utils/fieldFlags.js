export function buildFieldSeverityMap(validationErrors = []) {
  const map = {};
  for (const error of validationErrors) {
    const current = map[error.field];
    if (!current || error.severity === "error") {
      map[error.field] = error.severity;
    }
  }
  return map;
}

export function fieldClassName(path, severityMap) {
  const severity = severityMap[path];
  if (severity === "error") return "field-input field-error";
  if (severity === "warning") return "field-input field-warning";
  return "field-input";
}

export function fieldHint(path, validationErrors = []) {
  const matches = validationErrors.filter((error) => error.field === path);
  if (!matches.length) return null;
  return matches.map((error) => error.message).join(" ");
}
