import {
  formatCost,
  formatDurationSeconds,
  formatStepMetrics,
} from "../utils/processing.js";

const STAGE_LABELS = {
  text_extraction: "Text extraction",
  llm: "LLM extraction",
  vision: "Vision extraction",
  escalation: "Escalation",
  validation: "Validation",
};

function stageLabel(metric) {
  if (metric.model) {
    return `${STAGE_LABELS[metric.stage] || metric.stage} (${metric.model})`;
  }
  return STAGE_LABELS[metric.stage] || metric.stage;
}

export default function ProcessingCostSummary({ metadata, live = false }) {
  if (!metadata) return null;

  const {
    total_duration_seconds: totalDuration,
    estimated_cost: totalCost,
    step_metrics: stepMetrics = [],
    token_usage: tokenUsage,
  } = metadata;

  const hasTotals = totalDuration != null || totalCost != null;
  const hasSteps = stepMetrics.length > 0;

  if (!hasTotals && !hasSteps) return null;

  return (
    <section className="processing-cost-summary">
      <div className="processing-cost-header">
        <h3>{live ? "Processing cost (so far)" : "Processing cost"}</h3>
        {hasTotals && (
          <div className="processing-cost-totals">
            {totalDuration != null && (
              <span>Total time: {formatDurationSeconds(totalDuration)}</span>
            )}
            {totalCost != null && <span>Total cost: {formatCost(totalCost)}</span>}
            {tokenUsage?.total_tokens != null && (
              <span>Tokens: {tokenUsage.total_tokens}</span>
            )}
          </div>
        )}
      </div>

      {hasSteps && (
        <table className="processing-cost-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Time</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {stepMetrics.map((metric, index) => (
              <tr key={`${metric.stage}-${metric.model || index}`}>
                <td>{stageLabel(metric)}</td>
                <td>{formatDurationSeconds(metric.duration_seconds)}</td>
                <td>{formatCost(metric.estimated_cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export function stepMetricsSubtitle(step) {
  return formatStepMetrics(step?.duration_seconds, step?.estimated_cost);
}
