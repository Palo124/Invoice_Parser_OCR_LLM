import { stepMetricsSubtitle } from "./ProcessingCostSummary.jsx";

function StepBlock({ title, subtitle, metricsSubtitle, defaultOpen = false, children }) {
  const combinedSubtitle = [subtitle, metricsSubtitle].filter(Boolean).join(" · ");

  return (
    <details className="step-block" open={defaultOpen}>
      <summary>
        <span className="step-title">{title}</span>
        {combinedSubtitle && <span className="step-subtitle">{combinedSubtitle}</span>}
      </summary>
      <div className="step-body">{children}</div>
    </details>
  );
}

export default function PipelineStepsView({ steps, live = false }) {
  if (!steps || Object.keys(steps).length === 0) return null;

  return (
    <section className="steps-panel">
      <h3>{live ? "Live pipeline outputs" : "Pipeline step outputs"}</h3>

      {steps.text_extraction && (
        <StepBlock
          title="1. Text extraction"
          subtitle={`${steps.text_extraction.branch} · ${steps.text_extraction.char_count} characters`}
          metricsSubtitle={stepMetricsSubtitle(steps.text_extraction)}
          defaultOpen={live}
        >
          <pre>{JSON.stringify(steps.text_extraction, null, 2)}</pre>
          {steps.text_extraction.preview && (
            <>
              <h4>Text preview</h4>
              <pre>{steps.text_extraction.preview}</pre>
            </>
          )}
        </StepBlock>
      )}

      {steps.preprocessing && (
        <StepBlock
          title={steps.text_extraction ? "2. Preprocessing" : "1. Preprocessing"}
          subtitle={`${steps.preprocessing.page_count} page(s) · ${steps.preprocessing.source_type}`}
          metricsSubtitle={stepMetricsSubtitle(steps.preprocessing)}
          defaultOpen={live}
        >
          <pre>{JSON.stringify(steps.preprocessing, null, 2)}</pre>
        </StepBlock>
      )}

      {steps.ocr?.length > 0 &&
        steps.ocr.map((ocrStep) => (
          <StepBlock
            key={ocrStep.engine}
            title={`${steps.text_extraction ? "3" : "2"}. OCR — ${ocrStep.engine}`}
            subtitle={`${ocrStep.char_count} characters`}
            metricsSubtitle={stepMetricsSubtitle(ocrStep)}
          >
            <pre>{ocrStep.text || "(empty)"}</pre>
          </StepBlock>
        ))}

      {steps.llm?.length > 0 &&
        steps.llm.map((llmStep) => (
          <StepBlock
            key={`${llmStep.model}-${llmStep.ocr_engine}`}
            title={`${steps.text_extraction ? "4" : "3"}. LLM — ${llmStep.model}`}
            subtitle={`OCR input: ${llmStep.ocr_engine} · tokens: ${llmStep.prompt_tokens ?? "?"} + ${llmStep.completion_tokens ?? "?"}`}
            metricsSubtitle={stepMetricsSubtitle(llmStep)}
          >
            <h4>Raw model output</h4>
            <pre>{llmStep.raw_output || "(empty)"}</pre>
            <h4>Parsed JSON</h4>
            <pre>{JSON.stringify(llmStep.parsed_json, null, 2)}</pre>
          </StepBlock>
        ))}

      {steps.vision && (
        <StepBlock
          title="Vision extraction"
          subtitle={`${steps.vision.model} · ${steps.vision.page_count} page(s)`}
          metricsSubtitle={stepMetricsSubtitle(steps.vision)}
        >
          <pre>{JSON.stringify(steps.vision, null, 2)}</pre>
        </StepBlock>
      )}

      {steps.vision_merge && (
        <StepBlock
          title="Vision merge"
          subtitle={`${steps.vision_merge.merged_fields?.length || 0} field(s) from vision`}
        >
          <pre>{JSON.stringify(steps.vision_merge, null, 2)}</pre>
        </StepBlock>
      )}

      {steps.escalation && (
        <StepBlock
          title="Escalation extraction"
          subtitle={`${steps.escalation.model} · ${steps.escalation.triggers?.join(", ") || "triggered"}`}
          metricsSubtitle={stepMetricsSubtitle(steps.escalation)}
        >
          <pre>{JSON.stringify(steps.escalation, null, 2)}</pre>
        </StepBlock>
      )}

      {steps.escalation_merge && (
        <StepBlock
          title="Escalation merge"
          subtitle={`${steps.escalation_merge.applied_fields?.length || 0} field(s) overridden`}
        >
          <pre>{JSON.stringify(steps.escalation_merge, null, 2)}</pre>
        </StepBlock>
      )}

      {steps.tmr?.merged_json && (
        <StepBlock
          title={`${steps.text_extraction ? "5" : "4"}. TMR merge`}
          subtitle={
            steps.tmr.disagreements?.length
              ? `${steps.tmr.disagreements.length} field disagreement(s)`
              : "Final merged JSON"
          }
        >
          {steps.tmr.disagreements?.length > 0 && (
            <>
              <h4>Field disagreements</h4>
              <pre>{JSON.stringify(steps.tmr.disagreements, null, 2)}</pre>
            </>
          )}
          <h4>Merged JSON</h4>
          <pre>{JSON.stringify(steps.tmr.merged_json, null, 2)}</pre>
        </StepBlock>
      )}

      {steps.validation && (
        <StepBlock
          title="Validation"
          subtitle={`confidence: ${steps.validation.confidence || "?"} · review: ${steps.validation.needs_review ? "yes" : "no"}`}
          metricsSubtitle={stepMetricsSubtitle(steps.validation)}
          defaultOpen={!live}
        >
          <pre>{JSON.stringify(steps.validation, null, 2)}</pre>
        </StepBlock>
      )}
    </section>
  );
}
