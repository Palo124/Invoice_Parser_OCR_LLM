function StepBlock({ title, subtitle, defaultOpen = false, children }) {
  return (
    <details className="step-block" open={defaultOpen}>
      <summary>
        <span className="step-title">{title}</span>
        {subtitle && <span className="step-subtitle">{subtitle}</span>}
      </summary>
      <div className="step-body">{children}</div>
    </details>
  );
}

export default function PipelineStepsView({ steps }) {
  if (!steps) return null;

  return (
    <section className="steps-panel">
      <h3>Pipeline step outputs</h3>

      {steps.preprocessing && (
        <StepBlock
          title="1. Preprocessing"
          subtitle={`${steps.preprocessing.page_count} page(s) · ${steps.preprocessing.source_type}`}
          defaultOpen
        >
          <pre>{JSON.stringify(steps.preprocessing, null, 2)}</pre>
        </StepBlock>
      )}

      {steps.ocr?.length > 0 &&
        steps.ocr.map((ocrStep) => (
          <StepBlock
            key={ocrStep.engine}
            title={`2. OCR — ${ocrStep.engine}`}
            subtitle={`${ocrStep.char_count} characters`}
          >
            <pre>{ocrStep.text || "(empty)"}</pre>
          </StepBlock>
        ))}

      {steps.llm?.length > 0 &&
        steps.llm.map((llmStep) => (
          <StepBlock
            key={`${llmStep.model}-${llmStep.ocr_engine}`}
            title={`3. LLM — ${llmStep.model}`}
            subtitle={`OCR input: ${llmStep.ocr_engine} · tokens: ${llmStep.prompt_tokens ?? "?"} + ${llmStep.completion_tokens ?? "?"}`}
          >
            <h4>Raw model output</h4>
            <pre>{llmStep.raw_output || "(empty)"}</pre>
            <h4>Parsed JSON</h4>
            <pre>{JSON.stringify(llmStep.parsed_json, null, 2)}</pre>
          </StepBlock>
        ))}

      {steps.tmr?.merged_json && (
        <StepBlock title="4. TMR merge" subtitle="Final merged JSON">
          <pre>{JSON.stringify(steps.tmr.merged_json, null, 2)}</pre>
        </StepBlock>
      )}
    </section>
  );
}
