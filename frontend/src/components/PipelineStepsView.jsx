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

export default function PipelineStepsView({ steps, live = false }) {
  if (!steps || Object.keys(steps).length === 0) return null;

  return (
    <section className="steps-panel">
      <h3>{live ? "Live pipeline outputs" : "Pipeline step outputs"}</h3>

      {steps.text_extraction && (
        <StepBlock
          title="1. Text extraction"
          subtitle={`${steps.text_extraction.branch} · ${steps.text_extraction.char_count} characters`}
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
          >
            <h4>Raw model output</h4>
            <pre>{llmStep.raw_output || "(empty)"}</pre>
            <h4>Parsed JSON</h4>
            <pre>{JSON.stringify(llmStep.parsed_json, null, 2)}</pre>
          </StepBlock>
        ))}

      {steps.tmr?.merged_json && (
        <StepBlock title={`${steps.text_extraction ? "5" : "4"}. TMR merge`} subtitle="Final merged JSON">
          <pre>{JSON.stringify(steps.tmr.merged_json, null, 2)}</pre>
        </StepBlock>
      )}
    </section>
  );
}
