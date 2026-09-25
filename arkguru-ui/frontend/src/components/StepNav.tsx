import { STEPS, stepLabel, type Step } from "../types";

const TITLES: Record<Step, string> = {
  phase1: "Extract PDFs",
  phase2: "Crawl the web",
  phase3: "Combine + index",
  chat: "Ask with citations",
};

export function StepNav({ step, onChange }: { step: Step; onChange: (s: Step) => void }) {
  return (
    <nav className="stepper" aria-label="Pipeline phases">
      {STEPS.map((s) => (
        <button
          key={s}
          type="button"
          className={`step ${s === step ? "active" : ""}`}
          onClick={() => onChange(s)}
        >
          <span className="kicker">{stepLabel(s)}</span>
          <strong>{TITLES[s]}</strong>
        </button>
      ))}
    </nav>
  );
}
