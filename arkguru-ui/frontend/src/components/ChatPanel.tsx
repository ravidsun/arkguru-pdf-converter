import type { ChatTurn } from "../types";

export function ChatPanel({
  turns,
  question,
  busy,
  error,
  extractiveHint,
  onQuestion,
  onSend,
}: {
  turns: ChatTurn[];
  question: string;
  busy: boolean;
  error: string | null;
  extractiveHint: boolean;
  onQuestion: (q: string) => void;
  onSend: () => void;
}) {
  return (
    <section className="card">
      <h2>Grounded chat</h2>
      <p className="lead">
        Uses Phase 3 <code>quickstart.retrieve</code> + <code>quickstart.answer</code> on
        the local store (hashing embedder is the smoke path). Ollama is optional —
        without it the answer is extractive from the top hit.
      </p>
      {extractiveHint && (
        <div className="banner">
          Ollama is not reachable. Answers are extractive citations from retrieved chunks.
        </div>
      )}
      <div className="chat">
        {turns.length === 0 && (
          <p className="lead">Ask about the sample handbook after Phase 1 → 3, e.g. PPE or error code E14.</p>
        )}
        {turns.map((t, i) => (
          <div key={`${i}-${t.question}`}>
            <div className="bubble q">
              <div className="who">You</div>
              {t.question}
            </div>
            <div className="bubble">
              <div className="who">arkguru{t.extractive ? " · extractive" : ""}</div>
              {t.answer}
              {t.sources.map((s, j) => (
                <div className="cite" key={`${i}-${j}`}>
                  <code>
                    {s.source_id || "source"}
                    {s.page != null ? ` p.${s.page}` : ""}
                    {s.chunk_index != null ? ` #${s.chunk_index}` : ""}
                  </code>
                  {s.section ? ` · ${s.section}` : ""}
                  <div>{s.text}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="field wide" style={{ marginTop: 12 }}>
        <label htmlFor="chat-q">Question</label>
        <textarea
          id="chat-q"
          value={question}
          onChange={(e) => onQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSend();
            }
          }}
        />
      </div>
      <div className="row">
        <button className="btn primary" type="button" disabled={busy || !question.trim()} onClick={onSend}>
          Ask
        </button>
      </div>
      {error && <div className="err">{error}</div>}
    </section>
  );
}
