import { useState, useRef, useEffect, useCallback } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ThinkingStep {
  type: "status" | "tool_start" | "tool_done";
  message: string;
  timestamp: number;
}

const TOOL_LABELS: Record<string, string> = {
  ask_database: "Cortex Analyst",
  search_knowledge_base: "Cortex Search",
};

const THINKING_MESSAGES = [
  "Thinking...",
  "Analyzing your question...",
  "Determining which tools to use...",
  "Reasoning about the best approach...",
  "Processing...",
];

const SUGGESTIONS = [
  "How many tickets are there by priority?",
  "How do I reset my password?",
  "Which agent has the highest CSAT score?",
  "What are the API rate limits?",
];

function ThinkingIndicator({ steps, elapsed }: { steps: ThinkingStep[]; elapsed: number }) {
  const msgIndex = Math.floor(elapsed / 2000) % THINKING_MESSAGES.length;
  const hasToolSteps = steps.some((s) => s.type === "tool_start" || s.type === "tool_done");

  return (
    <div className="thinking-container">
      {!hasToolSteps && (
        <div className="thinking-step status" key={`thinking-${msgIndex}`}>
          <span className="tool-icon dot" />
          <span className="thinking-text">{THINKING_MESSAGES[msgIndex]}</span>
        </div>
      )}
      {steps
        .filter((s) => s.type !== "status")
        .map((step, i) => (
          <div
            key={i}
            className={`thinking-step ${step.type}`}
            style={{ animationDelay: `${i * 0.05}s` }}
          >
            {step.type === "tool_start" && <span className="tool-icon spinner" />}
            {step.type === "tool_done" && <span className="tool-icon check">✓</span>}
            <span className="thinking-text">{step.message}</span>
          </div>
        ))}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [runName, setRunName] = useState<string | null>(null);
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [streamedText, setStreamedText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [thinkingElapsed, setThinkingElapsed] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const thinkingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const hasMessages = messages.length > 0 || loading;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinkingSteps, streamedText, thinkingElapsed]);

  useEffect(() => {
    if (loading && !isStreaming) {
      setThinkingElapsed(0);
      thinkingTimerRef.current = setInterval(() => {
        setThinkingElapsed((prev) => prev + 500);
      }, 500);
    } else {
      if (thinkingTimerRef.current) {
        clearInterval(thinkingTimerRef.current);
        thinkingTimerRef.current = null;
      }
    }
    return () => {
      if (thinkingTimerRef.current) clearInterval(thinkingTimerRef.current);
    };
  }, [loading, isStreaming]);

  const addStep = useCallback((step: ThinkingStep) => {
    setThinkingSteps((prev) => [...prev, step]);
  }, []);

  const send = async (override?: string) => {
    const text = (override || input).trim();
    if (!text || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    setThinkingSteps([]);
    setStreamedText("");
    setIsStreaming(false);

    abortRef.current = new AbortController();

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, run_name: runName }),
        signal: abortRef.current.signal,
      });

      if (!res.ok || !res.body) throw new Error("Stream failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7);
          } else if (line.startsWith("data: ") && eventType) {
            try {
              const data = JSON.parse(line.slice(6));
              switch (eventType) {
                case "status":
                  if (data.run_name && !runName) setRunName(data.run_name);
                  break;
                case "tool_start": {
                  const label = TOOL_LABELS[data.tool] || data.tool;
                  addStep({ type: "tool_start", message: `Calling ${label}...`, timestamp: Date.now() });
                  break;
                }
                case "tool_done": {
                  setThinkingSteps((prev) => {
                    const updated = [...prev];
                    let idx = -1;
                    for (let j = updated.length - 1; j >= 0; j--) {
                      if (updated[j].type === "tool_start") { idx = j; break; }
                    }
                    if (idx !== -1) {
                      const label = updated[idx].message.replace("Calling ", "").replace("...", "");
                      updated[idx] = { type: "tool_done", message: `${label} returned results`, timestamp: Date.now() };
                    }
                    return updated;
                  });
                  break;
                }
                case "delta":
                  if (!isStreaming) {
                    setIsStreaming(true);
                  }
                  setStreamedText((prev) => prev + (data.text || ""));
                  break;
                case "response":
                  finalText = data.text;
                  break;
                case "done":
                  if (data.run_name && !runName) setRunName(data.run_name);
                  finalText = data.text || finalText;
                  break;
              }
            } catch {}
            eventType = "";
          }
        }
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: finalText || streamedText || "No response." },
      ]);
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "Error connecting to server." },
        ]);
      }
    } finally {
      setLoading(false);
      setThinkingSteps([]);
      setStreamedText("");
      setIsStreaming(false);
      abortRef.current = null;
    }
  };

  return (
    <div className={`app ${hasMessages ? "has-messages" : "empty"}`}>
      <div className="header">
        <div className="header-left">
          <div className="logo">SI</div>
          <h1>Support Intelligence</h1>
        </div>
      </div>

      {!hasMessages && (
        <div className="welcome">
          <div className="welcome-icon">?</div>
          <h2>What can I help you with?</h2>
          <p className="welcome-sub">Ask about support ticket metrics or search the knowledge base</p>
          <div className="suggestions">
            {SUGGESTIONS.map((s, i) => (
              <button key={i} className="suggestion" onClick={() => send(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {hasMessages && (
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              {m.role === "assistant" && <div className="avatar assistant-avatar">SI</div>}
              <div className={`bubble ${m.role}`}>{m.content}</div>
            </div>
          ))}
          {loading && !isStreaming && (
            <div className="message assistant">
              <div className="avatar assistant-avatar">SI</div>
              <ThinkingIndicator steps={thinkingSteps} elapsed={thinkingElapsed} />
            </div>
          )}
          {isStreaming && streamedText && (
            <div className="message assistant">
              <div className="avatar assistant-avatar">SI</div>
              <div className="bubble assistant streaming">{streamedText}<span className="cursor" /></div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      )}

      <div className={`input-area ${hasMessages ? "" : "centered"}`}>
        <div className="input-bar">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Ask a question..."
            disabled={loading}
          />
          <button onClick={() => send()} disabled={loading || !input.trim()} aria-label="Send">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
