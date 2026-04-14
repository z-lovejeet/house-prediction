"use client";

import { useState, useEffect } from "react";
import { IconDatabase, IconClose } from "./icons";

const STORAGE_KEY = "house_prediction_history";
const MAX_HISTORY = 20;

function formatPrice(p) {
  if (p >= 100) return `₹${(p / 100).toFixed(2)} Cr`;
  return `₹${p.toFixed(2)} L`;
}

function formatTime(ts) {
  const d = new Date(ts);
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return "Just now";
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

export function addToHistory(entry) {
  if (typeof window === "undefined") return;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const list = raw ? JSON.parse(raw) : [];
    list.unshift({ ...entry, timestamp: Date.now() });
    if (list.length > MAX_HISTORY) list.length = MAX_HISTORY;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {}
}

export default function PredictionHistory({ onReuse }) {
  const [history, setHistory] = useState([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setHistory(JSON.parse(raw));
    } catch {}

    const handler = () => {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) setHistory(JSON.parse(raw));
      } catch {}
    };
    window.addEventListener("storage", handler);
    window.addEventListener("prediction_added", handler);
    return () => {
      window.removeEventListener("storage", handler);
      window.removeEventListener("prediction_added", handler);
    };
  }, []);

  const clearHistory = () => {
    localStorage.removeItem(STORAGE_KEY);
    setHistory([]);
  };

  if (history.length === 0 && !open) return null;

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3
                   hover:bg-card-hover transition-colors cursor-pointer"
      >
        <span className="flex items-center gap-2 text-xs font-semibold text-muted uppercase tracking-widest">
          <IconDatabase className="w-3.5 h-3.5 text-primary" />
          Prediction History
          <span className="px-1.5 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-bold">
            {history.length}
          </span>
        </span>
        <svg className={`w-4 h-4 text-muted transition-transform ${open ? "rotate-180" : ""}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
      </button>

      {open && (
        <div className="border-t border-border">
          {history.length === 0 ? (
            <p className="px-5 py-4 text-xs text-muted">No predictions yet.</p>
          ) : (
            <>
              <div className="max-h-64 overflow-y-auto">
                {history.map((h, i) => (
                  <div key={i}
                    className="flex items-center justify-between px-5 py-2.5 border-b border-border/50
                               hover:bg-card-hover/50 transition-colors group"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-foreground/80">
                          {formatPrice(h.predicted_price)}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/5 text-primary font-medium">
                          {h.model_used?.replace("linear", "Linear").replace("elasticnet", "ElasticNet")
                            .replace("ridge", "Ridge").replace("lasso", "Lasso")}
                        </span>
                      </div>
                      <p className="text-[11px] text-muted truncate mt-0.5">
                        {h.area} sqft · {h.bedrooms} BHK · {h.bathrooms} Bath · {h.location}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 ml-3">
                      <span className="text-[10px] text-muted/60">{formatTime(h.timestamp)}</span>
                      {onReuse && (
                        <button onClick={() => onReuse(h)}
                          className="opacity-0 group-hover:opacity-100 text-[10px] px-2 py-1
                                     rounded bg-primary/10 text-primary hover:bg-primary/20
                                     transition-all cursor-pointer font-medium">
                          Reuse
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className="px-5 py-2 border-t border-border/50">
                <button onClick={clearHistory}
                  className="text-[11px] text-error/60 hover:text-error cursor-pointer transition-colors">
                  Clear all history
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
