"use client";

import { IconCrown } from "./icons";

export default function ModelSelector({ models, selected, onSelect }) {
  if (!models || models.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold text-muted uppercase tracking-widest">
        Select Model
      </h3>
      <div className="grid grid-cols-2 gap-3">
        {models.map((m) => {
          const isActive = selected === m.key;
          return (
            <button
              key={m.key}
              type="button"
              onClick={() => onSelect(m.key)}
              className={`relative p-4 rounded-xl border text-left transition-all cursor-pointer
                ${isActive
                  ? "border-primary bg-primary/5 ring-2 ring-primary/20 shadow-md"
                  : "border-border bg-card hover:border-primary/30 hover:bg-card-hover"
                }`}
            >
              {m.is_best && (
                <span className="absolute -top-2.5 -right-2.5 flex items-center gap-1 px-2 py-0.5
                                 bg-gradient-to-r from-amber-500 to-orange-500
                                 text-[10px] font-bold text-white rounded-full shadow-lg z-10">
                  <IconCrown className="w-3 h-3" /> BEST
                </span>
              )}

              <p className={`font-semibold text-sm ${isActive ? "text-primary" : "text-foreground"}`}>
                {m.name}
              </p>

              <p className="text-2xl font-bold mt-1.5 gradient-text">
                {(m.r2 * 100).toFixed(2)}%
              </p>
              <p className="text-[10px] text-muted -mt-0.5">R² Accuracy</p>

              <div className="mt-2.5 flex items-center gap-2 text-[11px] text-muted">
                <span>MSE {m.mse.toFixed(0)}</span>
                <span className="text-border">|</span>
                <span>{m.non_zero_features}/{m.total_features} features</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
