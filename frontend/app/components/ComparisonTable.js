"use client";

import { IconChart, IconClose, IconCrown } from "./icons";

export default function ComparisonTable({ data, onClose }) {
  if (!data || !data.comparisons) return null;

  const formatPrice = (p) => {
    if (p === null || p === undefined) return "--";
    if (p >= 100) return `₹ ${(p / 100).toFixed(2)} Cr`;
    return `₹ ${p.toFixed(2)} L`;
  };

  return (
    <div className="animate-fade-in-up">
      <div className="bg-card border border-border rounded-2xl overflow-hidden shadow-lg">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border">
          <h3 className="flex items-center gap-2 font-semibold text-sm text-foreground">
            <IconChart className="w-4 h-4 text-primary" />
            Model Comparison
          </h3>
          <button
            onClick={onClose}
            className="text-muted hover:text-foreground cursor-pointer transition-colors p-1 rounded-lg hover:bg-card-hover"
          >
            <IconClose className="w-4 h-4" />
          </button>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-muted text-xs uppercase tracking-wider border-b border-border">
                <th className="text-left px-5 py-3 font-medium">Model</th>
                <th className="text-right px-5 py-3 font-medium">Predicted Price</th>
                <th className="text-right px-5 py-3 font-medium">R² Score</th>
                <th className="text-right px-5 py-3 font-medium">MSE</th>
                <th className="text-right px-5 py-3 font-medium">Features</th>
              </tr>
            </thead>
            <tbody>
              {data.comparisons.map((c, i) => (
                <tr
                  key={c.model}
                  className={`border-b border-border/50 transition-colors
                    ${c.is_best ? "bg-primary/3" : i % 2 === 0 ? "" : "bg-card-hover/20"}`}
                >
                  <td className="px-5 py-3.5 font-medium">
                    <span className="flex items-center gap-2">
                      {c.is_best && <IconCrown className="w-3.5 h-3.5 text-amber-500" />}
                      <span className={c.is_best ? "text-primary font-semibold" : ""}>
                        {c.name}
                      </span>
                    </span>
                  </td>
                  <td className={`text-right px-5 py-3.5 font-bold
                    ${c.is_best ? "gradient-text text-lg" : ""}`}>
                    {formatPrice(c.predicted_price)}
                  </td>
                  <td className="text-right px-5 py-3.5">
                    <span className={`px-2 py-0.5 rounded text-xs font-mono
                      ${c.is_best ? "bg-success/10 text-success font-semibold" : "text-muted"}`}>
                      {(c.r2 * 100).toFixed(2)}%
                    </span>
                  </td>
                  <td className="text-right px-5 py-3.5 text-muted font-mono text-xs">
                    {c.mse.toFixed(0)}
                  </td>
                  <td className="text-right px-5 py-3.5 text-muted text-xs">
                    {c.non_zero_features}/262
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Footer */}
        <div className="px-5 py-2.5 border-t border-border/50 text-xs text-muted flex flex-wrap gap-x-3 gap-y-1">
          <span>Input: {data.input.area} sqft</span>
          <span className="text-border">|</span>
          <span>{data.input.bedrooms} BHK</span>
          <span className="text-border">|</span>
          <span>{data.input.bathrooms} Bath</span>
          <span className="text-border">|</span>
          <span>{data.input.location}</span>
        </div>
      </div>
    </div>
  );
}
