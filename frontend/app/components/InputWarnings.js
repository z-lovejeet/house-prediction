"use client";

import { useState, useEffect, useMemo } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * InputWarnings — Smart validation warnings based on data distribution.
 */
export default function InputWarnings({ form }) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/data/stats`)
      .then((r) => r.json())
      .then((d) => setStats(d.distributions))
      .catch(() => {});
  }, []);

  const warnings = useMemo(() => {
    if (!stats) return [];
    const w = [];
    const area = Number(form.area);
    const beds = Number(form.bedrooms);
    const baths = Number(form.bathrooms);

    if (area && beds) {
      const sqftPerBhk = area / beds;
      if (sqftPerBhk < stats.sqft_per_bhk.typical_min) {
        w.push({
          level: "warning",
          text: `${beds} BHK in ${area} sqft is unusually cramped (${Math.round(sqftPerBhk)} sqft/room). Typical: ${stats.sqft_per_bhk.typical_min}-${stats.sqft_per_bhk.typical_max} sqft/room.`,
        });
      }
    }

    if (area && (area < stats.area.q25 * 0.5 || area > stats.area.q75 * 2)) {
      const note = area < stats.area.q25 * 0.5 ? "very small" : "very large";
      w.push({
        level: "info",
        text: `${area} sqft is ${note} compared to the dataset (median: ${stats.area.median} sqft). Predictions may be less accurate for outliers.`,
      });
    }

    if (beds && beds > stats.bedrooms.typical_max) {
      w.push({
        level: "info",
        text: `${beds} bedrooms is above the typical range (1-${stats.bedrooms.typical_max}). Only ${beds > 8 ? "very few" : "a few"} properties in the dataset have this many.`,
      });
    }

    if (baths && beds && baths > beds + 1) {
      w.push({
        level: "info",
        text: `${baths} bathrooms for ${beds} bedrooms is unusual. Typical is ${beds}-${beds + 1} bathrooms.`,
      });
    }

    return w;
  }, [form, stats]);

  if (warnings.length === 0) return null;

  return (
    <div className="space-y-1.5 animate-fade-in-up">
      {warnings.map((w, i) => (
        <div key={i} className={`flex items-start gap-2 px-3.5 py-2.5 rounded-lg text-xs leading-relaxed
          ${w.level === "warning"
            ? "bg-warning/5 border border-warning/15 text-warning"
            : "bg-accent/5 border border-accent/15 text-accent"
          }`}
        >
          <svg className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24"
            stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
          </svg>
          <span>{w.text}</span>
        </div>
      ))}
    </div>
  );
}
