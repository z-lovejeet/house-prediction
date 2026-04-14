"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { IconChart } from "./icons";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * SensitivitySliders — Shows real-time price delta when adjusting inputs.
 */
export default function SensitivitySliders({ baseResult, form, selectedModel }) {
  const [deltas, setDeltas] = useState(null);
  const [loading, setLoading] = useState(false);
  const timeoutRef = useRef(null);

  const basePrice = baseResult?.predicted_price;
  const area = Number(form.area) || 1500;
  const beds = Number(form.bedrooms) || 3;

  const fetchDelta = useCallback(async (param, value) => {
    const body = {
      area: Number(form.area), bedrooms: Number(form.bedrooms),
      bathrooms: Number(form.bathrooms), location: form.location,
      model: selectedModel,
    };
    body[param] = value;

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",  headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data.predicted_price;
    } catch { return null; }
  }, [form, selectedModel]);

  const runSensitivity = useCallback(async () => {
    if (!basePrice || !form.area || !form.location) return;
    setLoading(true);

    const [areaUp, areaDown, bedUp, bedDown] = await Promise.all([
      fetchDelta("area", area + 200),
      fetchDelta("area", Math.max(100, area - 200)),
      fetchDelta("bedrooms", beds + 1),
      fetchDelta("bedrooms", Math.max(1, beds - 1)),
    ]);

    setDeltas({
      areaUp:  areaUp  !== null ? areaUp - basePrice : null,
      areaDown: areaDown !== null ? areaDown - basePrice : null,
      bedUp:   bedUp   !== null ? bedUp - basePrice : null,
      bedDown: bedDown !== null ? bedDown - basePrice : null,
    });
    setLoading(false);
  }, [basePrice, area, beds, fetchDelta]);

  useEffect(() => {
    if (baseResult) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(runSensitivity, 300);
    }
    return () => clearTimeout(timeoutRef.current);
  }, [baseResult, runSensitivity]);

  if (!baseResult || !deltas) return null;

  const items = [
    { label: "+200 sqft", delta: deltas.areaUp,  desc: `${area} → ${area+200} sqft` },
    { label: "−200 sqft", delta: deltas.areaDown, desc: `${area} → ${Math.max(100,area-200)} sqft` },
    { label: "+1 bedroom", delta: deltas.bedUp,   desc: `${beds} → ${beds+1} BHK` },
    { label: "−1 bedroom", delta: deltas.bedDown, desc: `${beds} → ${Math.max(1,beds-1)} BHK` },
  ];

  return (
    <div className="bg-card border border-border rounded-xl p-4 animate-fade-in-up">
      <h3 className="flex items-center gap-2 text-xs font-semibold text-muted uppercase tracking-widest mb-3">
        <IconChart className="w-3.5 h-3.5 text-primary" /> Price Sensitivity
      </h3>
      {loading ? (
        <p className="text-xs text-muted py-2">Calculating...</p>
      ) : (
        <div className="grid grid-cols-2 gap-2.5">
          {items.map(({ label, delta, desc }) => (
            <div key={label} className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-background border border-border/50">
              <div>
                <p className="text-xs font-medium text-foreground/70">{label}</p>
                <p className="text-[10px] text-muted">{desc}</p>
              </div>
              <span className={`text-sm font-bold font-mono ${
                delta > 0 ? "text-success" : delta < 0 ? "text-error" : "text-muted"
              }`}>
                {delta !== null ? `${delta > 0 ? "+" : ""}₹${delta.toFixed(1)}L` : "--"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
