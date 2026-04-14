"use client";

import { useState, useEffect } from "react";
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import { IconBrain, IconSpinner } from "./icons";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * ExplainabilityChart — Waterfall showing how each feature contributes to the final price.
 */
export default function ExplainabilityChart({ result, form, selectedModel }) {
  const [explain, setExplain] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!result) { setExplain(null); return; }

    const fetchExplain = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_URL}/explain`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            area: Number(form.area), bedrooms: Number(form.bedrooms),
            bathrooms: Number(form.bathrooms), balcony: form.balcony ? Number(form.balcony) : 1,
            location: form.location,
            model: selectedModel,
          }),
        });
        if (res.ok) setExplain(await res.json());
      } catch {}
      setLoading(false);
    };
    fetchExplain();
  }, [result, form, selectedModel]);

  if (!result) return null;

  if (loading) {
    return (
      <div className="bg-card border border-border rounded-xl p-5 flex items-center gap-2 text-sm text-muted">
        <IconSpinner className="w-4 h-4" /> Analyzing feature contributions...
      </div>
    );
  }

  if (!explain?.breakdown) return null;

  const items = explain.breakdown;

  const chartData = {
    labels: items.map((b) => b.feature),
    datasets: [{
      data: items.map((b) => b.contribution),
      backgroundColor: items.map((b) =>
        b.direction === "positive" ? "rgba(16, 185, 129, 0.65)" : "rgba(239, 68, 68, 0.65)"
      ),
      borderColor: items.map((b) =>
        b.direction === "positive" ? "#10b981" : "#ef4444"
      ),
      borderWidth: 2,
      borderRadius: 4,
      borderSkipped: false,
    }],
  };

  const chartOptions = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `₹ ${ctx.parsed.x > 0 ? "+" : ""}${ctx.parsed.x.toFixed(2)} Lakhs`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "rgba(148,163,184,0.7)", font: { size: 11 },
          callback: (v) => (v > 0 ? "+" : "") + v },
        grid: { color: "rgba(148,163,184,0.08)" },
      },
      y: {
        ticks: { color: "rgba(148,163,184,0.8)", font: { size: 11 } },
        grid: { display: false },
      },
    },
  };

  return (
    <div className="bg-card border border-border rounded-xl p-5 animate-fade-in-up">
      <h3 className="flex items-center gap-2 text-xs font-semibold text-muted uppercase tracking-widest mb-1">
        <IconBrain className="w-4 h-4 text-primary" /> Price Breakdown
      </h3>
      <p className="text-[11px] text-muted mb-4">
        How each feature contributes to the final prediction of ₹{explain.predicted_price} Lakhs
      </p>

      <div style={{ height: items.length * 38 + 30 }}>
        <Bar data={chartData} options={chartOptions} />
      </div>

      {/* Summary */}
      <div className="mt-4 pt-3 border-t border-border space-y-1.5">
        {items.map((b) => (
          <div key={b.feature} className="flex justify-between text-xs">
            <span className="text-foreground/60">{b.feature}</span>
            <span className={`font-mono font-medium ${
              b.direction === "positive" ? "text-success" : "text-error"
            }`}>
              {b.contribution > 0 ? "+" : ""}₹{b.contribution.toFixed(2)} L
            </span>
          </div>
        ))}
        <div className="flex justify-between text-sm font-semibold pt-1.5 border-t border-border">
          <span className="text-foreground/80">Total</span>
          <span className="gradient-text">₹{explain.predicted_price} L</span>
        </div>
      </div>
    </div>
  );
}
