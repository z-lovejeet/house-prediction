"use client";

import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, PointElement, LineElement,
  ArcElement, RadialLinearScale, Title, Tooltip, Legend, Filler,
} from "chart.js";
import { Bar, Doughnut, Radar } from "react-chartjs-2";
import { IconChart } from "./icons";

ChartJS.register(
  CategoryScale, LinearScale, BarElement, PointElement, LineElement,
  ArcElement, RadialLinearScale, Title, Tooltip, Legend, Filler
);

const COLORS = {
  elasticnet: { bg: "rgba(99, 102, 241, 0.65)",  border: "#6366f1" },
  ridge:      { bg: "rgba(6, 182, 212, 0.65)",   border: "#06b6d4" },
  lasso:      { bg: "rgba(16, 185, 129, 0.65)",  border: "#10b981" },
  linear:     { bg: "rgba(245, 158, 11, 0.65)",  border: "#f59e0b" },
};

const LABELS = {
  elasticnet: "ElasticNet", ridge: "Ridge", lasso: "Lasso", linear: "Linear Reg.",
};

export default function ModelCharts({ compareData, models }) {
  if (!models || models.length === 0) return null;

  const order = ["elasticnet", "ridge", "lasso", "linear"];
  const sorted = order.map((k) => models.find((m) => m.key === k)).filter(Boolean);

  const labels    = sorted.map((m) => LABELS[m.key]);
  const r2Values  = sorted.map((m) => m.r2 * 100);
  const features  = sorted.map((m) => m.non_zero_features);
  const bgColors  = sorted.map((m) => COLORS[m.key]?.bg);
  const borders   = sorted.map((m) => COLORS[m.key]?.border);

  const tickColor = "rgba(148,163,184,0.7)";
  const gridColor = "rgba(148,163,184,0.08)";

  const barBase = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: tickColor, font: { size: 11 } }, grid: { color: gridColor } },
      y: { ticks: { color: tickColor, font: { size: 11 } }, grid: { color: gridColor } },
    },
  };

  // R² chart
  const r2Data = {
    labels,
    datasets: [{
      data: r2Values,
      backgroundColor: bgColors,
      borderColor: borders,
      borderWidth: 2, borderRadius: 6, borderSkipped: false,
    }],
  };
  const r2Opts = {
    ...barBase,
    plugins: {
      ...barBase.plugins,
      tooltip: { callbacks: { label: (c) => `R²: ${c.parsed.y.toFixed(2)}%` } },
    },
    scales: {
      ...barBase.scales,
      y: {
        ...barBase.scales.y,
        min: Math.min(...r2Values) - 2,
        max: Math.max(...r2Values) + 1,
        ticks: { ...barBase.scales.y.ticks, callback: (v) => v.toFixed(1) + "%" },
      },
    },
  };

  // Prediction chart (only after compare)
  const hasCompare = compareData?.comparisons;
  let predData = null;
  if (hasCompare) {
    const sc = order.map((k) => compareData.comparisons.find((c) => c.model === k)).filter(Boolean);
    predData = {
      labels: sc.map((c) => LABELS[c.model]),
      datasets: [{
        data: sc.map((c) => c.predicted_price),
        backgroundColor: sc.map((c) => COLORS[c.model]?.bg),
        borderColor: sc.map((c) => COLORS[c.model]?.border),
        borderWidth: 2, borderRadius: 6, borderSkipped: false,
      }],
    };
  }
  const predOpts = {
    ...barBase,
    plugins: {
      ...barBase.plugins,
      tooltip: { callbacks: { label: (c) => `₹ ${c.parsed.y.toFixed(2)} Lakhs` } },
    },
    scales: {
      ...barBase.scales,
      y: { ...barBase.scales.y, ticks: { ...barBase.scales.y.ticks, callback: (v) => "₹" + v } },
    },
  };

  // Radar
  const coeffKeys = ["total_sqft", "bath", "balcony", "bhk"];
  const coeffLabels = ["Area (sqft)", "Bathrooms", "Balcony", "Bedrooms"];
  const radarData = {
    labels: coeffLabels,
    datasets: sorted.map((m) => ({
      label: LABELS[m.key],
      data: coeffKeys.map((k) => k === "total_sqft" ? (m.coefficients[k] || 0) / 10 : (m.coefficients[k] || 0)),
      backgroundColor: COLORS[m.key]?.bg.replace("0.65", "0.1"),
      borderColor: COLORS[m.key]?.border,
      borderWidth: 2, pointRadius: 3,
      pointBackgroundColor: COLORS[m.key]?.border,
    })),
  };
  const radarOpts = {
    responsive: true, maintainAspectRatio: false,
    scales: {
      r: {
        ticks: { color: tickColor, font: { size: 9 }, backdropColor: "transparent" },
        grid: { color: gridColor }, angleLines: { color: gridColor },
        pointLabels: { color: tickColor, font: { size: 11 } },
      },
    },
    plugins: {
      legend: { position: "bottom", labels: { color: tickColor, font: { size: 10 }, padding: 12 } },
      tooltip: {
        callbacks: {
          label: (c) => {
            const val = sorted[c.datasetIndex]?.coefficients[coeffKeys[c.dataIndex]] || 0;
            return `${c.dataset.label}: ${val.toFixed(4)}`;
          },
        },
      },
    },
  };

  // Doughnut
  const donutData = {
    labels: sorted.map((m) => LABELS[m.key]),
    datasets: [{ data: features, backgroundColor: bgColors, borderColor: borders, borderWidth: 2, hoverOffset: 6 }],
  };
  const donutOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: "bottom", labels: { color: tickColor, font: { size: 11 }, padding: 14 } },
      tooltip: { callbacks: { label: (c) => `${c.label}: ${c.parsed}/262 features` } },
    },
  };

  return (
    <div className="space-y-5 animate-fade-in-up">
      <h3 className="flex items-center gap-2 text-xs font-semibold text-muted uppercase tracking-widest">
        <IconChart className="w-4 h-4 text-primary" /> Model Analytics
      </h3>

      {/* Row 1 */}
      <div className={`grid gap-4 ${hasCompare ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}>
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs font-medium text-muted mb-3">R² Score Comparison</p>
          <div className="h-52"><Bar data={r2Data} options={r2Opts} /></div>
        </div>
        {hasCompare && predData && (
          <div className="bg-card border border-border rounded-xl p-4">
            <p className="text-xs font-medium text-muted mb-3">Price Predictions by Model</p>
            <div className="h-52"><Bar data={predData} options={predOpts} /></div>
          </div>
        )}
      </div>

      {/* Row 2 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs font-medium text-muted mb-3">Feature Coefficients (area ÷10)</p>
          <div className="h-64"><Radar data={radarData} options={radarOpts} /></div>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-xs font-medium text-muted mb-3">Active Features per Model</p>
          <div className="h-64 flex items-center justify-center"><Doughnut data={donutData} options={donutOpts} /></div>
        </div>
      </div>
    </div>
  );
}
