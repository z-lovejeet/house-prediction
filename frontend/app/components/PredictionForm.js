"use client";

import { useState, useCallback, useEffect } from "react";
import { LOCATIONS } from "../data/locations";
import ModelSelector from "./ModelSelector";
import ComparisonTable from "./ComparisonTable";
import ModelCharts from "./ModelCharts";
import InsightsPanel from "./InsightsPanel";
import {
  IconArea, IconBed, IconBath, IconLocation, IconSpinner, IconBolt, IconCrown,
} from "./icons";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function formatPrice(lakhs) {
  if (lakhs >= 100) return { value: (lakhs / 100).toFixed(2), unit: "Crores" };
  return { value: lakhs.toFixed(2), unit: "Lakhs" };
}

export default function PredictionForm() {
  const [form, setForm] = useState({ area: "", bedrooms: "", bathrooms: "", location: "" });
  const [selectedModel, setSelectedModel] = useState(null);
  const [models, setModels] = useState([]);
  const [result, setResult] = useState(null);
  const [compareData, setCompareData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then((r) => r.json())
      .then((data) => {
        setModels(data.models || []);
        if (data.best_model) setSelectedModel(data.best_model);
      })
      .catch(() => {});
  }, []);

  const handleChange = useCallback((e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
    if (error) setError(null);
  }, [error]);

  const handleReset = useCallback(() => {
    setForm({ area: "", bedrooms: "", bathrooms: "", location: "" });
    setResult(null);
    setCompareData(null);
    setError(null);
  }, []);

  const validate = () => {
    if (!form.area || Number(form.area) <= 0) return "Enter a valid area (positive number).";
    if (!form.bedrooms || Number(form.bedrooms) < 1) return "Enter at least 1 bedroom.";
    if (!form.bathrooms || Number(form.bathrooms) < 1) return "Enter at least 1 bathroom.";
    if (!form.location) return "Select a location.";
    return null;
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    const err = validate();
    if (err) return setError(err);

    setLoading(true); setError(null); setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          area: Number(form.area), bedrooms: Number(form.bedrooms),
          bathrooms: Number(form.bathrooms), location: form.location, model: selectedModel,
        }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => null);
        throw new Error(d?.detail?.[0]?.msg || d?.detail || `Server error ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message.includes("fetch") ? "Cannot connect to backend. Is it running?" : e.message);
    } finally { setLoading(false); }
  };

  const handleCompare = async () => {
    const err = validate();
    if (err) return setError(err);

    setComparing(true); setError(null); setCompareData(null);

    try {
      const res = await fetch(`${API_URL}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          area: Number(form.area), bedrooms: Number(form.bedrooms),
          bathrooms: Number(form.bathrooms), location: form.location,
        }),
      });
      if (!res.ok) throw new Error("Comparison failed");
      setCompareData(await res.json());
    } catch (e) { setError(e.message); }
    finally { setComparing(false); }
  };

  const price = result ? formatPrice(result.predicted_price) : null;
  const modelInfo = result ? models.find((m) => m.key === result.model_used) : null;

  const formatModelName = (key) =>
    key?.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())
      .replace("Elasticnet", "ElasticNet").replace("Linear", "Linear Regression");

  /* ── Input field config ───────────────────────────────────── */
  const inputClass = `w-full px-4 py-3 rounded-xl bg-background border border-border
    placeholder:text-muted/50 focus:outline-none focus:ring-2
    focus:ring-primary/30 focus:border-primary text-foreground transition-all`;

  return (
    <div className="w-full max-w-3xl mx-auto space-y-6">

      <ModelSelector models={models} selected={selectedModel} onSelect={setSelectedModel} />

      {/* ── Form ────────────────────────────────────────────── */}
      <form onSubmit={handlePredict} className="bg-card border border-border rounded-2xl p-5 md:p-6 shadow-sm">
        <div className="grid gap-4">
          <div className="space-y-1.5">
            <label htmlFor="area" className="flex items-center gap-2 text-xs font-medium text-muted">
              <IconArea className="w-3.5 h-3.5" /> Area (sq. ft)
            </label>
            <input id="area" name="area" type="number" min="1" step="any"
              placeholder="e.g. 1500" value={form.area} onChange={handleChange}
              className={inputClass} required />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <label htmlFor="bedrooms" className="flex items-center gap-2 text-xs font-medium text-muted">
                <IconBed className="w-3.5 h-3.5" /> Bedrooms
              </label>
              <input id="bedrooms" name="bedrooms" type="number" min="1" max="20"
                placeholder="e.g. 3" value={form.bedrooms} onChange={handleChange}
                className={inputClass} required />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="bathrooms" className="flex items-center gap-2 text-xs font-medium text-muted">
                <IconBath className="w-3.5 h-3.5" /> Bathrooms
              </label>
              <input id="bathrooms" name="bathrooms" type="number" min="1" max="20"
                placeholder="e.g. 2" value={form.bathrooms} onChange={handleChange}
                className={inputClass} required />
            </div>
          </div>

          <div className="space-y-1.5">
            <label htmlFor="location" className="flex items-center gap-2 text-xs font-medium text-muted">
              <IconLocation className="w-3.5 h-3.5" /> Location
            </label>
            <select id="location" name="location" value={form.location} onChange={handleChange}
              className={`${inputClass} appearance-none cursor-pointer`} required>
              <option value="" disabled>Select a location...</option>
              {LOCATIONS.map((loc) => <option key={loc} value={loc}>{loc}</option>)}
            </select>
          </div>
        </div>

        {error && (
          <div className="mt-3 px-4 py-2.5 rounded-xl bg-error-bg border border-error/15
                          text-error text-sm animate-fade-in-up">
            {error}
          </div>
        )}

        <div className="flex gap-3 mt-5">
          <button type="submit" disabled={loading}
            className="flex-1 py-3 rounded-xl font-semibold text-white
                       bg-gradient-to-r from-gradient-start to-gradient-end
                       hover:shadow-lg hover:shadow-primary-glow
                       active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed
                       transition-all cursor-pointer text-sm">
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <IconSpinner className="w-4 h-4" /> Predicting...
              </span>
            ) : "Predict Price"}
          </button>

          <button type="button" onClick={handleCompare} disabled={comparing}
            className="flex items-center gap-1.5 px-4 py-3 rounded-xl font-medium
                       border border-accent/30 text-accent text-sm
                       hover:bg-accent/5 active:scale-[0.98] disabled:opacity-50
                       disabled:cursor-not-allowed transition-all cursor-pointer">
            {comparing ? <IconSpinner className="w-3.5 h-3.5" /> : <IconBolt className="w-3.5 h-3.5" />}
            Compare All
          </button>

          <button type="button" onClick={handleReset}
            className="px-4 py-3 rounded-xl font-medium border border-border
                       text-muted hover:bg-card-hover hover:text-foreground
                       active:scale-[0.98] transition-all cursor-pointer text-sm">
            Reset
          </button>
        </div>
      </form>

      {/* ── Result ──────────────────────────────────────────── */}
      {result && price && (
        <div className="animate-fade-in-up">
          <div className="bg-card border border-success/20 rounded-2xl p-6 result-glow text-center">
            <p className="text-xs font-medium text-muted uppercase tracking-wider mb-1">
              Estimated Price
            </p>
            <p className="text-4xl md:text-5xl font-bold gradient-text mb-0.5">
              ₹ {price.value}
            </p>
            <p className="text-base font-medium text-foreground/50">{price.unit}</p>

            <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-full
                            bg-primary/5 border border-primary/15 text-xs text-primary font-medium">
              {modelInfo?.is_best && <IconCrown className="w-3 h-3 text-amber-500" />}
              {formatModelName(result.model_used)}
              <span className="text-muted">·</span>
              R² = {result.model_r2}
            </div>

            <div className="mt-3 flex flex-wrap justify-center gap-x-3 gap-y-1 text-xs text-muted">
              <span>{form.area} sqft</span>
              <span className="text-border">|</span>
              <span>{form.bedrooms} BHK</span>
              <span className="text-border">|</span>
              <span>{form.bathrooms} Bath</span>
              <span className="text-border">|</span>
              <span>{form.location}</span>
            </div>

            <p className="mt-2 text-[11px] text-muted/60">
              ≈ ₹ {((result.predicted_price * 100000) / Number(form.area)).toLocaleString("en-IN", { maximumFractionDigits: 0 })}/sqft
            </p>
          </div>
        </div>
      )}

      {compareData && <ComparisonTable data={compareData} onClose={() => setCompareData(null)} />}
      {models.length > 0 && <ModelCharts compareData={compareData} models={models} />}
      <InsightsPanel />
    </div>
  );
}
