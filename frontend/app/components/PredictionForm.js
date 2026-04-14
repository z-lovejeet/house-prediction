"use client";

import { useState, useCallback } from "react";
import { LOCATIONS } from "../data/locations";

/* ═══════════════════════════════════════════════════════════════
   API Configuration
   ═════════════════════════════════════════════════════════════ */
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ═══════════════════════════════════════════════════════════════
   Icons (inline SVG — no extra dependency)
   ═════════════════════════════════════════════════════════════ */
const AreaIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
  </svg>
);
const BedIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12l8.954-8.955a1.126 1.126 0 011.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25" />
  </svg>
);
const BathIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);
const LocationIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
  </svg>
);

/* ═══════════════════════════════════════════════════════════════
   Format price for display
   ═════════════════════════════════════════════════════════════ */
function formatPrice(lakhs) {
  if (lakhs >= 100) {
    const crores = (lakhs / 100).toFixed(2);
    return { value: crores, unit: "Crores", raw: lakhs };
  }
  return { value: lakhs.toFixed(2), unit: "Lakhs", raw: lakhs };
}

/* ═══════════════════════════════════════════════════════════════
   PredictionForm Component
   ═════════════════════════════════════════════════════════════ */
export default function PredictionForm() {
  /* ── State ────────────────────────────────────────────────── */
  const [form, setForm] = useState({
    area: "",
    bedrooms: "",
    bathrooms: "",
    location: "",
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /* ── Handlers ─────────────────────────────────────────────── */
  const handleChange = useCallback((e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    if (error) setError(null);
  }, [error]);

  const handleReset = useCallback(() => {
    setForm({ area: "", bedrooms: "", bathrooms: "", location: "" });
    setResult(null);
    setError(null);
  }, []);

  /* ── Validation ───────────────────────────────────────────── */
  const validate = () => {
    if (!form.area || Number(form.area) <= 0) {
      return "Please enter a valid area (positive number).";
    }
    if (!form.bedrooms || Number(form.bedrooms) < 1) {
      return "Please enter at least 1 bedroom.";
    }
    if (!form.bathrooms || Number(form.bathrooms) < 1) {
      return "Please enter at least 1 bathroom.";
    }
    if (!form.location) {
      return "Please select a location.";
    }
    return null;
  };

  /* ── Submit ───────────────────────────────────────────────── */
  const handleSubmit = async (e) => {
    e.preventDefault();

    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          area: Number(form.area),
          bedrooms: Number(form.bedrooms),
          bathrooms: Number(form.bathrooms),
          location: form.location,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(
          errData?.detail?.[0]?.msg ||
          errData?.detail ||
          `Server returned ${response.status}`
        );
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        setError("Cannot connect to the prediction server. Make sure the FastAPI backend is running on " + API_URL);
      } else {
        setError(err.message || "Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const price = result ? formatPrice(result.predicted_price) : null;

  /* ── Render ───────────────────────────────────────────────── */
  return (
    <div className="w-full max-w-xl mx-auto">
      {/* ── Form Card ─────────────────────────────────────────── */}
      <form
        onSubmit={handleSubmit}
        className="bg-card border border-border rounded-2xl p-6 md:p-8 shadow-lg"
      >
        <div className="grid gap-5">
          {/* Area */}
          <div className="space-y-1.5">
            <label htmlFor="area" className="flex items-center gap-2 text-sm font-medium text-foreground/70">
              <AreaIcon /> Area (sq. ft)
            </label>
            <input
              id="area"
              name="area"
              type="number"
              min="1"
              step="any"
              placeholder="e.g. 1500"
              value={form.area}
              onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl bg-background border border-border
                         placeholder:text-foreground/30 focus:outline-none focus:ring-2
                         focus:ring-primary/40 focus:border-primary text-foreground"
              required
            />
          </div>

          {/* Bedrooms & Bathrooms — side by side */}
          <div className="grid grid-cols-2 gap-4">
            {/* Bedrooms */}
            <div className="space-y-1.5">
              <label htmlFor="bedrooms" className="flex items-center gap-2 text-sm font-medium text-foreground/70">
                <BedIcon /> Bedrooms
              </label>
              <input
                id="bedrooms"
                name="bedrooms"
                type="number"
                min="1"
                max="20"
                placeholder="e.g. 3"
                value={form.bedrooms}
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl bg-background border border-border
                           placeholder:text-foreground/30 focus:outline-none focus:ring-2
                           focus:ring-primary/40 focus:border-primary text-foreground"
                required
              />
            </div>

            {/* Bathrooms */}
            <div className="space-y-1.5">
              <label htmlFor="bathrooms" className="flex items-center gap-2 text-sm font-medium text-foreground/70">
                <BathIcon /> Bathrooms
              </label>
              <input
                id="bathrooms"
                name="bathrooms"
                type="number"
                min="1"
                max="20"
                placeholder="e.g. 2"
                value={form.bathrooms}
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl bg-background border border-border
                           placeholder:text-foreground/30 focus:outline-none focus:ring-2
                           focus:ring-primary/40 focus:border-primary text-foreground"
                required
              />
            </div>
          </div>

          {/* Location */}
          <div className="space-y-1.5">
            <label htmlFor="location" className="flex items-center gap-2 text-sm font-medium text-foreground/70">
              <LocationIcon /> Location
            </label>
            <select
              id="location"
              name="location"
              value={form.location}
              onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl bg-background border border-border
                         text-foreground focus:outline-none focus:ring-2
                         focus:ring-primary/40 focus:border-primary appearance-none
                         cursor-pointer"
              required
            >
              <option value="" disabled>Select a location…</option>
              {LOCATIONS.map((loc) => (
                <option key={loc} value={loc}>
                  {loc}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* ── Error ───────────────────────────────────────────── */}
        {error && (
          <div className="mt-4 px-4 py-3 rounded-xl bg-error-bg border border-error/20
                          text-error text-sm animate-fade-in-up">
            ⚠️ {error}
          </div>
        )}

        {/* ── Buttons ─────────────────────────────────────────── */}
        <div className="flex gap-3 mt-6">
          <button
            type="submit"
            disabled={loading}
            className="flex-1 py-3.5 rounded-xl font-semibold text-white
                       bg-gradient-to-r from-gradient-start to-gradient-end
                       hover:shadow-lg hover:shadow-primary-glow
                       active:scale-[0.98] disabled:opacity-60
                       disabled:cursor-not-allowed transition-all cursor-pointer"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Predicting…
              </span>
            ) : (
              "Predict Price"
            )}
          </button>

          <button
            type="button"
            onClick={handleReset}
            className="px-5 py-3.5 rounded-xl font-medium border border-border
                       text-foreground/60 hover:bg-card-hover hover:text-foreground
                       active:scale-[0.98] transition-all cursor-pointer"
          >
            Reset
          </button>
        </div>
      </form>

      {/* ── Result Card ───────────────────────────────────────── */}
      {result && price && (
        <div className="mt-6 animate-fade-in-up">
          <div className="bg-card border border-success/30 rounded-2xl p-6 md:p-8
                          result-glow text-center">
            <p className="text-sm font-medium text-foreground/50 mb-1">
              Estimated Price
            </p>
            <p className="text-4xl md:text-5xl font-bold gradient-text mb-1">
              ₹ {price.value}
            </p>
            <p className="text-lg font-medium text-foreground/60">
              {price.unit}
            </p>

            {/* Details row */}
            <div className="mt-5 flex flex-wrap justify-center gap-x-4 gap-y-1
                            text-xs text-foreground/40">
              <span>{form.area} sq.ft</span>
              <span>•</span>
              <span>{form.bedrooms} BHK</span>
              <span>•</span>
              <span>{form.bathrooms} Bath</span>
              <span>•</span>
              <span>{form.location}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
