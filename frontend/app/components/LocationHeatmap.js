"use client";

import { useState, useEffect, useMemo } from "react";
import { IconLocation, IconSpinner } from "./icons";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function priceColor(price, min, max) {
  const norm = Math.max(0, Math.min(1, (price - min) / (max - min || 1)));
  // Low = green, mid = yellow, high = red
  if (norm < 0.33) return { bg: "rgba(16,185,129,0.15)", text: "text-success", border: "border-success/20" };
  if (norm < 0.66) return { bg: "rgba(245,158,11,0.15)", text: "text-warning", border: "border-warning/20" };
  return { bg: "rgba(239,68,68,0.15)", text: "text-error", border: "border-error/20" };
}

export default function LocationHeatmap() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("price");
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/locations/stats`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const { locations, min, max, avg } = useMemo(() => {
    if (!data?.locations) return { locations: [], min: 0, max: 0, avg: 0 };
    let locs = data.locations;

    if (search) {
      locs = locs.filter((l) => l.location.toLowerCase().includes(search.toLowerCase()));
    }

    locs.sort((a, b) => sortBy === "price" ? b.avg_price - a.avg_price : b.count - a.count);

    const prices = data.locations.map((l) => l.avg_price);
    return {
      locations: locs,
      min: Math.min(...prices),
      max: Math.max(...prices),
      avg: prices.reduce((s, p) => s + p, 0) / prices.length,
    };
  }, [data, search, sortBy]);

  const visible = showAll ? locations : locations.slice(0, 30);

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-border">
        <h3 className="flex items-center gap-2 text-xs font-semibold text-muted uppercase tracking-widest">
          <IconLocation className="w-3.5 h-3.5 text-accent" /> Location Heatmap
        </h3>
        <div className="flex items-center gap-2 text-[10px] text-muted">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-success/60" /> Affordable
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-warning/60" /> Mid-range
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-error/60" /> Premium
          </span>
        </div>
      </div>

      {loading ? (
        <div className="px-5 py-6 flex items-center gap-2 text-sm text-muted">
          <IconSpinner className="w-4 h-4" /> Loading location data...
        </div>
      ) : (
        <div className="p-4 space-y-3">
          {/* Controls */}
          <div className="flex gap-2">
            <input type="text" value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search location..."
              className="flex-1 px-3 py-2 rounded-lg bg-background border border-border
                         text-xs placeholder:text-muted/50 focus:outline-none
                         focus:ring-1 focus:ring-primary/30 text-foreground" />
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-2 rounded-lg bg-background border border-border text-xs
                         text-foreground cursor-pointer focus:outline-none focus:ring-1 focus:ring-primary/30">
              <option value="price">Sort by Price</option>
              <option value="count">Sort by Listings</option>
            </select>
          </div>

          {/* Stats summary */}
          <div className="grid grid-cols-3 gap-2">
            <div className="text-center px-3 py-2 rounded-lg bg-success/5 border border-success/15">
              <p className="text-[10px] text-muted">Lowest Avg</p>
              <p className="text-sm font-bold text-success">₹{min.toFixed(0)} L</p>
            </div>
            <div className="text-center px-3 py-2 rounded-lg bg-primary/5 border border-primary/15">
              <p className="text-[10px] text-muted">Average</p>
              <p className="text-sm font-bold text-primary">₹{avg.toFixed(0)} L</p>
            </div>
            <div className="text-center px-3 py-2 rounded-lg bg-error/5 border border-error/15">
              <p className="text-[10px] text-muted">Highest Avg</p>
              <p className="text-sm font-bold text-error">₹{max.toFixed(0)} L</p>
            </div>
          </div>

          {/* Heatmap grid */}
          <div className="flex flex-wrap gap-1.5">
            {visible.map((loc) => {
              const c = priceColor(loc.avg_price, min, max);
              return (
                <div key={loc.location}
                  className={`px-2.5 py-1.5 rounded-lg border text-[11px] cursor-default
                              transition-transform hover:scale-105 ${c.border}`}
                  style={{ backgroundColor: c.bg }}
                  title={`${loc.location}: Avg ₹${loc.avg_price.toFixed(0)}L (${loc.count} listings)`}
                >
                  <span className={`font-medium ${c.text}`}>{loc.location}</span>
                  <span className="text-muted ml-1">₹{loc.avg_price.toFixed(0)}L</span>
                </div>
              );
            })}
          </div>

          {locations.length > 30 && !showAll && (
            <button onClick={() => setShowAll(true)}
              className="w-full text-xs text-primary hover:underline cursor-pointer py-1">
              Show all {locations.length} locations
            </button>
          )}
          {showAll && locations.length > 30 && (
            <button onClick={() => setShowAll(false)}
              className="w-full text-xs text-muted hover:underline cursor-pointer py-1">
              Show fewer
            </button>
          )}
        </div>
      )}
    </div>
  );
}
