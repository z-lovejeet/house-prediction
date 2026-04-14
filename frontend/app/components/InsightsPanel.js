"use client";

import { useState } from "react";
import { IconLightbulb, IconChevron, IconArrowUp, IconArrowDown } from "./icons";

export default function InsightsPanel() {
  const [open, setOpen] = useState(false);

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3.5
                   hover:bg-card-hover transition-colors cursor-pointer"
      >
        <span className="flex items-center gap-2.5 text-sm font-semibold text-foreground/70">
          <IconLightbulb className="w-4 h-4 text-warning" />
          Why do more bedrooms decrease the price?
        </span>
        <IconChevron className="w-4 h-4 text-muted" direction={open ? "up" : "down"} />
      </button>

      {open && (
        <div className="px-5 pb-5 space-y-4 text-sm text-foreground/70 animate-fade-in-up border-t border-border pt-4">
          {/* Alert */}
          <div className="bg-warning/5 border border-warning/15 rounded-lg p-4">
            <p className="font-semibold text-warning mb-1.5 text-xs uppercase tracking-wider">
              Not a bug — learned market behavior
            </p>
            <p className="text-foreground/60 text-[13px] leading-relaxed">
              The model learned from 9,200+ Bengaluru listings that{" "}
              <strong className="text-foreground/80">more bedrooms in the same area</strong> signals a{" "}
              <strong className="text-foreground/80">lower quality property</strong>.
            </p>
          </div>

          {/* Comparison cards */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-success/5 border border-success/15 rounded-lg p-3.5 text-center">
              <div className="w-8 h-8 mx-auto mb-2 rounded-lg bg-success/10 flex items-center justify-center">
                <IconArrowUp className="w-4 h-4 text-success" />
              </div>
              <p className="font-semibold text-success text-xs">1500 sqft, 3 BHK</p>
              <p className="text-[11px] text-muted mt-1">~500 sqft per room</p>
              <p className="text-[11px] text-muted">Spacious premium flat</p>
              <p className="font-bold text-success mt-2 text-sm">₹ 1.13 Cr</p>
            </div>
            <div className="bg-error/5 border border-error/15 rounded-lg p-3.5 text-center">
              <div className="w-8 h-8 mx-auto mb-2 rounded-lg bg-error/10 flex items-center justify-center">
                <IconArrowDown className="w-4 h-4 text-error" />
              </div>
              <p className="font-semibold text-error text-xs">1500 sqft, 5 BHK</p>
              <p className="text-[11px] text-muted mt-1">~300 sqft per room</p>
              <p className="text-[11px] text-muted">Cramped layout</p>
              <p className="font-bold text-error mt-2 text-sm">₹ 82 L</p>
            </div>
          </div>

          {/* Coefficients */}
          <div className="bg-card-hover/50 rounded-lg p-4">
            <p className="font-semibold text-foreground/60 text-[11px] uppercase tracking-widest mb-2.5">
              Model Coefficients (per unit)
            </p>
            <div className="space-y-2 text-[13px]">
              {[
                { label: "Area (per sqft)",  value: "+₹ 0.097 L",  color: "text-success", icon: <IconArrowUp className="w-3.5 h-3.5" /> },
                { label: "Bathroom",         value: "+₹ 8.54 L",   color: "text-success", icon: <IconArrowUp className="w-3.5 h-3.5" /> },
                { label: "Bedroom (BHK)",    value: "−₹ 15.24 L",  color: "text-error",   icon: <IconArrowDown className="w-3.5 h-3.5" /> },
                { label: "Balcony",          value: "−₹ 1.24 L",   color: "text-error",   icon: <IconArrowDown className="w-3.5 h-3.5" /> },
              ].map(({ label, value, color, icon }) => (
                <div key={label} className="flex justify-between items-center">
                  <span className="text-foreground/60">{label}</span>
                  <span className={`font-mono font-medium flex items-center gap-1.5 ${color}`}>
                    {icon} {value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <p className="text-xs text-muted italic leading-relaxed">
            Tip: For a fair comparison, increase area proportionally with bedrooms.
            A 5 BHK at 2500+ sqft will predict a much higher price.
          </p>
        </div>
      )}
    </div>
  );
}
