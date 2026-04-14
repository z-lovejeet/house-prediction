import PredictionForm from "./components/PredictionForm";
import { IconHome, IconDatabase, IconLocation, IconTarget, IconBrain } from "./components/icons";

export default function Home() {
  return (
    <main className="flex-1 flex flex-col">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="pt-10 pb-4 px-4 text-center">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-gradient-start to-gradient-end
                          flex items-center justify-center shadow-lg shadow-primary-glow">
            <IconHome className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-3xl md:text-4xl font-bold gradient-text tracking-tight">
            House Price Predictor
          </h1>
        </div>
        <p className="text-muted max-w-lg mx-auto text-sm leading-relaxed">
          Predict property prices across Bengaluru with 4 ML models.
          Compare results, explore analytics, and understand model behavior.
        </p>

        {/* Stats bar */}
        <div className="mt-5 flex items-center justify-center gap-3 flex-wrap">
          {[
            { icon: <IconDatabase className="w-3.5 h-3.5 text-primary" />, label: "9,200+", sub: "Listings" },
            { icon: <IconLocation className="w-3.5 h-3.5 text-accent" />,  label: "178",    sub: "Locations" },
            { icon: <IconBrain className="w-3.5 h-3.5 text-success" />,    label: "4",      sub: "Models" },
            { icon: <IconTarget className="w-3.5 h-3.5 text-warning" />,   label: "71.58%", sub: "Best R²" },
          ].map(({ icon, label, sub }) => (
            <div key={sub}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full
                         bg-card border border-border text-xs">
              {icon}
              <span className="font-semibold text-foreground/80">{label}</span>
              <span className="text-muted">{sub}</span>
            </div>
          ))}
        </div>
      </header>

      {/* ── Content ────────────────────────────────────────────── */}
      <section className="flex-1 px-4 pb-12 pt-4">
        <PredictionForm />
      </section>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="py-5 text-center text-xs text-muted/60 border-t border-border">
        <p>Built with Next.js + FastAPI + scikit-learn</p>
        <p className="mt-0.5">CSE275 Project &mdash; House Price Prediction System</p>
      </footer>
    </main>
  );
}
