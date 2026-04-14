import PredictionForm from "./components/PredictionForm";

export default function Home() {
  return (
    <main className="flex-1 flex flex-col">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="pt-12 pb-6 px-4 text-center">
        <div className="flex items-center justify-center gap-3 mb-3">
          <span className="text-3xl">🏠</span>
          <h1 className="text-3xl md:text-4xl font-bold gradient-text">
            House Price Predictor
          </h1>
        </div>
        <p className="text-foreground/50 max-w-md mx-auto text-sm md:text-base">
          Predict property prices across Bengaluru using our trained ML model.
          Enter details below and get an instant estimate.
        </p>

        {/* Model badge */}
        <div className="mt-4 inline-flex items-center gap-2 px-3 py-1.5 rounded-full
                        bg-card border border-border text-xs text-foreground/50">
          <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
          ElasticNet Model — R² = 0.7158
        </div>
      </header>

      {/* ── Form ───────────────────────────────────────────────── */}
      <section className="flex-1 px-4 pb-16">
        <PredictionForm />
      </section>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="py-6 text-center text-xs text-foreground/30 border-t border-border">
        <p>
          Built with Next.js + FastAPI • Trained on 9,200+ Bengaluru listings
        </p>
        <p className="mt-1">
          CSE275 Project — House Price Prediction System
        </p>
      </footer>
    </main>
  );
}
