"use client";

import { useState } from "react";
import { IconBrain, IconChevron } from "./icons";

const MODEL_INFO = {
  linear: {
    fullName: "Linear Regression",
    type: "Baseline (No Regularization)",
    formula: "Price = w₁·Area + w₂·BHK + w₃·Bath + w₄·Balcony + Σ(wₖ·Locationₖ) + b",
    howItWorks:
      "Linear Regression fits a straight-line relationship between your input features and the house price. It finds the optimal weights (coefficients) by minimizing the sum of squared errors between predicted and actual prices in the training data.",
    regularization: "None. All 262 features are used with equal consideration, which can lead to overfitting on noisy location features.",
    strengths: "Simple, interpretable, fast to train. Good baseline to compare against.",
    weaknesses: "Prone to overfitting when there are many features (262 in this case). Coefficients can become unstable with correlated features.",
  },
  ridge: {
    fullName: "Ridge Regression (L2)",
    type: "L2 Regularization",
    formula: "Minimize: Σ(yᵢ - ŷᵢ)² + α·Σ(wⱼ²)",
    howItWorks:
      "Ridge Regression adds an L2 penalty term (α × sum of squared coefficients) to the standard linear regression loss. This penalty shrinks all coefficients toward zero proportionally, preventing any single feature from dominating the prediction.",
    regularization: "L2 (Ridge) — Shrinks coefficients toward zero but never sets them exactly to zero. All 262 features remain in the model. Alpha (α) controls the penalty strength.",
    strengths: "Handles multicollinearity well. Stable coefficients, more robust than plain Linear Regression.",
    weaknesses: "Does not perform feature selection — uses all features even if some are irrelevant.",
  },
  lasso: {
    fullName: "Lasso Regression (L1)",
    type: "L1 Regularization",
    formula: "Minimize: Σ(yᵢ - ŷᵢ)² + α·Σ|wⱼ|",
    howItWorks:
      "Lasso Regression adds an L1 penalty (α × sum of absolute coefficients). Unlike Ridge, L1 can drive coefficients to exactly zero, effectively removing irrelevant features from the model and creating a sparse, interpretable prediction.",
    regularization: "L1 (Lasso) — Can shrink coefficients to exactly zero, performing automatic feature selection. Only the most important features survive.",
    strengths: "Automatic feature selection. Produces simpler, more interpretable models. Great when many features are irrelevant.",
    weaknesses: "Can be unstable when features are correlated — may arbitrarily pick one and ignore others.",
  },
  elasticnet: {
    fullName: "ElasticNet Regression (L1 + L2)",
    type: "Combined L1 + L2 Regularization",
    formula: "Minimize: Σ(yᵢ - ŷᵢ)² + α·[ρ·Σ|wⱼ| + (1-ρ)·Σ(wⱼ²)]",
    howItWorks:
      "ElasticNet combines both L1 (Lasso) and L2 (Ridge) penalties. The l1_ratio (ρ) controls the mix — with ρ=0.8, it is 80% Lasso for feature selection and 20% Ridge for coefficient stability. This gives you the best of both worlds.",
    regularization: "Combined L1+L2 with α=0.001 and l1_ratio=0.8. This model uses only 251 out of 262 features, having eliminated 11 irrelevant location features while keeping coefficients stable.",
    strengths: "Best of both Ridge and Lasso. Handles correlated features while still performing feature selection. Most robust for real-world datasets.",
    weaknesses: "Two hyperparameters to tune (α and l1_ratio). Slightly more complex to interpret than pure Ridge or Lasso.",
  },
};

function formatContribution(b) {
  if (b.feature.includes("Intercept") || b.feature.includes("Base Price")) {
    return `The base price (intercept) of the model is ₹${Math.abs(b.contribution).toFixed(2)} Lakhs. This represents the average predicted price before any feature adjustments.`;
  }
  if (b.feature.includes("Area")) {
    const dir = b.direction === "positive" ? "increases" : "decreases";
    return `Your area input ${dir} the price by ₹${Math.abs(b.contribution).toFixed(2)} Lakhs. ${b.direction === "positive" ? "Larger areas command higher prices." : "This area is below the dataset average."}`;
  }
  if (b.feature.includes("Bedroom") || b.feature.includes("BHK")) {
    const dir = b.direction === "positive" ? "increases" : "decreases";
    return `The number of bedrooms ${dir} the price by ₹${Math.abs(b.contribution).toFixed(2)} Lakhs. ${b.direction === "negative" ? "More bedrooms in the same area often indicates smaller, cheaper rooms — the model has learned this pattern from the data." : ""}`;
  }
  if (b.feature.includes("Bathroom")) {
    const dir = b.direction === "positive" ? "increases" : "decreases";
    return `Bathrooms ${dir} the price by ₹${Math.abs(b.contribution).toFixed(2)} Lakhs.`;
  }
  if (b.feature.includes("Balcony")) {
    const dir = b.direction === "positive" ? "increases" : "decreases";
    return `Balconies ${dir} the price by ₹${Math.abs(b.contribution).toFixed(2)} Lakhs.`;
  }
  if (b.feature.includes("Location")) {
    const dir = b.direction === "positive" ? "premium" : "below-average";
    return `This location has a ${dir} effect of ₹${Math.abs(b.contribution).toFixed(2)} Lakhs compared to the baseline. ${b.direction === "positive" ? "Properties here tend to be priced higher." : "Properties in this area are typically more affordable."}`;
  }
  return `${b.feature} contributes ₹${b.contribution.toFixed(2)} Lakhs to the final price.`;
}

export default function ModelExplanation({ result, explainData, models }) {
  const [expanded, setExpanded] = useState(true);

  if (!result) return null;

  const modelKey = result.model_used;
  const info = MODEL_INFO[modelKey] || MODEL_INFO.elasticnet;
  const modelMeta = models?.find((m) => m.key === modelKey);

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden animate-fade-in-up">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-5 py-4 text-left
                   hover:bg-card-hover transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-primary/10">
            <IconBrain className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">How This Prediction Works</h3>
            <p className="text-[11px] text-muted">Model explanation, methodology, and price rationale</p>
          </div>
        </div>
        <IconChevron className="w-4 h-4 text-muted" direction={expanded ? "up" : "down"} />
      </button>

      {expanded && (
        <div className="px-5 pb-5 space-y-5 border-t border-border pt-4">

          {/* Section 1: Model Identity */}
          <div>
            <h4 className="text-xs font-semibold text-primary uppercase tracking-widest mb-2">
              Active Model
            </h4>
            <div className="bg-background rounded-xl p-4 border border-border">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-bold text-foreground">{info.fullName}</p>
                  <p className="text-[11px] text-muted mt-0.5">{info.type}</p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-bold gradient-text">R² = {modelMeta?.r2 || result.model_r2}</p>
                  <p className="text-[10px] text-muted">Accuracy Score</p>
                </div>
              </div>
              <div className="mt-3 px-3 py-2 bg-card rounded-lg border border-border">
                <p className="text-[11px] font-mono text-primary/80">{info.formula}</p>
              </div>
            </div>
          </div>

          {/* Section 2: How it works */}
          <div>
            <h4 className="text-xs font-semibold text-primary uppercase tracking-widest mb-2">
              How It Works
            </h4>
            <p className="text-xs text-foreground/70 leading-relaxed">{info.howItWorks}</p>
          </div>

          {/* Section 3: Regularization */}
          <div>
            <h4 className="text-xs font-semibold text-primary uppercase tracking-widest mb-2">
              Regularization
            </h4>
            <p className="text-xs text-foreground/70 leading-relaxed">{info.regularization}</p>
            <div className="mt-2 grid grid-cols-2 gap-3">
              <div className="bg-background rounded-lg p-3 border border-success/15">
                <p className="text-[10px] text-success font-semibold uppercase tracking-wider mb-1">Strengths</p>
                <p className="text-[11px] text-foreground/60 leading-relaxed">{info.strengths}</p>
              </div>
              <div className="bg-background rounded-lg p-3 border border-error/15">
                <p className="text-[10px] text-error font-semibold uppercase tracking-wider mb-1">Limitations</p>
                <p className="text-[11px] text-foreground/60 leading-relaxed">{info.weaknesses}</p>
              </div>
            </div>
          </div>

          {/* Section 4: Price Rationale */}
          {explainData?.breakdown && (
            <div>
              <h4 className="text-xs font-semibold text-primary uppercase tracking-widest mb-2">
                Why ₹{result.predicted_price >= 100
                  ? (result.predicted_price / 100).toFixed(2) + " Crores"
                  : result.predicted_price.toFixed(2) + " Lakhs"}?
              </h4>
              <p className="text-xs text-foreground/60 mb-3">
                The predicted price is the sum of each feature&apos;s contribution. Here&apos;s what each feature added or subtracted:
              </p>
              <div className="space-y-2.5">
                {explainData.breakdown.map((b, i) => (
                  <div key={i} className="flex gap-3 items-start">
                    <div className={`mt-0.5 w-2 h-2 rounded-full shrink-0 ${
                      b.direction === "positive" ? "bg-success" : "bg-error"
                    }`} />
                    <div>
                      <p className="text-xs font-medium text-foreground/80">
                        {b.feature}
                        <span className={`ml-2 font-mono text-[11px] ${
                          b.direction === "positive" ? "text-success" : "text-error"
                        }`}>
                          {b.contribution > 0 ? "+" : ""}₹{b.contribution.toFixed(2)} L
                        </span>
                      </p>
                      <p className="text-[11px] text-foreground/50 mt-0.5 leading-relaxed">
                        {formatContribution(b)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 bg-background rounded-xl p-4 border border-primary/15">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-medium text-foreground/70">Final Predicted Price</span>
                  <span className="text-base font-bold gradient-text">
                    ₹{result.predicted_price >= 100
                      ? (result.predicted_price / 100).toFixed(2) + " Crores"
                      : result.predicted_price.toFixed(2) + " Lakhs"}
                  </span>
                </div>
                <p className="text-[10px] text-muted mt-1">
                  = Base Price ({explainData.breakdown.find(b => b.feature.includes("Intercept"))?.contribution.toFixed(2) || "N/A"} L)
                  {explainData.breakdown
                    .filter(b => !b.feature.includes("Intercept"))
                    .map(b => ` ${b.contribution >= 0 ? "+" : ""}${b.contribution.toFixed(2)} L (${b.feature})`)
                    .join("")}
                </p>
              </div>
            </div>
          )}

          {/* Section 5: Dataset Context */}
          <div className="bg-background rounded-xl p-4 border border-border">
            <h4 className="text-xs font-semibold text-primary uppercase tracking-widest mb-2">
              Training Data Context
            </h4>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <p className="text-sm font-bold text-foreground">9,200+</p>
                <p className="text-[10px] text-muted">Training Samples</p>
              </div>
              <div>
                <p className="text-sm font-bold text-foreground">{modelMeta?.non_zero_features || 262}</p>
                <p className="text-[10px] text-muted">Features Used</p>
              </div>
              <div>
                <p className="text-sm font-bold text-foreground">178</p>
                <p className="text-[10px] text-muted">Locations</p>
              </div>
            </div>
            <p className="text-[11px] text-foreground/50 mt-3 leading-relaxed">
              The model was trained on Bengaluru housing data using 5-fold cross-validation with GridSearchCV for hyperparameter optimization. The final model was retrained on the complete dataset for maximum accuracy.
            </p>
          </div>

        </div>
      )}
    </div>
  );
}

// Export model info so PDFExport can use it
export { MODEL_INFO, formatContribution };
