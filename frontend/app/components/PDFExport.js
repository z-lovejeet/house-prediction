"use client";

import { useCallback } from "react";

/**
 * PDFExport — Generates a professional PDF report of the prediction.
 */
export default function PDFExport({ result, form, models, compareData, explainData }) {
  const generate = useCallback(async () => {
    const { jsPDF } = await import("jspdf");
    const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });

    const W = 210;
    const margin = 18;
    const cw = W - 2 * margin;
    let y = 20;

    const colors = {
      primary: [99, 102, 241],
      muted: [148, 163, 184],
      dark: [15, 23, 42],
      success: [16, 185, 129],
      error: [239, 68, 68],
      bg: [248, 250, 252],
    };

    // ── Header ──
    doc.setFillColor(...colors.primary);
    doc.rect(0, 0, W, 38, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.setFont(undefined, "bold");
    doc.text("House Price Prediction Report", margin, 18);
    doc.setFontSize(9);
    doc.setFont(undefined, "normal");
    doc.text(`Generated on ${new Date().toLocaleString("en-IN")}`, margin, 26);
    doc.text("CSE275 Project — Bengaluru Housing Analysis", margin, 32);
    y = 48;

    // ── Input Summary ──
    doc.setTextColor(...colors.dark);
    doc.setFontSize(12);
    doc.setFont(undefined, "bold");
    doc.text("Input Parameters", margin, y);
    y += 8;

    doc.setFillColor(...colors.bg);
    doc.roundedRect(margin, y - 3, cw, 28, 2, 2, "F");

    doc.setFontSize(9);
    doc.setFont(undefined, "normal");
    doc.setTextColor(...colors.muted);

    const inputs = [
      ["Area", `${form.area} sq. ft`],
      ["Bedrooms", `${form.bedrooms} BHK`],
      ["Bathrooms", form.bathrooms],
      ["Location", form.location],
    ];

    inputs.forEach(([label, val], i) => {
      const x = margin + 4 + (i % 2) * (cw / 2);
      const row = Math.floor(i / 2) * 12;
      doc.text(label, x, y + 4 + row);
      doc.setTextColor(...colors.dark);
      doc.setFont(undefined, "bold");
      doc.text(String(val), x + 28, y + 4 + row);
      doc.setFont(undefined, "normal");
      doc.setTextColor(...colors.muted);
    });
    y += 34;

    // ── Prediction Result ──
    if (result) {
      doc.setTextColor(...colors.dark);
      doc.setFontSize(12);
      doc.setFont(undefined, "bold");
      doc.text("Prediction Result", margin, y);
      y += 8;

      doc.setFillColor(...colors.primary);
      doc.roundedRect(margin, y - 3, cw, 22, 2, 2, "F");
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(18);
      const price = result.predicted_price >= 100
        ? `Rs ${(result.predicted_price / 100).toFixed(2)} Crores`
        : `Rs ${result.predicted_price.toFixed(2)} Lakhs`;
      doc.text(price, margin + 6, y + 8);
      doc.setFontSize(9);
      doc.setFont(undefined, "normal");
      doc.text(
        `Model: ${result.model_used} | R² = ${result.model_r2} | MSE = ${result.model_mse}`,
        margin + 6, y + 15
      );
      y += 30;
    }

    // ── Feature Contributions ──
    if (explainData?.breakdown) {
      doc.setTextColor(...colors.dark);
      doc.setFontSize(12);
      doc.setFont(undefined, "bold");
      doc.text("Price Breakdown (Feature Contributions)", margin, y);
      y += 8;

      doc.setFontSize(8);
      doc.setFont(undefined, "bold");
      doc.setTextColor(...colors.muted);
      doc.text("Feature", margin + 2, y);
      doc.text("Contribution", margin + cw - 30, y);
      y += 5;

      doc.setDrawColor(...colors.bg);
      doc.line(margin, y - 1, margin + cw, y - 1);

      doc.setFont(undefined, "normal");
      explainData.breakdown.forEach((b) => {
        doc.setTextColor(...colors.dark);
        doc.text(b.feature, margin + 2, y + 3);
        const col = b.direction === "positive" ? colors.success : colors.error;
        doc.setTextColor(...col);
        const sign = b.contribution > 0 ? "+" : "";
        doc.text(`${sign}Rs ${b.contribution.toFixed(2)} L`, margin + cw - 30, y + 3);
        y += 5;
      });

      y += 3;
      doc.setDrawColor(...colors.primary);
      doc.line(margin, y, margin + cw, y);
      y += 5;
      doc.setTextColor(...colors.dark);
      doc.setFont(undefined, "bold");
      doc.text("Total", margin + 2, y);
      doc.text(`Rs ${explainData.predicted_price} L`, margin + cw - 30, y);
      y += 10;
    }

    // ── Model Explanation ──
    if (result) {
      if (y > 200) { doc.addPage(); y = 20; }

      const { MODEL_INFO, formatContribution } = await import("./ModelExplanation");
      const mInfo = MODEL_INFO[result.model_used] || MODEL_INFO.elasticnet;

      doc.setTextColor(...colors.dark);
      doc.setFontSize(12);
      doc.setFont(undefined, "bold");
      doc.text("Model Explanation", margin, y);
      y += 8;

      // Model identity box
      doc.setFillColor(...colors.bg);
      doc.roundedRect(margin, y - 3, cw, 20, 2, 2, "F");
      doc.setFontSize(10);
      doc.setTextColor(...colors.dark);
      doc.setFont(undefined, "bold");
      doc.text(mInfo.fullName, margin + 4, y + 3);
      doc.setFontSize(7);
      doc.setFont(undefined, "normal");
      doc.setTextColor(...colors.muted);
      doc.text(mInfo.type, margin + 4, y + 9);
      doc.setFontSize(6);
      doc.setTextColor(...colors.primary);
      doc.text(mInfo.formula, margin + 4, y + 14);
      y += 24;

      // How it works
      doc.setTextColor(...colors.dark);
      doc.setFontSize(8);
      doc.setFont(undefined, "bold");
      doc.text("How It Works", margin, y);
      y += 4;
      doc.setFont(undefined, "normal");
      doc.setFontSize(7);
      doc.setTextColor(80, 80, 80);
      const howLines = doc.splitTextToSize(mInfo.howItWorks, cw);
      doc.text(howLines, margin, y);
      y += howLines.length * 3.5 + 4;

      // Regularization
      doc.setTextColor(...colors.dark);
      doc.setFontSize(8);
      doc.setFont(undefined, "bold");
      doc.text("Regularization", margin, y);
      y += 4;
      doc.setFont(undefined, "normal");
      doc.setFontSize(7);
      doc.setTextColor(80, 80, 80);
      const regLines = doc.splitTextToSize(mInfo.regularization, cw);
      doc.text(regLines, margin, y);
      y += regLines.length * 3.5 + 4;

      // Strengths & Limitations
      doc.setFontSize(7);
      doc.setFont(undefined, "bold");
      doc.setTextColor(...colors.success);
      doc.text("Strengths: ", margin, y);
      doc.setFont(undefined, "normal");
      doc.setTextColor(80, 80, 80);
      const strLines = doc.splitTextToSize(mInfo.strengths, cw - 20);
      doc.text(strLines, margin + 20, y);
      y += strLines.length * 3.5 + 2;

      doc.setFont(undefined, "bold");
      doc.setTextColor(...colors.error);
      doc.text("Limitations: ", margin, y);
      doc.setFont(undefined, "normal");
      doc.setTextColor(80, 80, 80);
      const limLines = doc.splitTextToSize(mInfo.weaknesses, cw - 20);
      doc.text(limLines, margin + 20, y);
      y += limLines.length * 3.5 + 6;

      // Feature-by-feature rationale
      if (explainData?.breakdown) {
        if (y > 220) { doc.addPage(); y = 20; }

        doc.setTextColor(...colors.dark);
        doc.setFontSize(8);
        doc.setFont(undefined, "bold");
        doc.text("Price Rationale (Feature-by-Feature)", margin, y);
        y += 5;

        doc.setFontSize(6.5);
        doc.setFont(undefined, "normal");
        explainData.breakdown.forEach((b) => {
          if (y > 275) { doc.addPage(); y = 20; }
          const explanation = formatContribution(b);
          const col = b.direction === "positive" ? colors.success : colors.error;

          doc.setTextColor(...col);
          doc.setFont(undefined, "bold");
          const sign = b.contribution > 0 ? "+" : "";
          doc.text(`${b.feature}: ${sign}Rs ${b.contribution.toFixed(2)} L`, margin + 2, y);
          y += 3.5;
          doc.setFont(undefined, "normal");
          doc.setTextColor(80, 80, 80);
          const expLines = doc.splitTextToSize(explanation, cw - 4);
          doc.text(expLines, margin + 2, y);
          y += expLines.length * 3 + 3;
        });
      }

      // Training data context
      if (y > 250) { doc.addPage(); y = 20; }
      y += 2;
      doc.setFillColor(...colors.bg);
      doc.roundedRect(margin, y - 3, cw, 14, 2, 2, "F");
      doc.setFontSize(7);
      doc.setTextColor(...colors.dark);
      doc.setFont(undefined, "bold");
      doc.text("Training Data: ", margin + 3, y + 2);
      doc.setFont(undefined, "normal");
      doc.setTextColor(...colors.muted);
      doc.text("9,200+ samples | 262 features | 178 locations | 5-fold CV + GridSearchCV optimization", margin + 28, y + 2);
      doc.text("Final model retrained on complete dataset for maximum prediction accuracy.", margin + 3, y + 8);
      y += 18;
    }

    // ── Model Comparison ──
    if (compareData?.comparisons) {
      if (y > 230) { doc.addPage(); y = 20; }

      doc.setTextColor(...colors.dark);
      doc.setFontSize(12);
      doc.setFont(undefined, "bold");
      doc.text("Model Comparison", margin, y);
      y += 8;

      // Table header
      doc.setFillColor(...colors.bg);
      doc.rect(margin, y - 3, cw, 7, "F");
      doc.setFontSize(7);
      doc.setFont(undefined, "bold");
      doc.setTextColor(...colors.muted);
      const cols = [margin + 2, margin + 38, margin + 78, margin + 108, margin + 138];
      doc.text("Model", cols[0], y + 1);
      doc.text("Predicted Price", cols[1], y + 1);
      doc.text("R² Score", cols[2], y + 1);
      doc.text("MSE", cols[3], y + 1);
      doc.text("Features", cols[4], y + 1);
      y += 7;

      doc.setFont(undefined, "normal");
      compareData.comparisons.forEach((c) => {
        doc.setTextColor(...colors.dark);
        doc.text(c.is_best ? `* ${c.name}` : c.name, cols[0], y + 1);
        doc.text(`Rs ${c.predicted_price?.toFixed(2)} L`, cols[1], y + 1);
        doc.text(`${(c.r2 * 100).toFixed(2)}%`, cols[2], y + 1);
        doc.text(String(c.mse.toFixed(0)), cols[3], y + 1);
        doc.text(`${c.non_zero_features}/262`, cols[4], y + 1);
        y += 5;
      });

      y += 5;
      doc.setFontSize(7);
      doc.setTextColor(...colors.muted);
      doc.text("* indicates best performing model", margin + 2, y);
      y += 8;
    }

    // ── Footer ──
    const pageCount = doc.internal.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(7);
      doc.setTextColor(...colors.muted);
      doc.text(
        "House Price Prediction — CSE275 Project | Built with FastAPI + Next.js + scikit-learn",
        margin, 290
      );
      doc.text(`Page ${i} of ${pageCount}`, W - margin - 20, 290);
    }

    doc.save(`prediction_report_${Date.now()}.pdf`);
  }, [result, form, models, compareData, explainData]);

  if (!result) return null;

  return (
    <button onClick={generate}
      className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium
                 border border-border text-foreground/60 hover:bg-card-hover hover:text-foreground
                 active:scale-[0.98] transition-all cursor-pointer">
      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m.75 12l3 3m0 0l3-3m-3 3v-6m-1.5-9H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
      Export PDF Report
    </button>
  );
}
