"use client";

import { useEffect, useState } from "react";
import { Info } from "lucide-react";
import { Card } from "@/components/ui";
import { getModels } from "@/utils/api";
import { FALLBACK_MODELS, MODEL_PROFILES } from "@/utils/models";
import type { ModelInfo } from "@/utils/types";
import { cn } from "@/utils/cn";
import { EfficiencyChart, ErrorRateChart, SizeBars } from "./benchmark-charts";

const COLUMNS: { key: keyof ModelInfo; label: string; format: (v: number) => string; best: "max" | "min" }[] = [
  { key: "accuracy", label: "Accuracy", format: (v) => `${v.toFixed(2)}%`, best: "max" },
  { key: "precision", label: "Precision", format: (v) => `${v.toFixed(2)}%`, best: "max" },
  { key: "recall", label: "Recall", format: (v) => `${v.toFixed(2)}%`, best: "max" },
  { key: "f1", label: "F1-score", format: (v) => `${v.toFixed(2)}%`, best: "max" },
  { key: "val_loss", label: "Val loss", format: (v) => v.toFixed(4), best: "min" },
  { key: "total_params", label: "Params", format: (v) => v.toLocaleString("en-US"), best: "min" },
  { key: "size_mb", label: "Size", format: (v) => `${v} MB`, best: "min" },
  { key: "latency_ms", label: "Latency", format: (v) => `${v} ms`, best: "min" },
];

export function BenchmarkView() {
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS);
  const [live, setLive] = useState(false);

  useEffect(() => {
    getModels()
      .then((m) => {
        setModels(m);
        setLive(true);
      })
      .catch(() => setLive(false));
  }, []);

  return (
    <div className="space-y-6">
      {!live && (
        <p className="flex items-center gap-2 text-xs text-ink-faint">
          <Info className="size-3.5" /> Showing recorded benchmark results. Start the API to load the latest metrics file.
        </p>
      )}

      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-left text-sm">
            <thead>
              <tr className="border-b border-white/5 text-[11px] tracking-wider text-ink-faint uppercase">
                <th className="px-6 py-4 font-medium">Architecture</th>
                {COLUMNS.map((c) => (
                  <th key={c.key} className="px-4 py-4 text-right font-medium">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const p = MODEL_PROFILES[m.key];
                return (
                  <tr key={m.key} className="border-b border-white/5 transition last:border-0 hover:bg-white/[0.02]">
                    <td className="px-6 py-4">
                      <span className="flex items-center gap-2.5">
                        <span className="size-2.5 rounded-full" style={{ background: p.color }} />
                        <span>
                          <span className="block font-semibold">{p.name}</span>
                          <span className="block text-xs text-ink-faint">{p.input}</span>
                        </span>
                      </span>
                    </td>
                    {COLUMNS.map((c) => {
                      const values = models.map((x) => x[c.key] as number);
                      const v = m[c.key] as number;
                      const isBest = v === (c.best === "max" ? Math.max(...values) : Math.min(...values));
                      return (
                        <td
                          key={c.key}
                          className={cn("px-4 py-4 text-right font-mono text-xs", isBest ? "text-ink" : "text-ink-muted")}
                        >
                          {isBest && <span className="mr-1.5 inline-block size-1.5 rounded-full bg-accent-2 align-middle" />}
                          {c.format(v)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="border-t border-white/5 px-6 py-3 text-xs text-ink-faint">
          <span className="mr-1.5 inline-block size-1.5 rounded-full bg-accent-2 align-middle" />
          Best in column. Peak validation metrics on the trajectory-isolated SPARK-2022 split.
        </p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <ErrorRateChart models={models} />
        <EfficiencyChart models={models} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <SizeBars models={models} />
        <Card className="p-5 sm:p-6">
          <h3 className="text-base font-semibold">Architecture notes</h3>
          <p className="mt-1 text-sm text-ink-muted">Input contract and fine-tuning strategy per backbone.</p>
          <div className="mt-6 divide-y divide-white/5">
            {models.map((m) => {
              const p = MODEL_PROFILES[m.key];
              return (
                <div key={m.key} className="grid gap-1 py-3 first:pt-0 last:pb-0 sm:grid-cols-[140px_1fr]">
                  <p className="flex items-center gap-2 text-sm font-medium">
                    <span className="size-2 rounded-full" style={{ background: p.color }} />
                    {p.name}
                  </p>
                  <p className="text-xs leading-relaxed text-ink-muted">
                    <span className="font-mono text-ink">{p.input}</span> · normalized {p.normalization} ·{" "}
                    {p.fineTune}. {p.description}
                  </p>
                </div>
              );
            })}
          </div>
        </Card>
      </div>
    </div>
  );
}
