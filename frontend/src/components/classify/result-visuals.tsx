"use client";

import { motion } from "motion/react";
import { ShieldAlert, ShieldCheck, TriangleAlert, type LucideIcon } from "lucide-react";
import { UNCERTAIN_LOWER, UNCERTAIN_UPPER } from "@/utils/models";
import type { PredictionLabel } from "@/utils/types";

export const VERDICTS: Record<
  PredictionLabel,
  { title: string; subtitle: string; color: string; icon: LucideIcon }
> = {
  Debris: {
    title: "Space Debris",
    subtitle: "Confident space debris detection",
    color: "var(--color-debris)",
    icon: ShieldAlert,
  },
  "Non-Debris": {
    title: "Active Spacecraft",
    subtitle: "Confident non-debris spacecraft",
    color: "var(--color-craft)",
    icon: ShieldCheck,
  },
  Uncertain: {
    title: "Uncertain",
    subtitle: "Flagged for human or multi-sensor audit",
    color: "var(--color-uncertain)",
    icon: TriangleAlert,
  },
};

export function ConfidenceGauge({ value, color }: { value: number; color: string }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative size-36 shrink-0">
      <svg viewBox="0 0 120 120" className="size-full -rotate-90" aria-hidden>
        <circle cx="60" cy="60" r={r} fill="none" stroke="rgb(255 255 255 / 0.06)" strokeWidth="8" />
        <motion.circle
          cx="60"
          cy="60"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - value / 100) }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <p className="font-mono text-2xl font-semibold tracking-tight">{value.toFixed(1)}%</p>
          <p className="text-[10px] font-medium uppercase tracking-wider text-ink-faint">Confidence</p>
        </div>
      </div>
    </div>
  );
}

export function ProbabilityBars({ probDebris, probNonDebris }: { probDebris: number; probNonDebris: number }) {
  const rows = [
    { label: "P(Debris)", value: probDebris, color: "var(--color-debris)" },
    { label: "P(Non-Debris)", value: probNonDebris, color: "var(--color-craft)" },
  ];
  return (
    <div className="flex w-full flex-col gap-4">
      {rows.map((row) => (
        <div key={row.label}>
          <div className="mb-1.5 flex items-baseline justify-between text-xs">
            <span className="flex items-center gap-2 font-medium text-ink-muted">
              <span className="size-2 rounded-full" style={{ background: row.color }} />
              {row.label}
            </span>
            <span className="font-mono text-ink">{row.value.toFixed(4)}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
            <motion.div
              className="h-full rounded-full"
              style={{ background: row.color }}
              initial={{ width: 0 }}
              animate={{ width: `${row.value * 100}%` }}
              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

/** Visualizes DebrisPredictor's 3-state policy on the P(Non-Debris) axis. */
export function DecisionZone({
  probNonDebris,
  threshold,
  allowUncertain,
}: {
  probNonDebris: number;
  threshold: number;
  allowUncertain: boolean;
}) {
  const pct = (v: number) => `${v * 100}%`;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="font-medium text-ink-muted">Decision zone · P(Non-Debris)</span>
        <span className="font-mono text-ink-faint">τ = {threshold.toFixed(2)}</span>
      </div>
      <div className="relative pt-6 pb-5">
        <div className="relative flex h-3 gap-0.5">
          <div className="h-full rounded-l-full bg-debris/45" style={{ width: pct(threshold) }} />
          <div className="h-full flex-1 rounded-r-full bg-craft/45" />
          {allowUncertain && (
            <div
              className="absolute inset-y-0 border-x-2 border-space-800"
              style={{
                left: pct(UNCERTAIN_LOWER),
                width: pct(UNCERTAIN_UPPER - UNCERTAIN_LOWER),
                background:
                  "repeating-linear-gradient(135deg, rgb(245 158 11 / 0.75) 0 4px, rgb(245 158 11 / 0.35) 4px 8px)",
              }}
            />
          )}
        </div>
        {/* threshold tick */}
        <div className="absolute top-5 h-5 w-px bg-white/60" style={{ left: pct(threshold) }} />
        {/* value marker */}
        <motion.div
          className="absolute top-0 -translate-x-1/2"
          initial={{ left: "50%" }}
          animate={{ left: pct(probNonDebris) }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="rounded bg-white px-1.5 py-0.5 font-mono text-[10px] font-semibold text-space-900">
            {probNonDebris.toFixed(3)}
          </div>
          <div className="mx-auto h-0 w-0 border-x-4 border-t-4 border-x-transparent border-t-white" />
        </motion.div>
        <div className="absolute inset-x-0 bottom-0 flex justify-between font-mono text-[10px] text-ink-faint">
          <span>0.00 · Debris</span>
          {allowUncertain && <span className="text-uncertain">0.35 – 0.65 audit band</span>}
          <span>Spacecraft · 1.00</span>
        </div>
      </div>
    </div>
  );
}
