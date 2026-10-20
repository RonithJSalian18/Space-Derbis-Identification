"use client";

import { AnimatePresence, motion } from "motion/react";
import { CircleAlert, LoaderCircle, Satellite } from "lucide-react";
import { Card, Stat } from "@/components/ui";
import { MODEL_PROFILES } from "@/utils/models";
import type { Prediction } from "@/utils/types";
import { ConfidenceGauge, DecisionZone, ProbabilityBars, VERDICTS } from "./result-visuals";
import { GradCamViewer } from "./gradcam-viewer";

export function PredictionResult({
  result,
  loading,
  loadingLabel,
  error,
  allowUncertain,
}: {
  result: Prediction | null;
  loading: boolean;
  loadingLabel: string;
  error: string | null;
  allowUncertain: boolean;
}) {
  return (
    <Card className="relative min-h-[520px] overflow-hidden">
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div key="loading" {...fade} className="grid min-h-[520px] place-items-center p-8">
            <div className="text-center">
              <LoaderCircle className="mx-auto size-8 animate-spin text-accent" />
              <p className="mt-4 text-sm font-medium">{loadingLabel}</p>
              <p className="mt-1 text-xs text-ink-faint">
                The first request for a model loads its weights. This can take a few seconds.
              </p>
            </div>
          </motion.div>
        ) : error ? (
          <motion.div key="error" {...fade} className="grid min-h-[520px] place-items-center p-8">
            <div className="max-w-sm text-center">
              <span className="mx-auto grid size-12 place-items-center rounded-2xl bg-debris/10">
                <CircleAlert className="size-6 text-debris" />
              </span>
              <p className="mt-4 text-sm font-semibold">Analysis failed</p>
              <p className="mt-1 text-sm text-ink-muted">{error}</p>
              <p className="mt-4 rounded-lg bg-white/[0.03] px-3 py-2 font-mono text-[11px] text-ink-faint">
                python -m uvicorn api.server:app --port 8000
              </p>
            </div>
          </motion.div>
        ) : result ? (
          <motion.div key={`${result.model}-${result.filename}-${result.prob_non_debris}`} {...fade}>
            <ResultBody result={result} allowUncertain={allowUncertain} />
          </motion.div>
        ) : (
          <motion.div key="empty" {...fade} className="grid min-h-[520px] place-items-center p-8">
            <EmptyState />
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

const fade = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
  transition: { duration: 0.25 },
};

function ResultBody({ result, allowUncertain }: { result: Prediction; allowUncertain: boolean }) {
  const verdict = VERDICTS[result.prediction];
  const Icon = verdict.icon;
  const profile = MODEL_PROFILES[result.model];

  return (
    <div>
      <div
        className="border-b border-white/5 p-6"
        style={{ background: `linear-gradient(135deg, color-mix(in oklab, ${verdict.color} 14%, transparent), transparent 70%)` }}
      >
        <div className="flex items-start gap-4">
          <span
            className="grid size-12 shrink-0 place-items-center rounded-2xl"
            style={{ background: `color-mix(in oklab, ${verdict.color} 20%, transparent)` }}
          >
            <Icon className="size-6" style={{ color: verdict.color }} />
          </span>
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: verdict.color }}>
              {result.status}
            </p>
            <h3 className="mt-1 text-2xl font-semibold tracking-tight">{verdict.title}</h3>
            <p className="text-sm text-ink-muted">{verdict.subtitle}</p>
          </div>
        </div>
      </div>

      <div className="space-y-6 p-6">
        <div className="flex flex-col items-center gap-6 sm:flex-row">
          <ConfidenceGauge value={result.confidence} color={verdict.color} />
          <ProbabilityBars probDebris={result.prob_debris} probNonDebris={result.prob_non_debris} />
        </div>

        <DecisionZone
          probNonDebris={result.prob_non_debris}
          threshold={result.threshold_applied}
          allowUncertain={allowUncertain}
        />

        <div className="grid grid-cols-2 gap-4 rounded-xl bg-white/[0.02] p-4 ring-1 ring-white/5 sm:grid-cols-4">
          <Stat label="Model" value={profile.name} />
          <Stat label="Round trip" value={`${result.inference_ms.toFixed(0)} ms`} />
          <Stat label="Threshold" value={result.threshold_applied.toFixed(2)} />
          <Stat label="Input" value={profile.input.split(" ")[1]} hint={profile.input.split(" ")[0]} />
        </div>

        {result.gradcam ? (
          <GradCamViewer gradcam={result.gradcam} />
        ) : result.warning ? (
          <p className="rounded-lg bg-uncertain/10 px-3 py-2 text-xs text-uncertain">{result.warning}</p>
        ) : null}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="text-center">
      <div className="relative mx-auto size-40">
        <div className="absolute inset-0 rounded-full border border-white/10" />
        <div className="absolute inset-6 rounded-full border border-dashed border-white/10" />
        <div className="absolute inset-0 animate-orbit">
          <span className="absolute top-1/2 -left-1.5 size-3 -translate-y-1/2 rounded-full bg-accent-2 shadow-[0_0_12px_rgb(34_211_238)]" />
        </div>
        <div className="absolute inset-0 grid place-items-center">
          <span className="grid size-14 place-items-center rounded-2xl bg-linear-to-br from-accent/30 to-accent-2/20 ring-1 ring-white/10">
            <Satellite className="size-7 text-ink" />
          </span>
        </div>
      </div>
      <p className="mt-6 text-base font-semibold">Awaiting telemetry</p>
      <p className="mx-auto mt-1 max-w-xs text-sm text-ink-muted">
        Choose a model, upload an image or pick a sample, then run the analysis.
      </p>
    </div>
  );
}
