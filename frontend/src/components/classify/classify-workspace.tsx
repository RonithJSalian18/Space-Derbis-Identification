"use client";

/* eslint-disable @next/next/no-img-element -- history thumbnails are local data URLs */
import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, GitCompareArrows, History, ScanEye, SlidersHorizontal } from "lucide-react";
import { Card, Switch } from "@/components/ui";
import { fetchSampleFile, getModels, predict } from "@/utils/api";
import { FALLBACK_MODELS, MODEL_ORDER, MODEL_PROFILES } from "@/utils/models";
import type { ModelInfo, ModelKey, Prediction, Sample } from "@/utils/types";
import { cn } from "@/utils/cn";
import { ModelSelector } from "./model-selector";
import { ImageDropzone } from "./image-dropzone";
import { SampleGallery } from "./sample-gallery";
import { PredictionResult } from "./prediction-result";
import { ModelConsensus, type ConsensusRow } from "./model-consensus";
import { VERDICTS } from "./result-visuals";

type HistoryItem = { id: number; thumb: string; result: Prediction; allowUncertain: boolean };

const toModelMap = (list: ModelInfo[]) =>
  Object.fromEntries(list.map((m) => [m.key, m])) as Record<ModelKey, ModelInfo>;

function readAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export function ClassifyWorkspace() {
  const [models, setModels] = useState(() => toModelMap(FALLBACK_MODELS));
  const [modelKey, setModelKey] = useState<ModelKey>("efficientnet");

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeSample, setActiveSample] = useState<string>();

  const [threshold, setThreshold] = useState(0.5);
  const [allowUncertain, setAllowUncertain] = useState(true);
  const [gradcam, setGradcam] = useState(true);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [resultUncertain, setResultUncertain] = useState(true);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [consensus, setConsensus] = useState<{ filename: string; rows: ConsensusRow[] } | null>(null);
  const [comparing, setComparing] = useState(false);

  const busy = loading || comparing;
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getModels()
      .then((list) => setModels(toModelMap(list)))
      .catch(() => {
        /* keep static fallback metrics; errors surface on analyze */
      });
  }, []);

  const selectFile = useCallback(async (f: File | null, sampleName?: string) => {
    setFile(f);
    setActiveSample(sampleName);
    setResult(null);
    setConsensus(null);
    setError(null);
    setPreviewUrl(f ? await readAsDataUrl(f) : null);
  }, []);

  const pickSample = async (sample: Sample) => {
    try {
      await selectFile(await fetchSampleFile(sample), sample.name);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    // On stacked (mobile) layouts the result panel sits below the controls
    if (window.matchMedia("(max-width: 1023px)").matches) {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    try {
      const res = await predict(file, { model: modelKey, threshold, allowUncertain, gradcam });
      setResult(res);
      setResultUncertain(allowUncertain);
      setModels((m) => ({ ...m, [modelKey]: { ...m[modelKey], loaded: true } }));
      setHistory((h) =>
        [{ id: Date.now(), thumb: previewUrl ?? "", result: res, allowUncertain }, ...h].slice(0, 8)
      );
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const compareAll = async () => {
    if (!file) return;
    const keys = MODEL_ORDER.filter((k) => models[k].available);
    setComparing(true);
    setConsensus({ filename: file.name, rows: keys.map((model) => ({ model })) });
    for (const key of keys) {
      let row: ConsensusRow;
      try {
        row = { model: key, result: await predict(file, { model: key, threshold, allowUncertain, gradcam: false }) };
      } catch (e) {
        row = { model: key, error: (e as Error).message };
      }
      setConsensus((c) => c && { ...c, rows: c.rows.map((r) => (r.model === key ? row : r)) });
    }
    setComparing(false);
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
      {/* Controls */}
      <div className="flex flex-col gap-6">
        <Card className="p-5 sm:p-6">
          <StepLabel n={1} title="Choose an architecture" />
          <ModelSelector models={models} value={modelKey} onChange={setModelKey} disabled={busy} />
        </Card>

        <Card className="p-5 sm:p-6">
          <StepLabel n={2} title="Provide an image" />
          <ImageDropzone
            file={file}
            previewUrl={previewUrl}
            onFile={(f) => selectFile(f)}
            onError={setError}
            scanning={busy}
          />
          <div className="mt-5">
            <p className="mb-2 text-xs font-medium text-ink-muted">Or try a SPARK-2022 sample</p>
            <SampleGallery onPick={pickSample} activeName={activeSample} disabled={busy} />
          </div>
        </Card>

        <Card className="p-5 sm:p-6">
          <button
            type="button"
            onClick={() => setAdvancedOpen((o) => !o)}
            className="flex w-full items-center justify-between"
            aria-expanded={advancedOpen}
          >
            <span className="flex items-center gap-2 text-sm font-semibold">
              <SlidersHorizontal className="size-4 text-ink-muted" /> Inference settings
            </span>
            <ChevronDown className={cn("size-4 text-ink-muted transition", advancedOpen && "rotate-180")} />
          </button>
          {advancedOpen && (
            <div className="mt-5 space-y-5">
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label htmlFor="threshold" className="text-sm font-medium">
                    Decision threshold (τ)
                  </label>
                  <span className="font-mono text-sm text-accent-2">{threshold.toFixed(2)}</span>
                </div>
                <input
                  id="threshold"
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.01}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="range w-full"
                />
                <p className="mt-1.5 text-xs text-ink-faint">
                  P(Non-Debris) above τ is classified as spacecraft. Default 0.50 matches the CLI.
                </p>
              </div>
              <Switch
                checked={allowUncertain}
                onChange={setAllowUncertain}
                label="Uncertainty flagging"
                description="Flag 0.35 ≤ P(Non-Debris) ≤ 0.65 for human audit"
              />
              <Switch
                checked={gradcam}
                onChange={setGradcam}
                label="Grad-CAM explainability"
                description="Render attention heatmaps for the prediction"
              />
            </div>
          )}
        </Card>

        <div className="flex flex-col gap-3 sm:flex-row">
          <button
            type="button"
            onClick={analyze}
            disabled={!file || busy}
            className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-linear-to-r from-accent to-[#5865f2] px-5 py-3.5 text-sm font-semibold text-white shadow-lg shadow-accent/25 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
          >
            <ScanEye className="size-4" />
            Analyze with {MODEL_PROFILES[modelKey].name}
          </button>
          <button
            type="button"
            onClick={compareAll}
            disabled={!file || busy}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 text-sm font-semibold text-ink transition hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <GitCompareArrows className="size-4" />
            Compare all models
          </button>
        </div>
      </div>

      {/* Results */}
      <div ref={resultsRef} className="flex scroll-mt-20 flex-col gap-6 lg:sticky lg:top-24 lg:self-start">
        <PredictionResult
          result={result}
          loading={loading}
          loadingLabel={`Running ${MODEL_PROFILES[modelKey].name} inference…`}
          error={error}
          allowUncertain={resultUncertain}
        />

        {consensus && <ModelConsensus rows={consensus.rows} filename={consensus.filename} />}

        {history.length > 0 && (
          <Card className="p-5">
            <p className="mb-3 flex items-center gap-2 text-sm font-semibold">
              <History className="size-4 text-ink-muted" /> Session history
            </p>
            <div className="flex gap-3 overflow-x-auto pb-1">
              {history.map((h) => {
                const v = VERDICTS[h.result.prediction];
                return (
                  <button
                    key={h.id}
                    type="button"
                    onClick={() => {
                      setResult(h.result);
                      setResultUncertain(h.allowUncertain);
                      setError(null);
                    }}
                    className={cn(
                      "w-24 shrink-0 text-left",
                      result === h.result ? "opacity-100" : "opacity-70 hover:opacity-100"
                    )}
                  >
                    <div
                      className="aspect-square overflow-hidden rounded-lg bg-black ring-2"
                      style={{ ["--tw-ring-color" as string]: result === h.result ? v.color : "rgb(255 255 255 / 0.08)" }}
                    >
                      {h.thumb && <img src={h.thumb} alt="" className="size-full object-cover" />}
                    </div>
                    <p className="mt-1.5 truncate text-[11px] font-medium" style={{ color: v.color }}>
                      {v.title}
                    </p>
                    <p className="truncate text-[10px] text-ink-faint">{MODEL_PROFILES[h.result.model].name}</p>
                  </button>
                );
              })}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

function StepLabel({ n, title }: { n: number; title: string }) {
  return (
    <div className="mb-4 flex items-center gap-3">
      <span className="grid size-6 place-items-center rounded-full bg-white/[0.07] font-mono text-xs text-ink-muted">
        {n}
      </span>
      <h2 className="text-sm font-semibold">{title}</h2>
    </div>
  );
}
