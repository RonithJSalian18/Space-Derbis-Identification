"use client";

import { Check, Cpu } from "lucide-react";
import { MODEL_ORDER, MODEL_PROFILES, formatParams } from "@/utils/models";
import type { ModelInfo, ModelKey } from "@/utils/types";
import { cn } from "@/utils/cn";

export function ModelSelector({
  models,
  value,
  onChange,
  disabled,
}: {
  models: Record<ModelKey, ModelInfo>;
  value: ModelKey;
  onChange: (key: ModelKey) => void;
  disabled?: boolean;
}) {
  return (
    <div role="radiogroup" aria-label="Model architecture" className="grid gap-3 sm:grid-cols-2">
      {MODEL_ORDER.map((key) => {
        const profile = MODEL_PROFILES[key];
        const info = models[key];
        const selected = value === key;
        const unavailable = !info.available;
        return (
          <button
            key={key}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled || unavailable}
            onClick={() => onChange(key)}
            className={cn(
              "group relative overflow-hidden rounded-xl border p-4 text-left transition-all",
              "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
              selected
                ? "border-accent/70 bg-accent/[0.08] shadow-[0_0_0_1px_rgb(109_124_255/0.35),0_12px_40px_-12px_rgb(109_124_255/0.45)]"
                : "border-white/[0.07] bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04]",
              (disabled || unavailable) && "cursor-not-allowed opacity-50"
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2.5">
                <span
                  className="grid size-8 place-items-center rounded-lg"
                  style={{ background: `color-mix(in oklab, ${profile.color} 18%, transparent)` }}
                >
                  <Cpu className="size-4" style={{ color: profile.color }} />
                </span>
                <div>
                  <p className="text-sm font-semibold leading-tight">{profile.name}</p>
                  <p className="text-[11px] text-ink-faint">{profile.tagline}</p>
                </div>
              </div>
              <span
                className={cn(
                  "grid size-5 shrink-0 place-items-center rounded-full border transition",
                  selected ? "border-accent bg-accent" : "border-white/20"
                )}
              >
                {selected && <Check className="size-3 text-white" strokeWidth={3} />}
              </span>
            </div>

            <dl className="mt-4 grid grid-cols-3 gap-2 font-mono text-[11px]">
              <div>
                <dt className="text-ink-faint">Params</dt>
                <dd className="mt-0.5 text-ink">{formatParams(info.total_params)}</dd>
              </div>
              <div>
                <dt className="text-ink-faint">Latency</dt>
                <dd className="mt-0.5 text-ink">{info.latency_ms} ms</dd>
              </div>
              <div>
                <dt className="text-ink-faint">Val acc</dt>
                <dd className="mt-0.5 text-ink">{info.accuracy}%</dd>
              </div>
            </dl>

            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="rounded-md bg-white/[0.06] px-2 py-0.5 text-[10px] font-medium text-ink-muted">
                {profile.input}
              </span>
              <span className="rounded-md bg-white/[0.06] px-2 py-0.5 text-[10px] font-medium text-ink-muted">
                {profile.tag}
              </span>
              {unavailable && (
                <span className="rounded-md bg-debris/15 px-2 py-0.5 text-[10px] font-medium text-debris">
                  Weights missing
                </span>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
