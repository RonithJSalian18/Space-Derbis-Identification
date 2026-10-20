"use client";

/* eslint-disable @next/next/no-img-element -- Grad-CAM frames are base64 data URLs from the API */
import { useState } from "react";
import { Layers } from "lucide-react";
import type { GradCam } from "@/utils/types";
import { cn } from "@/utils/cn";

const VIEWS = [
  { key: "overlay", label: "Overlay" },
  { key: "heatmap", label: "Heatmap" },
  { key: "input", label: "Model input" },
] as const;

type View = (typeof VIEWS)[number]["key"];

export function GradCamViewer({ gradcam }: { gradcam: GradCam }) {
  const [view, setView] = useState<View>("overlay");

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Layers className="size-4 text-accent-2" />
          <p className="text-sm font-semibold">Grad-CAM attention audit</p>
        </div>
        <div className="flex gap-1 rounded-lg bg-white/[0.04] p-1 text-xs font-medium">
          {VIEWS.map((v) => (
            <button
              key={v.key}
              type="button"
              onClick={() => setView(v.key)}
              className={cn(
                "rounded-md px-2.5 py-1 transition",
                view === v.key ? "bg-white/10 text-ink" : "text-ink-faint hover:text-ink-muted"
              )}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-[1fr_auto]">
        <div className="relative aspect-square overflow-hidden rounded-xl bg-black ring-1 ring-white/10">
          <img src={gradcam[view]} alt={`Grad-CAM ${view}`} className="size-full object-cover" />
          <span className="absolute top-2 left-2 rounded-md bg-black/60 px-2 py-1 font-mono text-[10px] text-ink-muted backdrop-blur">
            layer: {gradcam.layer}
          </span>
        </div>

        <div className="flex gap-4 sm:flex-col">
          {VIEWS.filter((v) => v.key !== view).map((v) => (
            <button
              key={v.key}
              type="button"
              onClick={() => setView(v.key)}
              className="group w-24 text-left"
              aria-label={`Show ${v.label}`}
            >
              <div className="aspect-square overflow-hidden rounded-lg bg-black ring-1 ring-white/10 transition group-hover:ring-white/30">
                <img src={gradcam[v.key]} alt="" className="size-full object-cover" />
              </div>
              <p className="mt-1 text-[11px] text-ink-faint group-hover:text-ink-muted">{v.label}</p>
            </button>
          ))}
          <div className="hidden sm:block">
            <div
              className="h-2 w-24 rounded-full"
              style={{ background: "linear-gradient(90deg,#00007f,#0000ff,#00ffff,#ffff00,#ff0000,#7f0000)" }}
            />
            <div className="mt-1 flex w-24 justify-between text-[10px] text-ink-faint">
              <span>Low</span>
              <span>High</span>
            </div>
          </div>
        </div>
      </div>
      <p className="mt-3 text-xs leading-relaxed text-ink-faint">
        Warm regions show where the network&apos;s final convolutional features drove the decision. Attention on the
        target body (not empty space or borders) indicates a trustworthy prediction.
      </p>
    </div>
  );
}
