"use client";

/* eslint-disable @next/next/no-img-element -- thumbnails are served by the local inference API */
import { useEffect, useState } from "react";
import { getSamples, sampleUrl } from "@/utils/api";
import type { Sample } from "@/utils/types";
import { cn } from "@/utils/cn";

type Filter = "all" | Sample["label"];

export function SampleGallery({
  onPick,
  activeName,
  disabled,
}: {
  onPick: (sample: Sample) => void;
  activeName?: string;
  disabled?: boolean;
}) {
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");

  useEffect(() => {
    getSamples()
      .then(setSamples)
      .catch(() => setFailed(true));
  }, []);

  if (failed) {
    return <p className="text-xs text-ink-faint">Sample images become available once the inference API is running.</p>;
  }

  const visible = (samples ?? []).filter((s) => filter === "all" || s.label === filter);

  return (
    <div>
      <div className="mb-3 flex gap-1 rounded-lg bg-white/[0.03] p-1 text-xs font-medium">
        {(
          [
            ["all", "All"],
            ["debris", "Debris"],
            ["non_debris", "Spacecraft"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setFilter(key)}
            className={cn(
              "flex-1 rounded-md px-2 py-1.5 transition",
              filter === key ? "bg-white/10 text-ink" : "text-ink-faint hover:text-ink-muted"
            )}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-5 gap-2">
        {samples === null
          ? Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="aspect-square animate-pulse rounded-lg bg-white/[0.04]" />
            ))
          : visible.map((s) => (
              <button
                key={`${s.label}/${s.name}`}
                type="button"
                disabled={disabled}
                onClick={() => onPick(s)}
                title={`${s.name} · ${s.label === "debris" ? "Debris" : "Spacecraft"}`}
                className={cn(
                  "group relative aspect-square overflow-hidden rounded-lg bg-black ring-1 transition",
                  activeName === s.name ? "ring-2 ring-accent" : "ring-white/10 hover:ring-white/30",
                  disabled && "cursor-not-allowed opacity-60"
                )}
              >
                <img
                  src={sampleUrl(s)}
                  alt={s.name}
                  loading="lazy"
                  crossOrigin="anonymous"
                  className="size-full object-cover transition group-hover:scale-105"
                />
                <span
                  className={cn(
                    "absolute right-1 bottom-1 size-1.5 rounded-full",
                    s.label === "debris" ? "bg-debris" : "bg-craft"
                  )}
                />
              </button>
            ))}
      </div>
    </div>
  );
}
