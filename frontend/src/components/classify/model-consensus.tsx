"use client";

import { Card } from "@/components/ui";
import { MODEL_ORDER, MODEL_PROFILES } from "@/utils/models";
import type { ModelKey, Prediction } from "@/utils/types";
import { VERDICTS } from "./result-visuals";

export type ConsensusRow = { model: ModelKey; result?: Prediction; error?: string };

export function ModelConsensus({ rows, filename }: { rows: ConsensusRow[]; filename: string }) {
  const done = rows.filter((r) => r.result);
  const votes = done.reduce<Record<string, number>>((acc, r) => {
    acc[r.result!.prediction] = (acc[r.result!.prediction] ?? 0) + 1;
    return acc;
  }, {});
  const leader = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];

  return (
    <Card className="p-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <p className="text-sm font-semibold">Model consensus</p>
          <p className="mt-0.5 truncate text-xs text-ink-faint">{filename}</p>
        </div>
        {leader && done.length === rows.length && (
          <p className="text-xs text-ink-muted">
            <span className="font-mono text-ink">
              {leader[1]}/{rows.length}
            </span>{" "}
            models agree:{" "}
            <span className="font-semibold" style={{ color: VERDICTS[leader[0] as Prediction["prediction"]].color }}>
              {VERDICTS[leader[0] as Prediction["prediction"]].title}
            </span>
          </p>
        )}
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[440px] text-left text-sm">
          <thead className="text-[11px] uppercase tracking-wider text-ink-faint">
            <tr className="border-b border-white/5">
              <th className="py-2 font-medium">Model</th>
              <th className="py-2 font-medium">Verdict</th>
              <th className="py-2 text-right font-medium">P(Non-Debris)</th>
              <th className="py-2 text-right font-medium">Confidence</th>
              <th className="py-2 text-right font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_ORDER.map((key) => rows.find((r) => r.model === key))
              .filter((r): r is ConsensusRow => Boolean(r))
              .map((row) => {
                const profile = MODEL_PROFILES[row.model];
                const v = row.result ? VERDICTS[row.result.prediction] : null;
                return (
                  <tr key={row.model} className="border-b border-white/5 last:border-0">
                    <td className="py-2.5">
                      <span className="flex items-center gap-2 font-medium">
                        <span className="size-2 rounded-full" style={{ background: profile.color }} />
                        {profile.name}
                      </span>
                    </td>
                    <td className="py-2.5">
                      {v ? (
                        <span
                          className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium"
                          style={{ color: v.color, background: `color-mix(in oklab, ${v.color} 14%, transparent)` }}
                        >
                          <v.icon className="size-3.5" /> {v.title}
                        </span>
                      ) : row.error ? (
                        <span className="text-xs text-debris">{row.error}</span>
                      ) : (
                        <span className="inline-block h-4 w-24 animate-pulse rounded bg-white/[0.06]" />
                      )}
                    </td>
                    <td className="py-2.5 text-right font-mono text-xs">{row.result?.prob_non_debris.toFixed(4) ?? "—"}</td>
                    <td className="py-2.5 text-right font-mono text-xs">
                      {row.result ? `${row.result.confidence.toFixed(2)}%` : "—"}
                    </td>
                    <td className="py-2.5 text-right font-mono text-xs text-ink-muted">
                      {row.result ? `${row.result.inference_ms.toFixed(0)} ms` : "—"}
                    </td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
