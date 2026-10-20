"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Card } from "@/components/ui";
import { MODEL_ORDER, MODEL_PROFILES, formatParams } from "@/utils/models";
import type { ModelInfo, ModelKey } from "@/utils/types";

const AXIS = { stroke: "rgb(255 255 255 / 0.08)", tick: { fill: "#9aa3b8", fontSize: 11 } };
const GRID = "rgb(255 255 255 / 0.05)";

type TooltipEntry = { name?: string; value?: number | string; color?: string; payload?: Record<string, unknown> };

function ChartTooltip({
  active,
  payload,
  label,
  unit,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string | number;
  unit: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-white/10 bg-space-800/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      {label !== undefined && <p className="mb-1.5 font-medium text-ink">{label}</p>}
      {payload.map((p) => (
        <p key={p.name} className="flex items-center gap-2 text-ink-muted">
          <span className="size-2 rounded-full" style={{ background: p.color }} />
          {p.name}
          <span className="ml-auto pl-4 font-mono text-ink">
            {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
            {unit}
          </span>
        </p>
      ))}
    </div>
  );
}

function ChartCard({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <Card className="p-5 sm:p-6">
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="mt-1 text-sm text-ink-muted">{description}</p>
      <div className="mt-6">{children}</div>
    </Card>
  );
}

/** Error rates (100 − metric) so differences are visible on a zero-based axis. */
export function ErrorRateChart({ models }: { models: ModelInfo[] }) {
  const byKey = Object.fromEntries(models.map((m) => [m.key, m])) as Record<ModelKey, ModelInfo>;
  const metrics = [
    { label: "Error (1 − Acc)", field: "accuracy" },
    { label: "Miss rate (1 − Recall)", field: "recall" },
    { label: "False alarm (1 − Prec)", field: "precision" },
    { label: "1 − F1", field: "f1" },
  ] as const;
  const data = metrics.map((m) => ({
    metric: m.label,
    ...Object.fromEntries(MODEL_ORDER.map((k) => [k, +(100 - byKey[k][m.field]).toFixed(2)])),
  }));

  return (
    <ChartCard
      title="Validation error rates"
      description="Lower is better. Shown as 100 − metric so the gaps between models near 99.9% stay readable."
    >
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} barGap={2} barCategoryGap="22%" margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke={GRID} />
            <XAxis dataKey="metric" {...AXIS} tickLine={false} interval={0} tick={{ ...AXIS.tick, fontSize: 10 }} />
            <YAxis {...AXIS} tickLine={false} axisLine={false} unit="%" />
            <Tooltip cursor={{ fill: "rgb(255 255 255 / 0.03)" }} content={<ChartTooltip unit="%" />} />
            <Legend
              iconType="circle"
              iconSize={8}
              itemSorter={null}
              formatter={(value: string) => <span style={{ color: "#9aa3b8" }}>{value}</span>}
              wrapperStyle={{ fontSize: 12, paddingTop: 12 }}
            />
            {MODEL_ORDER.map((k) => (
              <Bar
                key={k}
                dataKey={k}
                name={MODEL_PROFILES[k].name}
                fill={MODEL_PROFILES[k].color}
                radius={[4, 4, 0, 0]}
                maxBarSize={22}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

export function EfficiencyChart({ models }: { models: ModelInfo[] }) {
  return (
    <ChartCard
      title="Latency vs. model size"
      description="GPU latency per image against total parameters (log scale). Bottom-left is most efficient."
    >
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 16, right: 32, left: -12, bottom: 8 }}>
            <CartesianGrid stroke={GRID} />
            <XAxis
              type="number"
              dataKey="params"
              name="Parameters"
              scale="log"
              domain={[0.3, 40]}
              ticks={[0.5, 1, 2, 5, 10, 20]}
              tickFormatter={(v: number) => `${v}M`}
              {...AXIS}
              tickLine={false}
            />
            <YAxis
              type="number"
              dataKey="latency"
              name="Latency"
              unit=" ms"
              domain={[0, 5]}
              {...AXIS}
              tickLine={false}
              axisLine={false}
            />
            <ZAxis range={[140, 140]} />
            <Tooltip
              cursor={{ stroke: "rgb(255 255 255 / 0.15)" }}
              content={({ active, payload }) => {
                const p = payload?.[0]?.payload as { name: string; params: number; latency: number; color: string } | undefined;
                if (!active || !p) return null;
                return (
                  <div className="rounded-lg border border-white/10 bg-space-800/95 px-3 py-2 text-xs shadow-xl">
                    <p className="mb-1 flex items-center gap-2 font-medium text-ink">
                      <span className="size-2 rounded-full" style={{ background: p.color }} />
                      {p.name}
                    </p>
                    <p className="text-ink-muted">
                      Params <span className="font-mono text-ink">{p.params.toFixed(2)}M</span>
                    </p>
                    <p className="text-ink-muted">
                      Latency <span className="font-mono text-ink">{p.latency} ms</span>
                    </p>
                  </div>
                );
              }}
            />
            {models.map((m) => {
              const profile = MODEL_PROFILES[m.key];
              return (
                <Scatter
                  key={m.key}
                  name={profile.name}
                  data={[{ name: profile.name, params: m.total_params / 1e6, latency: m.latency_ms, color: profile.color }]}
                  fill={profile.color}
                  stroke="#0c1222"
                  strokeWidth={2}
                >
                  <LabelList dataKey="name" position="top" offset={12} fill="#e8ecf5" fontSize={11} />
                </Scatter>
              );
            })}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

export function SizeBars({ models }: { models: ModelInfo[] }) {
  const max = Math.max(...models.map((m) => m.size_mb));
  return (
    <ChartCard title="Checkpoint footprint" description="On-disk .h5 size. This matters for uplink bandwidth and onboard storage.">
      <div className="space-y-4">
        {models.map((m) => {
          const p = MODEL_PROFILES[m.key];
          return (
            <div key={m.key} className="group">
              <div className="mb-1.5 flex items-baseline justify-between text-sm">
                <span className="flex items-center gap-2 font-medium">
                  <span className="size-2 rounded-full" style={{ background: p.color }} />
                  {p.name}
                </span>
                <span className="font-mono text-xs text-ink-muted">
                  {m.size_mb} MB · {formatParams(m.total_params)} params
                </span>
              </div>
              <div className="h-2.5 rounded-full bg-white/[0.04]">
                <div
                  className="h-full rounded-full transition-[filter] group-hover:brightness-125"
                  style={{ width: `${Math.max((m.size_mb / max) * 100, 1.5)}%`, background: p.color }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}
