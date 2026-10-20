import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  Brain,
  Cpu,
  Flame,
  Fingerprint,
  GitBranch,
  ScanEye,
  ShieldAlert,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { Card, Eyebrow, SectionHeading } from "@/components/ui";
import { FadeIn } from "@/components/fade-in";
import { HeroVisual } from "@/components/hero-visual";
import { FALLBACK_MODELS, MODEL_PROFILES, formatParams } from "@/utils/models";

const STATS = [
  { value: "110K", label: "SPARK-2022 orbital images" },
  { value: "4", label: "Deep learning architectures" },
  { value: "99.90%", label: "Peak validation accuracy" },
  { value: "1.42 ms", label: "Fastest GPU latency / image" },
];

const PIPELINE = [
  {
    icon: GitBranch,
    title: "Trajectory-based splitting",
    body: "GroupShuffleSplit by trajectory ID guarantees zero frame leakage across 70/15/15 train, validation and test partitions.",
  },
  {
    icon: Fingerprint,
    title: "Perceptual-hash deduplication",
    body: "Parallel 64-bit pHash removes near-identical frozen frames (Hamming distance ≤ 2) before training.",
  },
  {
    icon: Flame,
    title: "Space-physics augmentation",
    body: "Solar glare, cosmic-ray sensor noise, photometric jitter, zero-g rotations and cutout simulate harsh orbital optics.",
  },
  {
    icon: Brain,
    title: "Two-phase transfer learning",
    body: "Head warm-up with a frozen backbone, then selective fine-tuning with BatchNorm layers locked in inference mode.",
  },
  {
    icon: ScanEye,
    title: "Calibrated, explainable inference",
    body: "Leak-free threshold calibration, ECE and Brier scoring, a 3-state uncertainty policy and Grad-CAM audits.",
  },
];

const POLICY = [
  {
    icon: ShieldAlert,
    color: "var(--color-debris)",
    title: "Confident space debris",
    rule: "P(Non-Debris) < 0.35",
    body: "Rocket bodies, structural fragments and mission-related debris.",
  },
  {
    icon: TriangleAlert,
    color: "var(--color-uncertain)",
    title: "Uncertain: flagged",
    rule: "0.35 ≤ P ≤ 0.65",
    body: "Ambiguous, low-illumination frames are routed to human or multi-sensor audit.",
  },
  {
    icon: ShieldCheck,
    color: "var(--color-craft)",
    title: "Confident spacecraft",
    rule: "P(Non-Debris) > τ*",
    body: "Active satellites and spacecraft that pose no debris risk.",
  },
];

export default function Home() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="mx-auto grid max-w-7xl items-center gap-12 px-4 pt-16 pb-20 sm:px-6 lg:grid-cols-2 lg:px-8 lg:pt-24">
          <FadeIn>
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs font-medium text-ink-muted">
              <span className="size-1.5 rounded-full bg-accent-2" />
              Space Domain Awareness · Computer Vision
            </span>
            <h1 className="mt-6 text-4xl leading-[1.1] font-semibold tracking-tight sm:text-5xl lg:text-6xl">
              Identify orbital debris <span className="text-gradient">with confidence.</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-muted">
              SpaceGuard Vision separates space debris from active spacecraft using four deep learning architectures
              trained on SPARK-2022, with calibrated uncertainty and visual explanations for every decision.
            </p>
            <div className="mt-10 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/classify"
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-white px-6 py-3.5 text-sm font-semibold text-space-900 transition hover:bg-white/90"
              >
                Start classifying <ArrowRight className="size-4" />
              </Link>
              <Link
                href="/models"
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-6 py-3.5 text-sm font-semibold transition hover:bg-white/[0.07]"
              >
                View benchmarks
              </Link>
            </div>
          </FadeIn>
          <FadeIn delay={0.15}>
            <HeroVisual />
          </FadeIn>
        </div>
      </section>

      {/* Stats */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <Card className="grid grid-cols-2 divide-white/5 lg:grid-cols-4 lg:divide-x">
          {STATS.map((s) => (
            <div key={s.label} className="p-6 sm:p-8">
              <p className="font-mono text-3xl font-semibold tracking-tight sm:text-4xl">{s.value}</p>
              <p className="mt-2 text-sm text-ink-muted">{s.label}</p>
            </div>
          ))}
        </Card>
      </section>

      {/* Pipeline */}
      <section className="mx-auto max-w-7xl px-4 pt-28 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Engineering pipeline"
          title="Built for scientific validity"
          description="Every stage is designed to prevent leakage, survive orbital imaging conditions and keep predictions auditable."
        />
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {PIPELINE.map((step, i) => (
            <FadeIn key={step.title} delay={i * 0.06}>
              <Card className="h-full p-6">
                <div className="flex items-center justify-between">
                  <span className="grid size-10 place-items-center rounded-xl bg-accent/10 ring-1 ring-accent/20">
                    <step.icon className="size-5 text-accent" />
                  </span>
                  <span className="font-mono text-xs text-ink-faint">0{i + 1}</span>
                </div>
                <h3 className="mt-5 text-base font-semibold">{step.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-muted">{step.body}</p>
              </Card>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* Models */}
      <section className="mx-auto max-w-7xl px-4 pt-28 sm:px-6 lg:px-8">
        <div className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <SectionHeading
            eyebrow="Model zoo"
            title="Four architectures, one interface"
            description="From a 0.42M-parameter edge CNN to ResNet-50, choose the trade-off between latency and representational depth."
          />
          <Link href="/models" className="inline-flex items-center gap-1.5 text-sm font-semibold text-accent-2 hover:underline">
            Compare benchmarks <ArrowRight className="size-4" />
          </Link>
        </div>
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {FALLBACK_MODELS.map((m, i) => {
            const p = MODEL_PROFILES[m.key];
            return (
              <FadeIn key={m.key} delay={i * 0.06}>
                <Card className="group h-full p-6 transition hover:border-white/15">
                  <div className="flex items-center justify-between">
                    <span
                      className="grid size-10 place-items-center rounded-xl"
                      style={{ background: `color-mix(in oklab, ${p.color} 18%, transparent)` }}
                    >
                      <Cpu className="size-5" style={{ color: p.color }} />
                    </span>
                    <span className="rounded-md bg-white/[0.06] px-2 py-0.5 text-[11px] font-medium text-ink-muted">
                      {p.tag}
                    </span>
                  </div>
                  <h3 className="mt-5 text-lg font-semibold">{p.name}</h3>
                  <p className="mt-1 text-sm text-ink-muted">{p.description}</p>
                  <dl className="mt-6 grid grid-cols-3 gap-2 border-t border-white/5 pt-4 font-mono text-xs">
                    <div>
                      <dt className="text-ink-faint">Params</dt>
                      <dd className="mt-1">{formatParams(m.total_params)}</dd>
                    </div>
                    <div>
                      <dt className="text-ink-faint">F1</dt>
                      <dd className="mt-1">{m.f1}%</dd>
                    </div>
                    <div>
                      <dt className="text-ink-faint">Latency</dt>
                      <dd className="mt-1">{m.latency_ms}ms</dd>
                    </div>
                  </dl>
                </Card>
              </FadeIn>
            );
          })}
        </div>
      </section>

      {/* Decision policy */}
      <section className="mx-auto max-w-7xl px-4 pt-28 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Safety-critical decisions"
          title="A calibrated 3-state policy"
          description="Forcing binary answers on ambiguous imagery risks missed collisions. Uncertain frames are escalated instead of guessed."
        />
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {POLICY.map((p) => (
            <Card key={p.title} className="relative overflow-hidden p-6">
              <div className="absolute inset-x-0 top-0 h-px" style={{ background: p.color }} />
              <p.icon className="size-6" style={{ color: p.color }} />
              <h3 className="mt-4 text-base font-semibold">{p.title}</h3>
              <p className="mt-1 font-mono text-xs text-ink-faint">{p.rule}</p>
              <p className="mt-3 text-sm leading-relaxed text-ink-muted">{p.body}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto max-w-7xl px-4 pt-28 sm:px-6 lg:px-8">
        <Card className="grid-lines relative overflow-hidden px-6 py-14 text-center sm:px-12">
          <div className="pointer-events-none absolute -top-24 left-1/2 h-48 w-[36rem] -translate-x-1/2 rounded-full bg-accent/25 blur-3xl" />
          <Boxes className="relative mx-auto size-8 text-accent-2" />
          <Eyebrow className="relative mt-5">Ready for launch</Eyebrow>
          <h2 className="relative mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">Run your first classification</h2>
          <p className="relative mx-auto mt-4 max-w-xl text-ink-muted">
            Upload an orbital frame or pick a SPARK-2022 sample, choose an architecture and see the verdict in
            milliseconds.
          </p>
          <Link
            href="/classify"
            className="relative mt-8 inline-flex items-center gap-2 rounded-xl bg-white px-6 py-3.5 text-sm font-semibold text-space-900 transition hover:bg-white/90"
          >
            Open the classifier <ArrowRight className="size-4" />
          </Link>
        </Card>
      </section>
    </>
  );
}
