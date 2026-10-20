import type { Metadata } from "next";
import { Eyebrow } from "@/components/ui";
import { BenchmarkView } from "@/components/models/benchmark-view";

export const metadata: Metadata = {
  title: "Benchmarks",
  description: "Accuracy, latency and footprint comparison of the four space debris classifiers.",
};

export default function ModelsPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 pt-10 pb-8 sm:px-6 lg:px-8">
      <div className="mb-8 max-w-2xl">
        <Eyebrow>Model benchmarks</Eyebrow>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">Architecture comparison</h1>
        <p className="mt-3 text-ink-muted">
          All four classifiers exceed 99.7% validation accuracy. The real trade-off is latency, footprint and
          miss rate, which matters most when a missed fragment means a collision.
        </p>
      </div>
      <BenchmarkView />
    </div>
  );
}
