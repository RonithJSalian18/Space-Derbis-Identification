import type { Metadata } from "next";
import { ClassifyWorkspace } from "@/components/classify/classify-workspace";
import { Eyebrow } from "@/components/ui";

export const metadata: Metadata = {
  title: "Classifier",
  description: "Upload orbital imagery and classify it as space debris or active spacecraft.",
};

export default function ClassifyPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 pt-10 pb-8 sm:px-6 lg:px-8">
      <div className="mb-8 max-w-2xl">
        <Eyebrow>Inference console</Eyebrow>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">Debris classifier</h1>
        <p className="mt-3 text-ink-muted">
          Select one of four trained architectures, submit an orbital image and inspect the calibrated 3-state verdict
          together with its Grad-CAM attention audit.
        </p>
      </div>
      <ClassifyWorkspace />
    </div>
  );
}
