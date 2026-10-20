import type { ModelInfo, ModelKey } from "./types";

export interface ModelProfile {
  key: ModelKey;
  name: string;
  tag: string;
  tagline: string;
  description: string;
  input: string;
  normalization: string;
  fineTune: string;
  color: string;
}

/** Static architecture notes (from README) — rendered even when the API is offline. */
export const MODEL_PROFILES: Record<ModelKey, ModelProfile> = {
  cnn: {
    key: "cnn",
    name: "Custom CNN",
    tag: "Fastest",
    tagline: "Ultra-lightweight edge inference",
    description: "4-stage Conv2D network built for power-constrained CubeSat payloads.",
    input: "224×224×1 Grayscale",
    normalization: "[0, 1]",
    fineTune: "Trained from scratch",
    color: "#3987e5",
  },
  mobilenet: {
    key: "mobilenet",
    name: "MobileNetV2",
    tag: "Embedded",
    tagline: "Depthwise inverted residuals",
    description: "Inverted residual depthwise blocks for fast embedded space hardware.",
    input: "224×224×3 RGB",
    normalization: "[-1, 1]",
    fineTune: "Top 2 inverted residual blocks",
    color: "#d95926",
  },
  efficientnet: {
    key: "efficientnet",
    name: "EfficientNet-B0",
    tag: "Most accurate",
    tagline: "Compound-scaled backbone",
    description: "Compound scaling for the best accuracy-to-parameter trade-off.",
    input: "224×224×3 RGB",
    normalization: "[0, 255] (internal rescale)",
    fineTune: "Top 2 MBConv blocks",
    color: "#199e70",
  },
  resnet: {
    key: "resnet",
    name: "ResNet-50",
    tag: "Deepest",
    tagline: "Deep residual features",
    description: "Deep residual skip connections for complex structural feature modeling.",
    input: "224×224×3 RGB",
    normalization: "BGR mean subtraction",
    fineTune: "conv5 stage",
    color: "#c98500",
  },
};

export const MODEL_ORDER: ModelKey[] = ["cnn", "mobilenet", "efficientnet", "resnet"];

/** Benchmark fallback mirroring plots/comparison/model_comparison_metrics.json. */
export const FALLBACK_MODELS: ModelInfo[] = [
  { key: "cnn", name: "Custom CNN", total_params: 422785, trainable_params: 421825, size_mb: 1.65, latency_ms: 1.42, val_loss: 0.0375, accuracy: 99.72, precision: 99.95, recall: 99.89, f1: 99.92, available: true, loaded: false },
  { key: "mobilenet", name: "MobileNetV2", total_params: 2427713, trainable_params: 1677633, size_mb: 9.49, latency_ms: 2.51, val_loss: 0.1277, accuracy: 99.78, precision: 99.95, recall: 99.95, f1: 99.95, available: true, loaded: false },
  { key: "efficientnet", name: "EfficientNet-B0", total_params: 4219300, trainable_params: 3296797, size_mb: 16.36, latency_ms: 3.36, val_loss: 0.1232, accuracy: 99.9, precision: 99.99, recall: 99.97, f1: 99.98, available: true, loaded: false },
  { key: "resnet", name: "ResNet-50", total_params: 23858817, trainable_params: 15220225, size_mb: 91.25, latency_ms: 4.05, val_loss: 0.1215, accuracy: 99.89, precision: 99.99, recall: 99.95, f1: 99.97, available: true, loaded: false },
];

export function formatParams(n: number) {
  return n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : `${(n / 1e3).toFixed(0)}K`;
}

export const UNCERTAIN_LOWER = 0.35;
export const UNCERTAIN_UPPER = 0.65;
