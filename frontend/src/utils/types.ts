export type ModelKey = "cnn" | "mobilenet" | "efficientnet" | "resnet";

export type PredictionLabel = "Debris" | "Non-Debris" | "Uncertain";

export interface ModelInfo {
  key: ModelKey;
  name: string;
  total_params: number;
  trainable_params: number;
  size_mb: number;
  latency_ms: number;
  val_loss: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  available: boolean;
  loaded: boolean;
}

export interface GradCam {
  layer: string;
  input: string;
  heatmap: string;
  overlay: string;
}

export interface Prediction {
  prediction: PredictionLabel;
  status: string;
  confidence: number;
  prob_debris: number;
  prob_non_debris: number;
  threshold_applied: number;
  model: ModelKey;
  filename: string | null;
  inference_ms: number;
  gradcam: GradCam | null;
  warning: string | null;
}

export interface Sample {
  name: string;
  label: "debris" | "non_debris";
  url: string;
}

export interface Health {
  status: string;
  loaded_models: ModelKey[];
  gpu: boolean;
}

export interface PredictOptions {
  model: ModelKey;
  threshold: number;
  allowUncertain: boolean;
  gradcam: boolean;
}
