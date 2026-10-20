import type { Health, ModelInfo, Prediction, PredictOptions, Sample } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(`Cannot reach the inference API at ${API_URL}. Is the server running?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return res.json() as Promise<T>;
}

export const getHealth = (signal?: AbortSignal) => request<Health>("/api/health", { signal });

export const getModels = () => request<{ models: ModelInfo[] }>("/api/models").then((r) => r.models);

export const getSamples = () => request<{ samples: Sample[] }>("/api/samples").then((r) => r.samples);

export const sampleUrl = (sample: Sample) => `${API_URL}${sample.url}`;

export async function fetchSampleFile(sample: Sample): Promise<File> {
  let res: Response;
  try {
    res = await fetch(sampleUrl(sample), { cache: "no-store" });
  } catch {
    throw new ApiError(`Could not download sample ${sample.name} from ${API_URL}.`);
  }
  if (!res.ok) throw new ApiError(`Could not load sample ${sample.name}`, res.status);
  const blob = await res.blob();
  return new File([blob], sample.name, { type: blob.type || "image/jpeg" });
}

export function predict(file: File, opts: PredictOptions): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", opts.model);
  form.append("threshold", String(opts.threshold));
  form.append("allow_uncertain", String(opts.allowUncertain));
  form.append("gradcam", String(opts.gradcam));
  return request<Prediction>("/api/predict", { method: "POST", body: form });
}
