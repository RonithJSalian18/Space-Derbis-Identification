"use client";

/* eslint-disable @next/next/no-img-element -- previews are local data URLs */
import { useCallback, useEffect, useRef, useState } from "react";
import { ImagePlus, RotateCcw } from "lucide-react";
import { cn } from "@/utils/cn";

const ACCEPTED = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
const MAX_BYTES = 15 * 1024 * 1024;

export function ImageDropzone({
  file,
  previewUrl,
  onFile,
  onError,
  scanning,
}: {
  file: File | null;
  previewUrl: string | null;
  onFile: (file: File | null) => void;
  onError: (message: string) => void;
  scanning?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const accept = useCallback(
    (f: File | undefined | null) => {
      if (!f) return;
      if (!ACCEPTED.includes(f.type)) return onError("Unsupported format. Use JPG, PNG, BMP or WebP.");
      if (f.size > MAX_BYTES) return onError("Image exceeds the 15 MB limit.");
      onFile(f);
    },
    [onFile, onError]
  );

  // Paste an image from the clipboard anywhere on the page
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith("image/"));
      if (item) accept(item.getAsFile());
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [accept]);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        accept(e.dataTransfer.files?.[0]);
      }}
      className={cn(
        "relative overflow-hidden rounded-xl border border-dashed transition-colors",
        dragging ? "border-accent bg-accent/[0.06]" : "border-white/15 bg-white/[0.02]",
        !previewUrl && "hover:border-white/25"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        className="sr-only"
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />

      {previewUrl && file ? (
        <div className="flex flex-col gap-4 p-4 sm:flex-row sm:items-center">
          <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-black sm:w-40">
            <img src={previewUrl} alt={`Preview of ${file.name}`} className="size-full object-cover" />
            {scanning && (
              <>
                <div className="absolute inset-0 bg-accent/10" />
                <div className="absolute inset-x-0 h-0.5 animate-scan bg-accent-2 shadow-[0_0_16px_4px_rgb(34_211_238/0.6)]" />
              </>
            )}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{file.name}</p>
            <p className="mt-0.5 font-mono text-xs text-ink-faint">
              {(file.size / 1024).toFixed(1)} KB · {file.type.replace("image/", "").toUpperCase()}
            </p>
            <div className="mt-3 flex gap-2">
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs font-medium text-ink-muted transition hover:bg-white/5 hover:text-ink"
              >
                <RotateCcw className="size-3.5" /> Replace
              </button>
              <button
                type="button"
                onClick={() => onFile(null)}
                className="rounded-lg px-3 py-1.5 text-xs font-medium text-ink-faint transition hover:text-ink"
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex w-full flex-col items-center justify-center gap-3 px-6 py-10 text-center focus-visible:outline-2 focus-visible:outline-accent"
        >
          <span className="grid size-12 place-items-center rounded-2xl bg-linear-to-br from-accent/25 to-accent-2/20 ring-1 ring-white/10">
            <ImagePlus className="size-5 text-accent-2" />
          </span>
          <span>
            <span className="block text-sm font-medium">
              Drop an orbital image, or <span className="text-accent-2">browse</span>
            </span>
            <span className="mt-1 block text-xs text-ink-faint">JPG, PNG, BMP or WebP up to 15 MB · paste with Ctrl+V</span>
          </span>
        </button>
      )}
    </div>
  );
}
