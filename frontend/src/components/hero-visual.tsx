import { Layers, ShieldAlert, Timer } from "lucide-react";

/** Decorative orbital scene for the landing hero. */
export function HeroVisual() {
  return (
    <div className="relative mx-auto aspect-square w-full max-w-[460px]" aria-hidden>
      {/* planet */}
      <div className="absolute inset-[30%] rounded-full bg-[radial-gradient(circle_at_30%_30%,#8b9bff_0%,#3b46c4_35%,#111735_70%)] shadow-[0_0_80px_10px_rgb(109_124_255/0.35)]" />
      <div className="absolute inset-[30%] rounded-full bg-[radial-gradient(circle_at_70%_75%,transparent_45%,rgb(4_6_13/0.85)_75%)]" />

      {/* orbits */}
      {[
        { inset: "18%", duration: "18s", dot: "bg-accent-2", size: "size-2.5" },
        { inset: "8%", duration: "32s", dot: "bg-debris", size: "size-2" },
        { inset: "0%", duration: "46s", dot: "bg-craft", size: "size-3" },
      ].map((o, i) => (
        <div key={i} className="absolute rounded-full border border-white/[0.08]" style={{ inset: o.inset }}>
          <div className="absolute inset-0 animate-orbit" style={{ animationDuration: o.duration }}>
            <span
              className={`absolute top-1/2 -left-1 -translate-y-1/2 rounded-full ${o.dot} ${o.size} shadow-[0_0_14px_currentColor]`}
            />
          </div>
        </div>
      ))}

      {/* floating telemetry chips */}
      <div className="glass absolute top-[6%] right-[-2%] rounded-xl px-3.5 py-2.5 shadow-2xl">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold text-debris">
          <ShieldAlert className="size-3.5" /> CONFIDENT SPACE DEBRIS
        </p>
        <p className="mt-0.5 font-mono text-xs text-ink-muted">P(debris) 0.9776</p>
      </div>
      <div className="glass absolute bottom-[12%] left-[-4%] rounded-xl px-3.5 py-2.5 shadow-2xl">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold text-accent-2">
          <Layers className="size-3.5" /> Grad-CAM audit
        </p>
        <p className="mt-0.5 font-mono text-xs text-ink-muted">layer: top_conv</p>
      </div>
      <div className="glass absolute right-[2%] bottom-[2%] rounded-xl px-3.5 py-2.5 shadow-2xl">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold text-ink">
          <Timer className="size-3.5 text-accent" /> 1.42 ms / image
        </p>
        <p className="mt-0.5 font-mono text-xs text-ink-muted">Custom CNN · GPU</p>
      </div>
    </div>
  );
}
