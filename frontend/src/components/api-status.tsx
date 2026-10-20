"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/utils/api";
import type { Health } from "@/utils/types";
import { cn } from "@/utils/cn";

type State = { kind: "checking" } | { kind: "online"; health: Health } | { kind: "offline" };

export function ApiStatus() {
  const [state, setState] = useState<State>({ kind: "checking" });

  useEffect(() => {
    let active = true;
    const check = async () => {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 4000);
      try {
        const health = await getHealth(ctrl.signal);
        if (active) setState({ kind: "online", health });
      } catch {
        if (active) setState({ kind: "offline" });
      } finally {
        clearTimeout(timer);
      }
    };
    check();
    const id = setInterval(check, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const label =
    state.kind === "online"
      ? `API online · ${state.health.gpu ? "GPU" : "CPU"}`
      : state.kind === "offline"
        ? "API offline"
        : "Connecting…";

  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-ink-muted"
      role="status"
    >
      <span className="relative flex size-2">
        {state.kind === "online" && (
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-craft opacity-60" />
        )}
        <span
          className={cn(
            "relative inline-flex size-2 rounded-full",
            state.kind === "online" && "bg-craft",
            state.kind === "offline" && "bg-debris",
            state.kind === "checking" && "bg-ink-faint"
          )}
        />
      </span>
      {label}
    </span>
  );
}
