"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, Satellite, X } from "lucide-react";
import { ApiStatus } from "./api-status";
import { cn } from "@/utils/cn";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/classify", label: "Classify" },
  { href: "/models", label: "Benchmarks" },
] as const;

export function Navbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-white/5 bg-space-900/70 backdrop-blur-xl">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="group flex items-center gap-2.5" onClick={() => setOpen(false)}>
          <span className="grid size-9 place-items-center rounded-xl bg-linear-to-br from-accent to-accent-2 shadow-lg shadow-accent/25">
            <Satellite className="size-5 text-white" strokeWidth={2} />
          </span>
          <span className="leading-tight">
            <span className="block text-[15px] font-semibold tracking-tight">SpaceGuard Vision</span>
            <span className="block text-[11px] font-medium uppercase tracking-[0.18em] text-ink-faint">
              Debris Identification
            </span>
          </span>
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                "rounded-lg px-3.5 py-2 text-sm font-medium transition-colors",
                pathname === l.href ? "bg-white/[0.07] text-ink" : "text-ink-muted hover:text-ink"
              )}
            >
              {l.label}
            </Link>
          ))}
        </div>

        <div className="hidden items-center gap-3 md:flex">
          <ApiStatus />
          <Link
            href="/classify"
            className="rounded-lg bg-white px-4 py-2 text-sm font-semibold text-space-900 transition hover:bg-white/90"
          >
            Launch classifier
          </Link>
        </div>

        <button
          className="grid size-10 place-items-center rounded-lg text-ink-muted hover:bg-white/5 md:hidden"
          onClick={() => setOpen((o) => !o)}
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
        >
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </nav>

      {open && (
        <div className="border-t border-white/5 px-4 pb-4 md:hidden">
          <div className="flex flex-col gap-1 pt-3">
            {LINKS.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className={cn(
                  "rounded-lg px-3 py-2.5 text-sm font-medium",
                  pathname === l.href ? "bg-white/[0.07] text-ink" : "text-ink-muted"
                )}
              >
                {l.label}
              </Link>
            ))}
          </div>
          <div className="mt-3 px-3">
            <ApiStatus />
          </div>
        </div>
      )}
    </header>
  );
}
