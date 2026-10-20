import Link from "next/link";
import { Satellite } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-24 border-t border-white/5">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-10 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8">
        <div className="flex items-center gap-3">
          <span className="grid size-8 place-items-center rounded-lg bg-white/5">
            <Satellite className="size-4 text-accent" />
          </span>
          <div>
            <p className="text-sm font-semibold">SpaceGuard Vision</p>
            <p className="text-xs text-ink-faint">
              Built on the SPARK-2022 dataset (Stanford SLAB &amp; ESA) · MIT License
            </p>
          </div>
        </div>
        <div className="flex gap-6 text-sm text-ink-muted">
          <Link href="/classify" className="hover:text-ink">Classify</Link>
          <Link href="/models" className="hover:text-ink">Benchmarks</Link>
          <a
            href="https://github.com/RonithJSalian18/Space-Derbis-Identification"
            className="hover:text-ink"
            target="_blank"
            rel="noreferrer"
          >
            Source code
          </a>
        </div>
      </div>
    </footer>
  );
}
