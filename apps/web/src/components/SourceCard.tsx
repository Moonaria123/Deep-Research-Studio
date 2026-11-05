import { useEffect, useRef } from "react";
import { ExternalLink } from "lucide-react";
import { Link, useInRouterContext } from "react-router-dom";
import type { EvidenceSource } from "@/lib/types";
import { domainOf } from "@/lib/utils";
import { evidenceArtifactLabel, evidenceHref } from "@/lib/evidenceRoutes";
import { Badge } from "@/components/ui/badge";

export function SourceCard({
  index,
  source,
  active = false,
  onFocus,
  onPin,
}: {
  index: number;
  source: EvidenceSource;
  active?: boolean;
  onFocus?: (index: number | null) => void;
  onPin?: (index: number) => void;
}) {
  const ref = useRef<HTMLElement>(null);
  const inRouter = useInRouterContext();
  const target = evidenceHref(source);
  const title = source.title || (source.url ? domainOf(source.url) : evidenceArtifactLabel(source));
  const linkClass = "flex min-w-0 items-center gap-1 text-sm font-medium text-text hover:text-accent focus-ring";

  // Scroll into view when this card becomes the active (highlighted) source.
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [active]);

  return (
    <article
      ref={ref}
      id={`source-${index}`}
      data-anim
      data-source-active={active ? "true" : undefined}
      className={`rounded-md border bg-surface p-3 shadow-card transition-colors ${
        active ? "border-accent ring-1 ring-accent" : "border-border"
      }`}
      onMouseEnter={onFocus ? () => onFocus(index) : undefined}
      onMouseLeave={onFocus ? () => onFocus(null) : undefined}
      onClick={onPin ? () => onPin(index) : undefined}
    >
      <div className="flex items-start gap-2">
        <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-accent-soft text-[11px] font-semibold text-accent">
          {index}
        </span>
        <div className="min-w-0">
          {target ? (
            target.external || !inRouter ? (
              <a
                href={target.href}
                target={target.external ? "_blank" : undefined}
                rel={target.external ? "noreferrer" : undefined}
                className={linkClass}
              >
                <span className="truncate">{title}</span>
                {target.external ? <ExternalLink className="h-3.5 w-3.5 shrink-0" aria-hidden /> : null}
              </a>
            ) : (
              <Link to={target.href} className={linkClass}>
                <span className="truncate">{title}</span>
              </Link>
            )
          ) : (
            <p className="truncate text-sm font-medium text-text">{title}</p>
          )}
          <div className="mt-1 flex flex-wrap items-center gap-1">
            <Badge variant="default">origin: {source.origin ?? "external"}</Badge>
            <Badge variant="default">artifact: {evidenceArtifactLabel(source)}</Badge>
            {source.search_provider ? <Badge variant="default">search: {source.search_provider}</Badge> : null}
            {source.fetch_provider ? <Badge variant="default">fetch: {source.fetch_provider}</Badge> : null}
            {source.source_status === "archived" ? <Badge variant="warning">archived</Badge> : null}
          </div>
          {source.url ? <p className="mt-1 text-xs text-text-faint">{domainOf(source.url)}</p> : null}
          <p className="mt-1 line-clamp-2 text-xs text-text-muted">{source.text}</p>
        </div>
      </div>
    </article>
  );
}
