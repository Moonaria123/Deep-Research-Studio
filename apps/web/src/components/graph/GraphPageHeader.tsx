import type { ReactNode } from "react";

type GraphPageHeaderProps = {
  icon: ReactNode;
  title: string;
  subtitle: string;
  badges: ReactNode;
  actions: ReactNode;
  statusStrip?: ReactNode;
};

export function GraphPageHeader({ icon, title, subtitle, badges, actions, statusStrip }: GraphPageHeaderProps) {
  return (
    <header aria-label="Graph page header" className="border-b border-border px-5 py-4 lg:px-6">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
        <div className="min-w-0">
          <h1 className="flex items-start gap-2 text-2xl font-bold text-text">
            {icon}
            <span className="min-w-0 [overflow-wrap:anywhere]">{title}</span>
          </h1>
          <p className="mt-1 text-sm text-text-muted">{subtitle}</p>
        </div>
        <div className="flex min-w-0 flex-col gap-2 sm:items-start xl:items-end">
          <div aria-label="Graph status badges" className="flex flex-wrap items-center gap-2">
            {badges}
          </div>
          <div aria-label="Graph actions" className="flex flex-wrap items-center gap-2">
            {actions}
          </div>
        </div>
      </div>
      {statusStrip ? <div className="mt-3">{statusStrip}</div> : null}
    </header>
  );
}
