import type { ReactNode } from "react";

type GraphToolbarProps = {
  tabs: ReactNode;
  sessionFilter: ReactNode;
  viewControls: ReactNode;
  search: ReactNode;
  typeFilters: ReactNode;
};

export function GraphToolbar({ tabs, sessionFilter, viewControls, search, typeFilters }: GraphToolbarProps) {
  return (
    <div className="border-b border-border bg-surface px-5 py-3 lg:px-6">
      <div
        aria-label="Graph toolbar"
        data-layout="graph-toolbar-grid"
        className="grid gap-3 lg:grid-cols-[auto_minmax(180px,auto)_minmax(0,1fr)] lg:items-center xl:grid-cols-[auto_auto_auto_minmax(260px,1fr)]"
      >
        <div className="min-w-0">{tabs}</div>
        <div className="min-w-0">{sessionFilter}</div>
        <div className="min-w-0">{viewControls}</div>
        <div className="min-w-0">{search}</div>
        <div className="min-w-0 lg:col-span-3 xl:col-span-4">{typeFilters}</div>
      </div>
    </div>
  );
}
