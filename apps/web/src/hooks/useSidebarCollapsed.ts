import { useCallback, useEffect, useState } from "react";

const KEY = "deepresearch-sidebar-collapsed";

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

function writeCollapsed(collapsed: boolean): void {
  try {
    localStorage.setItem(KEY, collapsed ? "1" : "0");
  } catch {
    // Ignore storage failures (private mode, test env).
  }
}

/** Persisted sidebar rail (icon-only) vs expanded layout. */
export function useSidebarCollapsed(): { collapsed: boolean; toggle: () => void } {
  const [collapsed, setCollapsed] = useState(readCollapsed);

  useEffect(() => {
    writeCollapsed(collapsed);
  }, [collapsed]);

  const toggle = useCallback(() => setCollapsed((value) => !value), []);
  return { collapsed, toggle };
}
