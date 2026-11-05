import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Outlet } from "react-router-dom";
import { DemoNotice } from "./DemoNotice";
import type { ProviderStatus } from "@/lib/types";
import { api } from "@/lib/api";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

const ProviderCtx = createContext<ProviderStatus | undefined>(undefined);
const ProviderActionsCtx = createContext<{
  publishProviderStatus: (next: ProviderStatus) => void;
}>({ publishProviderStatus: () => undefined });
const COMPACT_QUERY = "(max-width: 767px)";

/** Access the latest secret-free provider status. */
export const useProvider = () => useContext(ProviderCtx);

/** Publish a secret-free provider snapshot to all routes without changing useProvider(). */
export const useProviderActions = () => useContext(ProviderActionsCtx);

export function selectLatestProviderStatus(
  current: ProviderStatus | undefined,
  next: ProviderStatus,
): ProviderStatus {
  if (!current) return next;
  const currentRevision = current.provider_config_revision;
  const nextRevision = next.provider_config_revision;
  if (currentRevision !== undefined && (nextRevision === undefined || nextRevision < currentRevision)) {
    return current;
  }
  return next;
}

export function AppShell() {
  const [provider, setProvider] = useState<ProviderStatus>();
  const [isCompact, setIsCompact] = useState(() => readCompactShell());
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement>(null);

  const publishProviderStatus = useCallback((next: ProviderStatus) => {
    setProvider((current) => selectLatestProviderStatus(current, next));
  }, []);
  const providerActions = useMemo(() => ({ publishProviderStatus }), [publishProviderStatus]);

  useEffect(() => {
    api.providers().then(publishProviderStatus).catch(() => undefined);
  }, [publishProviderStatus]);

  useEffect(() => {
    const media = window.matchMedia?.(COMPACT_QUERY);
    if (!media) return;
    const syncCompact = (event?: MediaQueryListEvent) => {
      const compact = event?.matches ?? media.matches;
      setIsCompact(compact);
      if (!compact) setMobileNavOpen(false);
    };
    syncCompact();
    media.addEventListener("change", syncCompact);
    return () => media.removeEventListener("change", syncCompact);
  }, []);

  const closeMobileNav = useCallback(() => {
    setMobileNavOpen(false);
    menuButtonRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!isCompact || !mobileNavOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeMobileNav();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [closeMobileNav, isCompact, mobileNavOpen]);

  return (
    <ProviderActionsCtx.Provider value={providerActions}>
      <ProviderCtx.Provider value={provider}>
        <div className="flex h-screen overflow-hidden bg-bg" data-testid="app-shell-root">
          {!isCompact && <Sidebar />}
      <div className="flex min-w-0 flex-1 flex-col">
        <DemoNotice />
            <TopBar
              provider={provider}
              compact={isCompact}
              onOpenNavigation={() => setMobileNavOpen(true)}
              menuButtonRef={menuButtonRef}
            />
            <main className="min-w-0 flex-1 overflow-y-auto" data-testid="app-shell-main">
              <Outlet />
            </main>
          </div>
          {isCompact && mobileNavOpen && (
            <>
              <button
                type="button"
                aria-label="Close navigation backdrop"
                data-testid="app-shell-nav-backdrop"
                className="fixed inset-0 z-40 cursor-default bg-black/40"
                onClick={closeMobileNav}
              />
              <div
                role="dialog"
                aria-modal="true"
                aria-label="Primary navigation"
                className="fixed inset-y-0 left-0 z-50 w-[min(20rem,85vw)] max-w-full shadow-xl"
              >
                <Sidebar variant="drawer" onClose={closeMobileNav} onNavigate={closeMobileNav} />
              </div>
            </>
          )}
        </div>
      </ProviderCtx.Provider>
    </ProviderActionsCtx.Provider>
  );
}

function readCompactShell(): boolean {
  return window.matchMedia?.(COMPACT_QUERY).matches ?? false;
}
