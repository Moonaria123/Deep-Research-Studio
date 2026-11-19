export const IS_DEMO = import.meta.env.MODE === "demo";
export const IS_DESKTOP = import.meta.env.MODE === "desktop";
export const EXTERNAL_LINK_TARGET = IS_DESKTOP ? "system-browser" : "_blank";

/** Desktop and static Demo builds use hash routes so deep links stay within one asset. */
export function usesHashRouting(mode: string): boolean {
  return mode === "demo" || mode === "desktop";
}

class MemoryStorage implements Storage {
  private values = new Map<string, string>();
  get length() { return this.values.size; }
  clear() { this.values.clear(); }
  getItem(key: string) { return this.values.get(key) ?? null; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string) { this.values.delete(key); }
  setItem(key: string, value: string) { this.values.set(key, String(value)); }
}

/** Install non-persistent storage before React mounts in the public preview build. */
export function installDemoRuntime(): void {
  if (!IS_DEMO) return;
  const memory = new MemoryStorage();
  Object.defineProperty(window, "localStorage", { configurable: true, value: memory });
  Object.defineProperty(window, "sessionStorage", { configurable: true, value: new MemoryStorage() });
  document.documentElement.dataset.demo = "true";
}
