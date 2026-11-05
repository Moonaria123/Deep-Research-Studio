export const IS_DEMO = import.meta.env.MODE === "demo";

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
