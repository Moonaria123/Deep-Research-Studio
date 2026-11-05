import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Sidebar } from "./Sidebar";

vi.mock("./ThemeToggle", () => ({
  ThemeToggle: () => <button type="button">Theme</button>,
}));

function stubLocalStorage() {
  const store = new Map<string, string>();
  const storage = {
    getItem: vi.fn((key: string) => store.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => store.set(key, value)),
    removeItem: vi.fn((key: string) => store.delete(key)),
    clear: vi.fn(() => store.clear()),
  };
  vi.stubGlobal("localStorage", storage);
}

function renderSidebar() {
  return render(
    <MemoryRouter>
      <Sidebar />
    </MemoryRouter>,
  );
}

describe("Sidebar", () => {
  beforeEach(() => {
    stubLocalStorage();
  });

  it("collapses to icon-only rail when the collapse button is clicked", () => {
    renderSidebar();
    expect(screen.getByText("Deep Research Studio")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "History" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Knowledge Graph" })).toHaveAttribute("href", "/graph");

    fireEvent.click(screen.getByRole("button", { name: "Collapse sidebar" }));

    expect(screen.queryByText("Deep Research Studio")).toBeNull();
    expect(screen.queryByText("History")).toBeNull();
    expect(screen.queryByText("Knowledge Graph")).toBeNull();
    expect(screen.getByTitle("History")).toBeInTheDocument();
    expect(screen.getByTitle("Knowledge Graph")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand sidebar" })).toBeInTheDocument();
  });

  it("expands again when the expand button is clicked", () => {
    renderSidebar();
    fireEvent.click(screen.getByRole("button", { name: "Collapse sidebar" }));
    fireEvent.click(screen.getByRole("button", { name: "Expand sidebar" }));

    expect(screen.getByText("Deep Research Studio")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "History" })).toBeInTheDocument();
  });

  it("renders the drawer variant expanded without desktop collapse controls", () => {
    const onClose = vi.fn();
    const onNavigate = vi.fn();
    localStorage.setItem("deepresearch-sidebar-collapsed", "1");

    render(
      <MemoryRouter>
        <Sidebar variant="drawer" onClose={onClose} onNavigate={onNavigate} />
      </MemoryRouter>,
    );

    expect(screen.getByText("Deep Research Studio")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "History" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Collapse sidebar" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Expand sidebar" })).toBeNull();
    expect(localStorage.getItem("deepresearch-sidebar-collapsed")).toBe("1");

    fireEvent.click(screen.getByRole("button", { name: "Close navigation" }));
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("link", { name: "History" }));
    expect(onNavigate).toHaveBeenCalledTimes(1);
  });
});
