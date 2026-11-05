import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SearchUsageEditor } from "./SearchUsageEditor";

describe("SearchUsageEditor", () => {
  it("keeps Brave out of the extract selector and exposes the locked local tail", () => {
    const onChange = vi.fn();
    render(
      <SearchUsageEditor
        value={{
          usage_mode: "dual_role",
          search_provider: "brave",
          extract_provider: "tavily",
          fallback_chain: ["brave", "tavily", "exa", "firecrawl"],
        }}
        onChange={onChange}
      />,
    );

    const extract = screen.getByRole("combobox", { name: /extract provider/i }) as HTMLSelectElement;
    expect(Array.from(extract.options).map((option) => option.value)).toEqual(["tavily", "exa", "firecrawl"]);
    expect(screen.getByText(/cache → indexed_external/i)).toBeInTheDocument();
  });

  it("reorders the single external fallback chain accessibly", () => {
    const onChange = vi.fn();
    render(
      <SearchUsageEditor
        value={{ usage_mode: "auto", fallback_chain: ["brave", "tavily", "exa", "firecrawl"] }}
        onChange={onChange}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /move exa up/i }));
    expect(onChange).toHaveBeenCalledWith({
      usage_mode: "auto",
      fallback_chain: ["brave", "exa", "tavily", "firecrawl"],
    });
  });

  it("shows configured and capability state without exposing credentials", () => {
    render(
      <SearchUsageEditor
        value={{ usage_mode: "auto", fallback_chain: ["brave", "tavily"] }}
        capabilities={{
          brave: { configured: true, search_ready: true, extract_ready: false },
          tavily: { configured: false, search_ready: false, extract_ready: false },
        }}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByText(/brave/i).parentElement).toHaveTextContent(/configured.*search only/i);
    expect(screen.getByText(/tavily/i).parentElement).toHaveTextContent(/not configured.*search only/i);
  });

  it("preserves an unavailable existing choice but blocks it as a new selection", () => {
    render(
      <SearchUsageEditor
        value={{ usage_mode: "single", primary_provider: "brave", fallback_chain: ["brave"] }}
        capabilities={{
          brave: { configured: true, search_ready: true, extract_ready: false, selectable: false },
          tavily: { configured: true, search_ready: true, extract_ready: true, selectable: false },
          exa: { configured: true, search_ready: true, extract_ready: true, selectable: true },
          firecrawl: { configured: false, search_ready: false, extract_ready: false, selectable: false },
        }}
        onChange={vi.fn()}
      />,
    );
    const primary = screen.getByRole("combobox", { name: /primary provider/i }) as HTMLSelectElement;
    expect(primary.value).toBe("brave");
    expect(primary.querySelector('option[value="brave"]')).not.toBeDisabled();
    expect(primary.querySelector('option[value="tavily"]')).toBeDisabled();
    expect(primary.querySelector('option[value="exa"]')).not.toBeDisabled();
  });

  it("hides disabled providers while reordering the retained canonical chain", () => {
    const onChange = vi.fn();
    const capabilities = {
      brave: { configured: true, search_ready: true, extract_ready: false, enabled: true, selectable: true },
      firecrawl: { configured: true, search_ready: true, extract_ready: true, enabled: false, selectable: false },
      tavily: { configured: true, search_ready: true, extract_ready: true, enabled: true, selectable: true },
      exa: { configured: true, search_ready: true, extract_ready: true, enabled: true, selectable: true },
    } as const;
    const { rerender } = render(
      <SearchUsageEditor
        value={{ usage_mode: "auto", fallback_chain: ["brave", "firecrawl", "tavily", "exa"] }}
        capabilities={capabilities}
        onChange={onChange}
      />,
    );

    const fallback = screen.getByRole("list", { name: /external fallback order/i });
    expect(within(fallback).queryByText(/firecrawl/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /move tavily up/i }));
    expect(onChange).toHaveBeenLastCalledWith({
      usage_mode: "auto",
      fallback_chain: ["tavily", "firecrawl", "brave", "exa"],
    });

    rerender(
      <SearchUsageEditor
        value={{ usage_mode: "auto", fallback_chain: ["tavily", "firecrawl", "brave", "exa"] }}
        capabilities={{ ...capabilities, firecrawl: { ...capabilities.firecrawl, enabled: true, selectable: true } }}
        onChange={onChange}
      />,
    );
    expect(
      within(screen.getByRole("list", { name: /external fallback order/i }))
        .getAllByRole("listitem")
        .map((item) => item.textContent?.match(/brave|tavily|exa|firecrawl/i)?.[0]?.toLowerCase()),
    ).toEqual(["tavily", "firecrawl", "brave", "exa"]);
  });

  it("keeps a disabled current role as an explicit fallback placeholder", () => {
    render(
      <SearchUsageEditor
        value={{
          usage_mode: "single",
          primary_provider: "firecrawl",
          fallback_chain: ["brave", "firecrawl", "tavily"],
        }}
        capabilities={{
          brave: { configured: true, search_ready: true, extract_ready: false, enabled: true, selectable: true },
          firecrawl: { configured: true, search_ready: true, extract_ready: true, enabled: false, selectable: false },
          tavily: { configured: true, search_ready: true, extract_ready: true, enabled: true, selectable: true },
          exa: { configured: true, search_ready: true, extract_ready: true, enabled: false, selectable: false },
        }}
        onChange={vi.fn()}
      />,
    );

    const primary = screen.getByRole("combobox", { name: /primary provider/i }) as HTMLSelectElement;
    expect(primary.value).toBe("firecrawl");
    expect(primary.querySelector('option[value="firecrawl"]')).toHaveTextContent(/disabled; runtime uses fallback/i);
    expect(primary.querySelector('option[value="firecrawl"]')).toBeDisabled();
    expect(primary.querySelector('option[value="exa"]')).toBeNull();
    expect(within(screen.getByRole("list", { name: /external fallback order/i })).queryByText(/firecrawl/i)).toBeNull();
  });

  it("removes and adds visible providers without deleting a hidden canonical entry", () => {
    const onChange = vi.fn();
    const capabilities = {
      brave: { configured: true, search_ready: true, extract_ready: false, enabled: true, selectable: true },
      firecrawl: { configured: true, search_ready: true, extract_ready: true, enabled: false, selectable: false },
      tavily: { configured: true, search_ready: true, extract_ready: true, enabled: true, selectable: true },
      exa: { configured: true, search_ready: true, extract_ready: true, enabled: true, selectable: true },
    } as const;
    const { rerender } = render(
      <SearchUsageEditor
        value={{ usage_mode: "auto", fallback_chain: ["brave", "firecrawl", "tavily"] }}
        capabilities={capabilities}
        onChange={onChange}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /remove brave/i }));
    expect(onChange).toHaveBeenLastCalledWith({
      usage_mode: "auto",
      fallback_chain: ["firecrawl", "tavily"],
    });

    rerender(
      <SearchUsageEditor
        value={{ usage_mode: "auto", fallback_chain: ["firecrawl", "tavily"] }}
        capabilities={capabilities}
        onChange={onChange}
      />,
    );
    const add = screen.getByRole("combobox", { name: /add fallback provider/i }) as HTMLSelectElement;
    expect(Array.from(add.options).map((option) => option.value)).toEqual(["", "brave", "exa"]);
    fireEvent.change(add, { target: { value: "exa" } });
    expect(onChange).toHaveBeenLastCalledWith({
      usage_mode: "auto",
      fallback_chain: ["firecrawl", "tavily", "exa"],
    });
  });

  it("blocks single and dual-role mode changes when no eligible provider exists", () => {
    render(
      <SearchUsageEditor
        value={{ usage_mode: "auto", fallback_chain: ["brave", "tavily", "exa", "firecrawl"] }}
        capabilities={{
          brave: { configured: true, search_ready: true, extract_ready: false, enabled: false, selectable: false },
          tavily: { configured: true, search_ready: true, extract_ready: true, enabled: false, selectable: false },
          exa: { configured: true, search_ready: true, extract_ready: true, enabled: false, selectable: false },
          firecrawl: { configured: true, search_ready: true, extract_ready: true, enabled: false, selectable: false },
        }}
        onChange={vi.fn()}
      />,
    );

    const mode = screen.getByRole("combobox", { name: /search usage mode/i }) as HTMLSelectElement;
    expect(mode.querySelector('option[value="single"]')).toBeDisabled();
    expect(mode.querySelector('option[value="dual_role"]')).toBeDisabled();
    expect(screen.getByText(/enable an eligible external search provider/i)).toBeInTheDocument();
  });
});
