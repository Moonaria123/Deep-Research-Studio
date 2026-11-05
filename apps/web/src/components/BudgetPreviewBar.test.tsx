import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { BudgetPreviewBar } from "./BudgetPreviewBar";

describe("BudgetPreviewBar", () => {
  it("discloses the single conditional recovery allowance without inflating ordinary searches", () => {
    render(<BudgetPreviewBar mode="fast" />);

    expect(screen.getByText("Searches").parentElement).toHaveTextContent("1");
    expect(screen.getByText(/Up to 1 conditional focused recovery search/i)).toBeInTheDocument();
  });

  it("does not advertise T48 recovery for council runs", () => {
    render(<BudgetPreviewBar mode="council" />);

    expect(screen.queryByText(/conditional focused recovery search/i)).toBeNull();
  });
});
