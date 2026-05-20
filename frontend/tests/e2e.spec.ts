import { expect, test } from "@playwright/test";

test("renders the WebUI shell with runtime health", async ({ page }) => {
  await page.goto("http://127.0.0.1:5173");

  await expect(page.getByRole("heading", { name: "ReAct 知识库问答" })).toBeVisible();
  await expect(page.getByText("GLM-5.1", { exact: true })).toBeVisible();
  await expect(page.getByText("/mnt/d/workNote")).toBeVisible();
});
