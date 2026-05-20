import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import App from "../src/App";

const healthResponse = {
  model: "glm-5.1",
  baseUrl: "https://ark.cn-beijing.volces.com/api/coding/v3",
  knowledgeBasePath: "/mnt/d/workNote",
  markdownFileCount: 3,
  apiKeyConfigured: true
};

const chatResponse = {
  answer: "测试类路径规范见知识库。[S1]",
  sources: [
    {
      sourceId: "S1",
      filePath: "rule.md",
      title: "测试规范",
      lineStart: 1,
      lineEnd: 3,
      score: 0.42,
      content: "测试类必须放在 src/test/java。"
    }
  ],
  trace: [
    { action: "search_knowledge_base", content: "需要查知识库" },
    { action: "observation", content: "[S1] rule.md:1-3" }
  ]
};

describe("App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return new Response(JSON.stringify(healthResponse), { status: 200 });
      }
      if (url.endsWith("/api/chat")) {
        return new Response(JSON.stringify(chatResponse), { status: 200 });
      }
      return new Response("Not found", { status: 404 });
    });
  });

  test("renders runtime health summary", async () => {
    render(<App />);

    expect(await screen.findByText("GLM-5.1")).toBeInTheDocument();
    expect(screen.getByText("/mnt/d/workNote")).toBeInTheDocument();
    expect(screen.getByText("3 Markdown")).toBeInTheDocument();
  });

  test("submits a question and renders answer with source card", async () => {
    render(<App />);

    await userEvent.type(screen.getByLabelText("输入问题"), "测试类应该放哪里？");
    await userEvent.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("测试类路径规范见知识库。[S1]")).toBeInTheDocument();
    expect(screen.getByText("rule.md:1-3")).toBeInTheDocument();
    expect(screen.getByText("测试类必须放在 src/test/java。")).toBeInTheDocument();
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({ method: "POST" })
    ));
  });
});
