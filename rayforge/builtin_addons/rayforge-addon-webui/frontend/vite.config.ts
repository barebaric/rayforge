/// <reference types="vitest/config" />
import { defineConfig } from "vite";

const nodeProcess = (
  globalThis as typeof globalThis & {
    process?: { env: Record<string, string | undefined> };
  }
).process;
const apiPort = nodeProcess?.env.RF_API_PORT || "8132";

export default defineConfig({
  base: "/",
  build: {
    outDir: "../webui/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${apiPort}`,
        ws: true,
      },
    },
  },
  test: {
    environment: "happy-dom",
  },
});
