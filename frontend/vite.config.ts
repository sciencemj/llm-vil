import { defineConfig } from "vite";

const backend = process.env.LLM_VIL_BACKEND ?? "127.0.0.1:8780";

export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: `http://${backend}`, changeOrigin: true },
      "/ws": { target: `ws://${backend}`, ws: true },
    },
  },
  build: { target: "es2022", sourcemap: true },
});
