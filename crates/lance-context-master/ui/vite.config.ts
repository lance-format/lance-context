import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the admin API to the running master process so the SPA
// can be developed with `npm run dev` while `lance-context-master` serves data.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: process.env.MASTER_URL ?? "http://localhost:8090",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
