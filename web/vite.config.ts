/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    proxy: {
      // The AI engine (ai-engine/sentinel_ai/api/app.py) sends no CORS headers.
      // Proxying here keeps the browser request same-origin instead of patching
      // ai-engine, which is out of scope for this slice. See src/api/config.ts.
      '/engine': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/engine/, ''),
      },
      // The recorder appliance (`sentinel-ingest`, a DIFFERENT product from the
      // AI engine) likewise sends no CORS headers for this origin. Same fix,
      // separate prefix so the two products never share a path namespace:
      // `/recorder/api/alerts` -> `http://127.0.0.1:8080/api/alerts`.
      //
      // DEPLOYMENT: dev-only, exactly like `/engine` above. Production needs
      // CORS on the recorder, a reverse proxy in front of both, or this console
      // served from the recorder itself. See src/recorder/config.ts.
      '/recorder': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/recorder/, ''),
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    // Timestamps render in the operator's local zone. Pin the zone so an
    // assertion on a formatted `at_ns` does not depend on where CI runs.
    env: { TZ: 'UTC' },
  },
})
