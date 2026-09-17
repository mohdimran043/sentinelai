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
      // **Scoped to `/recorder/api`, not `/recorder`**, and the difference is not
      // cosmetic. The console's own recorder screens live at `/recorder/report`,
      // `/recorder/storage` and so on — browser routes that React Router answers.
      // A rule matching bare `/recorder` captures those too and sends a page
      // navigation to the appliance, so every recorder screen renders as a 502
      // from a service that was never meant to serve HTML. `RECORDER_BASE_URL` is
      // `/recorder/api` (see src/recorder/config.ts), so this matches exactly the
      // prefix the client actually calls and leaves the route namespace alone.
      //
      // DEPLOYMENT: dev-only, exactly like `/engine` above. Production needs
      // CORS on the recorder, a reverse proxy in front of both, or this console
      // served from the recorder itself. See src/recorder/config.ts.
      '/recorder/api': {
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
