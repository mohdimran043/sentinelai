/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ENGINE_API_URL?: string
  readonly VITE_EVENTS_API_URL?: string
  /** Base URL of the recorder appliance's API — a different product from the engine. */
  readonly VITE_RECORDER_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
