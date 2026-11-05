/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DEEPRESEARCH_API_ORIGIN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
