/** Vite env typings — avoids requiring `vite/client` at the tsconfig level when node_modules is absent in the IDE. */
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
