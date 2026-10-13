import tailwindcss from "@tailwindcss/vite"

export default defineNuxtConfig({
  compatibilityDate: "2026-10-01",
  ssr: false,
  devtools: { enabled: false },
  telemetry: false,
  css: ["~/assets/css/main.css"],
  vite: { plugins: [tailwindcss()] },
  devServer: { host: "localhost", port: 3000 },
  runtimeConfig: { public: { apiBase: "http://127.0.0.1:8000" } },
  app: {
    head: {
      title: "TVision — Tumor Nucleus Detection",
      htmlAttrs: { lang: "en" },
      meta: [{ name: "viewport", content: "width=device-width, initial-scale=1" }],
    },
  },
})