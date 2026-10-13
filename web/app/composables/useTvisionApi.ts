import type { DetectionResult, ServiceInfo } from "~/utils/detections"

export class ApiError extends Error {}

function detailMessage(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === "string") return detail
  if (Array.isArray(detail)) return detail.map(d => (d as { msg?: string }).msg ?? String(d)).join("; ")
  return `the API answered ${status}`
}

export function useTvisionApi() {
  const apiBase = String(useRuntimeConfig().public.apiBase).replace(/\/+$/, "")

  async function call<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response
    try {
      response = await fetch(`${apiBase}${path}`, init)
    }
    catch (e) {
      if ((e as Error).name === "AbortError") throw e
      throw new ApiError(`Cannot reach the TVision API at ${apiBase}. Start it with \`python -m tvision serve\`.`)
    }
    const body = await response.json().catch(() => null)
    if (!response.ok) throw new ApiError(detailMessage(body, response.status))
    return body as T
  }

  async function service(): Promise<ServiceInfo> {
    await call<{ status: string }>("/health")
    return call<ServiceInfo>("/model")
  }

  function detect(file: File, conf: number, signal?: AbortSignal): Promise<DetectionResult> {
    const form = new FormData()
    form.append("file", file, file.name)
    form.append("conf", String(conf))
    return call<DetectionResult>("/detect", { method: "POST", body: form, signal })
  }

  return { apiBase, service, detect }
}