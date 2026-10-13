import type { components } from "../types/api"

export type Detection = components["schemas"]["Detection"]
export type DetectionResult = components["schemas"]["DetectionResult"]
export type InputLimits = components["schemas"]["InputLimits"]
export type ServiceInfo = components["schemas"]["ServiceInfo"]

// src/tvision/draw.py CLASS_COLORS, BGR → RGB, so the overlay matches the OpenCV output
export const CLASS_COLORS: Record<string, string> = {
  tumor: "rgb(230 40 40)",
  non_tumor: "rgb(30 160 220)",
  ambiguous: "rgb(230 200 0)",
}
export const DEFAULT_COLOR = "rgb(200 200 200)"

export const FORMAT_LABELS: Record<string, string> = { "image/png": "PNG", "image/jpeg": "JPEG" }

export function classLabel(name: string): string {
  return name.replace(/_/g, "-")
}

export function shown(result: DetectionResult, t: number): Detection[] {
  return result.detections.filter(d => d.confidence > t)
}

export interface ClassSummary { name: string, count: number, maxConfidence: number | null }

export function summarize(detections: Detection[], classes: string[]): ClassSummary[] {
  return classes.map((name) => {
    const own = detections.filter(d => d.class_name === name)
    return { name, count: own.length, maxConfidence: own.length ? Math.max(...own.map(d => d.confidence)) : null }
  })
}

export function isCapped(result: DetectionResult): boolean {
  return result.detections.length >= result.model.max_det
}

/** Lowest confidence in a capped response, or null if the response holds every detection above its threshold. */
export function capFloor(result: DetectionResult): number | null {
  return isCapped(result) ? Math.min(...result.detections.map(d => d.confidence)) : null
}

/**
 * Whether filtering `result` at `t` shows every detection the model makes above `t`. Thresholds nest and
 * /detect keeps the max_det most confident boxes, so a filtered response equals a fresh request at any t at or above
 * the request threshold; below a capped response's lowest confidence both are cut at max_det ("capped").
 */
export function coverage(result: DetectionResult, t: number): "complete" | "capped" {
  if (t < result.threshold) throw new RangeError(`threshold ${t} is below the request threshold ${result.threshold}`)
  const floor = capFloor(result)
  return floor !== null && t < floor ? "capped" : "complete"
}

export function checkFile(file: { type: string, size: number }, limits: InputLimits): string | null {
  if (!limits.content_types.includes(file.type)) {
    return `This file is ${file.type || "of an unknown type"}; use ${formatList(limits)}.`
  }
  if (file.size === 0) return "This file is empty."
  if (file.size > limits.max_upload_bytes) {
    return `This file is ${formatBytes(file.size)}; the limit is ${formatBytes(limits.max_upload_bytes)}.`
  }
  return null
}

export function checkDimensions(width: number, height: number, limits: InputLimits): string | null {
  const long = Math.max(width, height)
  if (long < limits.min_long_side || long > limits.max_long_side) {
    return `This image is ${width}×${height} px; its long side must be ${limits.min_long_side}–${limits.max_long_side} px.`
  }
  return null
}

export function sizeMismatch(result: DetectionResult, width: number, height: number): string | null {
  const { width: w, height: h } = result.image
  return w === width && h === height ? null
    : `The API analysed a ${w}×${h} px image but the browser shows ${width}×${height} px, so boxes cannot be placed.`
}

export function formatList(limits: InputLimits): string {
  return limits.content_types.map(t => FORMAT_LABELS[t] ?? t).join(" or ")
}

export function formatBytes(n: number): string {
  return n >= 1024 ** 2 ? `${+(n / 1024 ** 2).toFixed(2)} MiB` : `${+(n / 1024).toFixed(1)} KiB`
}

export function formatConfidence(c: number): string {
  return c.toFixed(2)
}