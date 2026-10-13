import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import {
  CLASS_COLORS, capFloor, checkDimensions, checkFile, coverage, formatBytes, isCapped, shown, sizeMismatch, summarize,
  type Detection, type DetectionResult, type InputLimits,
} from "../app/utils/detections"

const CLASSES = ["tumor", "non_tumor", "ambiguous"]
const LIMITS: InputLimits = {
  min_long_side: 192, max_long_side: 1024, min_threshold: 0.01, max_upload_bytes: 8 * 1024 ** 2,
  content_types: ["image/png", "image/jpeg"],
}

function det(classId: number, confidence: number): Detection {
  return { class_id: classId, class_name: CLASSES[classId]!, confidence, box: { x1: 1, y1: 2, x2: 30, y2: 40 } }
}

function result(detections: Detection[], maxDet = 1000, threshold = 0.01): DetectionResult {
  const counts = Object.fromEntries(CLASSES.map(n => [n, detections.filter(d => d.class_name === n).length]))
  return {
    image: { width: 300, height: 200, scale: 640 / 300 },
    model: {
      name: "stub", weights_sha256: "0".repeat(64), imgsz: 640, max_det: maxDet, iou: 0.7, classes: CLASSES,
      headline_classes: ["tumor", "non_tumor"], default_threshold: 0.24,
    },
    threshold, detections, counts,
  }
}

const SAMPLE = [det(0, 0.91), det(1, 0.6), det(0, 0.24), det(2, 0.2), det(1, 0.05)]

describe("filtering", () => {
  it("keeps only detections strictly above the threshold", () => {
    expect(shown(result(SAMPLE), 0.24).map(d => d.confidence)).toEqual([0.91, 0.6])
    expect(shown(result(SAMPLE), 0.01)).toHaveLength(SAMPLE.length)
    expect(shown(result(SAMPLE), 0.91)).toEqual([])
  })

  it("summarizes every class in order, matching an independent tally", () => {
    for (const t of [0.01, 0.1, 0.24, 0.5, 0.95]) {
      const kept = shown(result(SAMPLE), t)
      const tally: Record<string, number> = { tumor: 0, non_tumor: 0, ambiguous: 0 }
      for (const d of SAMPLE) if (d.confidence > t) tally[d.class_name]! += 1
      const rows = summarize(kept, CLASSES)
      expect(rows.map(r => r.name)).toEqual(CLASSES)
      expect(Object.fromEntries(rows.map(r => [r.name, r.count]))).toEqual(tally)
    }
    expect(summarize(SAMPLE, CLASSES).map(r => r.maxConfidence)).toEqual([0.91, 0.6, 0.2])
    expect(summarize([], CLASSES).every(r => r.maxConfidence === null)).toBe(true)
  })
})

describe("max_det coverage", () => {
  it("is complete at every threshold when the response is below max_det", () => {
    const r = result(SAMPLE, SAMPLE.length + 1)
    expect(isCapped(r)).toBe(false)
    expect(capFloor(r)).toBeNull()
    for (const t of [0.01, 0.04, 0.5]) expect(coverage(r, t)).toBe("complete")
  })

  it("is capped below the lowest returned confidence and complete from it upwards", () => {
    const r = result(SAMPLE, SAMPLE.length)
    expect(isCapped(r)).toBe(true)
    expect(capFloor(r)).toBe(0.05)
    expect(coverage(r, 0.01)).toBe("capped")
    expect(coverage(r, 0.049)).toBe("capped")
    expect(coverage(r, 0.05)).toBe("complete")
    expect(coverage(r, 0.3)).toBe("complete")
  })

  it("refuses thresholds below the request threshold", () => {
    expect(() => coverage(result(SAMPLE, 1000, 0.1), 0.05)).toThrow(RangeError)
  })
})

describe("client pre-checks", () => {
  it("accepts only the API's content types", () => {
    expect(checkFile({ type: "image/png", size: 10 }, LIMITS)).toBeNull()
    expect(checkFile({ type: "image/jpeg", size: 10 }, LIMITS)).toBeNull()
    expect(checkFile({ type: "image/webp", size: 10 }, LIMITS)).toMatch(/PNG or JPEG/)
    expect(checkFile({ type: "", size: 10 }, LIMITS)).toMatch(/unknown type/)
  })

  it("enforces the byte limit inclusively and refuses empty files", () => {
    expect(checkFile({ type: "image/png", size: LIMITS.max_upload_bytes }, LIMITS)).toBeNull()
    expect(checkFile({ type: "image/png", size: LIMITS.max_upload_bytes + 1 }, LIMITS)).toMatch(/limit is 8 MiB/)
    expect(checkFile({ type: "image/png", size: 0 }, LIMITS)).toMatch(/empty/)
  })

  it("bounds the long side inclusively", () => {
    expect(checkDimensions(191, 100, LIMITS)).not.toBeNull()
    expect(checkDimensions(192, 100, LIMITS)).toBeNull()
    expect(checkDimensions(100, 1024, LIMITS)).toBeNull()
    expect(checkDimensions(1025, 1024, LIMITS)).not.toBeNull()
  })

  it("refuses to place boxes when the analysed and displayed sizes differ", () => {
    expect(sizeMismatch(result([]), 300, 200)).toBeNull()
    expect(sizeMismatch(result([]), 200, 300)).toMatch(/300×200/)
  })

  it("formats byte sizes", () => {
    expect(formatBytes(8 * 1024 ** 2)).toBe("8 MiB")
    expect(formatBytes(1536)).toBe("1.5 KiB")
  })
})

describe("overlay colours", () => {
  it("equal the OpenCV CLASS_COLORS in src/tvision/draw.py, converted from BGR", () => {
    const source = readFileSync(new URL("../../src/tvision/draw.py", import.meta.url), "utf8")
    const block = source.match(/CLASS_COLORS[^{]*\{([^}]*)\}/)![1]!
    const bgr = [...block.matchAll(/"(\w+)":\s*\((\d+),\s*(\d+),\s*(\d+)\)/g)]
    expect(bgr.map(m => m[1])).toEqual(Object.keys(CLASS_COLORS))
    for (const [, name, b, g, r] of bgr) expect(CLASS_COLORS[name!]).toBe(`rgb(${r} ${g} ${b})`)
  })
})