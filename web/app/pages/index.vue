<script setup lang="ts">
import {
  checkDimensions, checkFile, shown, sizeMismatch, type DetectionResult, type ServiceInfo,
} from "~/utils/detections"

const api = useTvisionApi()

const service = ref<ServiceInfo | null>(null)
const serviceError = ref<string | null>(null)
const threshold = ref(0)

const file = ref<File | null>(null)
const imageUrl = ref<string | null>(null)
const size = ref<{ width: number, height: number } | null>(null)
const decodeError = ref<string | null>(null)

const result = ref<DetectionResult | null>(null)
const running = ref(false)
const runError = ref<string | null>(null)
let controller: AbortController | null = null

const fileError = computed(() => {
  const limits = service.value?.limits
  if (!file.value || !limits) return null
  return checkFile(file.value, limits) ?? decodeError.value
    ?? (size.value ? checkDimensions(size.value.width, size.value.height, limits) : null)
})
const canRun = computed(() => !!service.value && !!file.value && !!size.value && !fileError.value)
const detections = computed(() => result.value ? shown(result.value, threshold.value) : [])

async function connect() {
  serviceError.value = null
  try {
    service.value = await api.service()
    threshold.value = service.value.model.default_threshold
  }
  catch (e) {
    serviceError.value = (e as Error).message
  }
}

function clear() {
  controller?.abort()
  controller = null
  running.value = false
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
  file.value = imageUrl.value = size.value = decodeError.value = result.value = runError.value = null
  if (service.value) threshold.value = service.value.model.default_threshold
}

async function select(f: File) {
  clear()
  file.value = f
  const url = URL.createObjectURL(f)
  const img = new Image()
  img.src = url
  const decoded = await img.decode().then(() => true, () => false)
  if (file.value !== f || !decoded) {
    URL.revokeObjectURL(url)
    if (file.value === f) decodeError.value = "The browser could not decode this file as an image."
    return
  }
  imageUrl.value = url
  size.value = { width: img.naturalWidth, height: img.naturalHeight }
}

async function run() {
  if (!canRun.value || !file.value || !size.value || !service.value) return
  controller?.abort()
  const current = controller = new AbortController()
  running.value = true
  runError.value = null
  result.value = null
  try {
    const r = await api.detect(file.value, service.value.limits.min_threshold, current.signal)
    const mismatch = sizeMismatch(r, size.value.width, size.value.height)
    if (mismatch) runError.value = mismatch
    else result.value = r
  }
  catch (e) {
    if ((e as Error).name !== "AbortError") runError.value = (e as Error).message
  }
  finally {
    if (controller === current) running.value = false
  }
}

onMounted(connect)
onBeforeUnmount(clear)
</script>

<template>
  <main class="page-shell">
    <HeroBanner />
    <div class="mt-8 grid gap-6 lg:grid-cols-2">
      <div class="space-y-6">
        <UploadPanel :limits="service?.limits ?? null" :file-name="file?.name ?? null" :error="fileError" @select="select" />
        <PreviewPanel
          :image-url="imageUrl"
          :size="result ? size : null"
          :detections="detections"
          :can-run="canRun"
          :running="running"
          :error="runError"
          @run="run"
          @reset="clear"
        />
      </div>
      <div class="space-y-6">
        <ResultsPanel
          v-model:threshold="threshold"
          :service="service"
          :service-error="serviceError"
          :api-base="api.apiBase"
          :result="result"
          :detections="detections"
          @retry="connect"
        />
        <InfoPanel />
        <DisclaimerCard :api-base="api.apiBase" />
      </div>
    </div>
  </main>
</template>