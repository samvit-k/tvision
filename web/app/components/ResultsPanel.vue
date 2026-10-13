<script setup lang="ts">
import {
  CLASS_COLORS, DEFAULT_COLOR, capFloor, classLabel, coverage, formatConfidence, summarize,
  type Detection, type DetectionResult, type ServiceInfo,
} from "~/utils/detections"

const props = defineProps<{
  service: ServiceInfo | null
  serviceError: string | null
  apiBase: string
  result: DetectionResult | null
  detections: Detection[]
}>()
const threshold = defineModel<number>("threshold", { required: true })
defineEmits<{ retry: [] }>()

const model = computed(() => props.service?.model ?? null)
const rows = computed(() => model.value ? summarize(props.detections, model.value.classes) : [])
const headline = computed(() => rows.value.filter(r => model.value?.headline_classes.includes(r.name)))
const others = computed(() => rows.value.filter(r => !model.value?.headline_classes.includes(r.name)))
const capped = computed(() => props.result !== null && coverage(props.result, threshold.value) === "capped")
const floor = computed(() => props.result ? capFloor(props.result) : null)
</script>

<template>
  <section class="panel space-y-5">
    <div class="space-y-2">
      <p class="kicker">Results</p>
      <h2 class="text-2xl font-bold tracking-tight text-stone-950">Detected nuclei</h2>
      <p class="text-sm leading-7 text-stone-700">
        Each box comes from the API's JSON response. Moving the threshold filters that response in the browser; the
        model does not run again.
      </p>
    </div>

    <div class="card bg-white/80">
      <p class="text-xs font-bold tracking-[0.16em] text-stone-500 uppercase">API status</p>
      <p v-if="serviceError" class="mt-2 text-sm leading-7 text-stone-700" role="alert">
        <strong>Unavailable</strong> — {{ serviceError }}
        <button type="button" class="ml-1 font-bold text-teal-700 underline" @click="$emit('retry')">Retry</button>
      </p>
      <p v-else-if="model" class="mt-2 text-sm leading-7 text-stone-700">
        <strong>Ready</strong> — model {{ model.name }} (<code class="text-xs">{{ model.weights_sha256.slice(0, 8) }}</code>),
        input {{ model.imgsz }} px, at most {{ model.max_det.toLocaleString() }} detections per image.
      </p>
      <p v-else class="mt-2 text-sm leading-7 text-stone-700"><strong>Connecting</strong> — contacting {{ apiBase }}…</p>
    </div>

    <div class="card bg-cream">
      <label for="threshold" class="text-sm font-bold text-stone-900">
        Confidence threshold: {{ service ? formatConfidence(threshold) : "—" }}
      </label>
      <input
        id="threshold"
        v-model.number="threshold"
        type="range"
        :min="service?.limits.min_threshold ?? 0.01"
        max="0.99"
        step="0.01"
        :disabled="!service"
        class="mt-3 w-full accent-emerald-600"
      >
      <p class="mt-2 text-xs leading-6 text-stone-600">
        Boxes scoring above the threshold are shown. Lowering it reveals more nuclei and more false positives; raising
        it hides weaker detections, including some true nuclei.
        <template v-if="model">
          The default, {{ formatConfidence(model.default_threshold) }}, gave the best mean tumor / non-tumor F1 on the
          validation set.
          <button
            v-if="threshold !== model.default_threshold"
            type="button"
            class="font-bold text-teal-700 underline"
            @click="threshold = model.default_threshold"
          >
            Use default
          </button>
        </template>
      </p>
    </div>

    <div class="card bg-white/80">
      <p class="text-sm font-bold text-stone-900">Nuclei shown</p>
      <p class="mt-2 text-3xl font-bold text-stone-950">{{ result ? detections.length.toLocaleString() : "—" }}</p>
      <p v-if="result" class="mt-1 text-xs leading-6 text-stone-600">
        <template v-for="(r, i) in headline" :key="r.name">{{ i ? " · " : "" }}{{ classLabel(r.name) }} {{ r.count }}</template>
        <template v-for="r in others" :key="r.name"> · {{ classLabel(r.name) }} {{ r.count }} (reported separately)</template>
      </p>
    </div>

    <div class="card bg-white/60 text-sm leading-7 text-stone-700">
      <p v-if="!result">Run detection to list the nuclei found in each class.</p>
      <template v-else>
        <p v-if="!detections.length">No nuclei score above this threshold.</p>
        <ul v-else class="space-y-1">
          <li v-for="r in [...headline, ...others]" :key="r.name" class="flex items-center gap-3">
            <span class="h-3 w-3 shrink-0 rounded-sm" :style="{ background: CLASS_COLORS[r.name] ?? DEFAULT_COLOR }" />
            <span class="font-bold text-stone-900">{{ classLabel(r.name) }}</span>
            <span class="ml-auto tabular-nums">{{ r.count }}</span>
            <span class="w-28 text-right text-xs text-stone-500 tabular-nums">
              {{ r.maxConfidence === null ? "" : `top ${formatConfidence(r.maxConfidence)}` }}
            </span>
          </li>
        </ul>
        <p v-if="capped && floor !== null" class="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs leading-6 text-amber-900">
          The response reached the API's limit of {{ result.model.max_det.toLocaleString() }} detections, keeping those
          scoring {{ floor.toFixed(4) }} or more. Any weaker detections were cut off and cannot be shown at this
          threshold.
        </p>
      </template>
    </div>
  </section>
</template>