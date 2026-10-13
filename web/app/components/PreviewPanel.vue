<script setup lang="ts">
import { RotateCcw, ScanSearch } from "lucide-vue-next"
import { CLASS_COLORS, DEFAULT_COLOR, classLabel, formatConfidence, type Detection } from "~/utils/detections"

defineProps<{
  imageUrl: string | null
  size: { width: number, height: number } | null
  detections: Detection[]
  canRun: boolean
  running: boolean
  error: string | null
}>()
defineEmits<{ run: [], reset: [] }>()
</script>

<template>
  <section class="panel space-y-4">
    <p class="kicker">Preview</p>
    <h2 class="text-2xl font-bold tracking-tight text-stone-950">Image and detections</h2>
    <div class="relative overflow-hidden rounded-[1.4rem] border border-stone-300 bg-black/5">
      <template v-if="imageUrl">
        <img :src="imageUrl" alt="Uploaded H&amp;E image" class="block h-auto w-full">
        <svg
          v-if="size"
          class="absolute inset-0 h-full w-full"
          :viewBox="`0 0 ${size.width} ${size.height}`"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <rect
            v-for="(d, i) in detections"
            :key="i"
            :x="d.box.x1"
            :y="d.box.y1"
            :width="d.box.x2 - d.box.x1"
            :height="d.box.y2 - d.box.y1"
            fill="none"
            :stroke="CLASS_COLORS[d.class_name] ?? DEFAULT_COLOR"
            stroke-width="1.5"
            vector-effect="non-scaling-stroke"
          >
            <title>{{ classLabel(d.class_name) }} {{ formatConfidence(d.confidence) }}</title>
          </rect>
        </svg>
      </template>
      <p v-else class="p-6 text-sm leading-7 text-stone-700">
        The chosen image appears here, with a box around each detected nucleus once detection has run.
      </p>
    </div>
    <p v-if="error" class="text-sm leading-6 text-red-700" role="alert">{{ error }}</p>
    <div class="flex flex-wrap gap-3">
      <button type="button" class="btn-primary" :disabled="!canRun || running" @click="$emit('run')">
        <ScanSearch class="mr-2 h-4 w-4" aria-hidden="true" />{{ running ? "Detecting…" : "Detect nuclei" }}
      </button>
      <button type="button" class="btn-secondary" @click="$emit('reset')">
        <RotateCcw class="mr-2 h-4 w-4" aria-hidden="true" />Clear
      </button>
    </div>
  </section>
</template>