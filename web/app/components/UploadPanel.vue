<script setup lang="ts">
import { CloudUpload } from "lucide-vue-next"
import { formatBytes, formatList, type InputLimits } from "~/utils/detections"

const props = defineProps<{ limits: InputLimits | null, fileName: string | null, error: string | null }>()
const emit = defineEmits<{ select: [file: File] }>()

const inputId = useId()
const dragging = ref(false)
const accept = computed(() => props.limits?.content_types.join(",") ?? "")

function pick(files: FileList | null | undefined) {
  const file = files?.[0]
  if (file) emit("select", file)
}

function onChange(e: Event) {
  const input = e.target as HTMLInputElement
  pick(input.files)
  input.value = ""
}

function onDrop(e: DragEvent) {
  dragging.value = false
  pick(e.dataTransfer?.files)
}
</script>

<template>
  <section class="panel space-y-5">
    <div class="space-y-2">
      <p class="kicker">Upload</p>
      <h2 class="text-2xl font-bold tracking-tight text-stone-950">Choose an H&amp;E field of view</h2>
      <p v-if="limits" class="text-sm leading-7 text-stone-700">
        {{ formatList(limits) }}, up to {{ formatBytes(limits.max_upload_bytes) }}, long side
        {{ limits.min_long_side }}–{{ limits.max_long_side }} px. The model was trained on NuCLS crops at about
        0.2 µm/px.
      </p>
      <p v-else class="text-sm leading-7 text-stone-500">Input limits appear once the API is reachable.</p>
    </div>
    <label
      :for="inputId"
      class="flex cursor-pointer items-center justify-center rounded-[1.1rem] border border-dashed px-6 py-10 text-center transition"
      :class="dragging ? 'border-teal-600 bg-teal-50/60' : 'border-stone-300 bg-white/80 hover:border-stone-500'"
      @dragover.prevent="dragging = true"
      @dragleave="dragging = false"
      @drop.prevent="onDrop"
    >
      <span class="space-y-3">
        <CloudUpload class="mx-auto h-9 w-9 text-stone-800" aria-hidden="true" />
        <span class="block text-sm font-bold text-stone-900">{{ fileName ?? "Drag an image onto this box" }}</span>
        <span class="block text-xs text-stone-600">{{ fileName ? "Click to choose another" : "or click to browse" }}</span>
      </span>
      <input :id="inputId" type="file" :accept="accept" class="sr-only" @change="onChange">
    </label>
    <p v-if="error" class="text-sm leading-6 text-red-700" role="alert">{{ error }}</p>
  </section>
</template>