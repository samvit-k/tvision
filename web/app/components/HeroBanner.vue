<script setup lang="ts">
import { Microscope } from "lucide-vue-next"

const pills = ["Local API", "YOLO26n detector", "H&E histopathology", "Not for clinical use"]
</script>

<template>
  <header class="hero">
    <div class="flex flex-wrap items-center gap-2">
      <span class="pill uppercase"><Microscope class="h-3.5 w-3.5" aria-hidden="true" />Research tool</span>
      <span v-for="p in pills" :key="p" class="pill">{{ p }}</span>
    </div>
    <h1 class="hero-title mt-5">Tumor Nucleus Detection</h1>
    <p class="mt-4 max-w-2xl text-base leading-8 text-stone-200">
      Upload an H&amp;E breast-tissue field of view to see where the TVision detector places tumor and non-tumor
      nuclei, each with a class and a confidence score.
    </p>
  </header>
</template>