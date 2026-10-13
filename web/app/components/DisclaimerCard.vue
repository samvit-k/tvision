<script setup lang="ts">
defineProps<{ apiBase: string }>()
</script>

<template>
  <aside class="rounded-[1.4rem] border border-stone-200 bg-stone-100/80 p-5">
    <p class="kicker">Research use only</p>
    <p class="mt-2 text-sm leading-7 text-stone-700">
      TVision is a research tool, not a clinical diagnostic system; its output must not guide patient care. Your image
      is sent to the TVision API at {{ apiBase }} for analysis, and the API keeps no copy.
    </p>
  </aside>
</template>