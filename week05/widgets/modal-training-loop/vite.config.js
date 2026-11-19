import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';
import { typesetMath } from './typeset-math.js';

export default defineConfig({
  plugins: [svelte({ preprocess: typesetMath }), viteSingleFile()],
  build: {
    outDir: '../../slides/widgets',
    emptyOutDir: false,
    rolldownOptions: { input: 'modal-training-loop.html' }
  }
});
