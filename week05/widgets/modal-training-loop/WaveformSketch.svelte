<script>
  import { pickupModeSignals, pickupSignal, plotTrace, stringModeShapes, stringShape } from './example-waveforms.js';

  let { title, kind, projection } = $props();
  const modeBaselines = [15, 40, 65];
  let modeSamples = $derived(projection ? stringModeShapes : pickupModeSignals);
  let totalSamples = $derived(projection ? stringShape : pickupSignal);
</script>

<div class="waveform-sketch {kind}">
  <strong>{title}</strong>
  <svg viewBox="0 0 178 80" role="img" aria-label={projection
    ? 'One fixed-end string shape decomposed into three spatial sine modes'
    : 'Three weighted modal signals over time summed at the pickup'}>
    <line class="baseline" x1="3" y1="40" x2="69" y2="40" />
    <polyline class="trace combined" points={plotTrace(totalSamples, { x: 3, y: 40, width: 66, scale: 16 })} />
    {#each modeSamples as samples, index}
      <line class="baseline" x1="109" y1={modeBaselines[index]} x2="175" y2={modeBaselines[index]} />
      <polyline class="trace" points={plotTrace(samples, { x: 109, y: modeBaselines[index], width: 66, scale: 16 })} />
    {/each}
    <path class="branch" d="M74 40H86 M86 40H104 M86 40C94 40 93 15 104 15 M86 40C94 40 93 65 104 65" />
    {#if projection}
      <path class="arrows" d="M100 12l5 3-5 3Z M100 37l5 3-5 3Z M100 62l5 3-5 3Z" />
    {:else}
      <path class="arrows" d="m78 37-5 3 5 3z" />
    {/if}
  </svg>
  <div class="labels" aria-hidden="true">
    <span>{projection ? 'String shape' : 'Pickup signal'}</span>
    <span>{projection ? 'Sine modes' : 'Mode signals'}</span>
  </div>
  <div class="sketch-formula">
    {#if projection}
      <tex>q_\mu(t)=\int_0^L w(x,t)\,\phi_\mu(x)\,\mathrm{d}x</tex>
    {:else}
      <tex>\hat{y}^n=\sum_\mu C_\mu q_\mu^n</tex>
    {/if}
  </div>
</div>

<style>
  .waveform-sketch {
    width: 192px;
    padding: 6px;
    border: 1px solid var(--rim);
    border-radius: 6px;
    background: var(--region);
    color: var(--accent);
    text-align: center;
  }
  .waveform-sketch.physics { width: 250px; }
  strong { display: block; font-size: 16px; line-height: 1.1; }
  svg { display: block; width: 100%; max-width: 200px; margin-inline: auto; }
  .baseline, .trace, .branch { fill: none; stroke: currentColor; }
  .baseline { opacity: .32; }
  .trace { stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
  .physics .combined { stroke: var(--ink); }
  .branch { stroke-width: 1.2; }
  .arrows { fill: currentColor; }
  .labels { display: flex; justify-content: space-between; max-width: 200px; margin-inline: auto; font-size: 13px; line-height: 1.1; }
  .physics .labels span:first-child { color: var(--ink); }
  .sketch-formula { margin-top: 5px; color: var(--ink); font-size: 18px; }
  .physics .sketch-formula { font-size: 17px; }
  .sketch-formula :global(svg) { max-width: 100%; }
</style>
