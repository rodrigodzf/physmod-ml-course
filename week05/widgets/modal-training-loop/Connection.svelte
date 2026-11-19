<script>
  import { BaseEdge, EdgeLabel, getSmoothStepPath } from '@xyflow/svelte';
  import WaveformSketch from './WaveformSketch.svelte';

  let { id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd, style } = $props();

  let label = $derived(id === 'parameters' && data.model.force === 'neural'
    ? 'Updated tension, damping and weights'
    : data.label);

  let path = $derived.by(() => {
    if (data.railX !== undefined) {
      const rail = data.railX;
      const bend = 15;
      return `M ${sourceX} ${sourceY} H ${rail - bend}
        Q ${rail} ${sourceY} ${rail} ${sourceY - bend} V ${targetY + bend}
        Q ${rail} ${targetY} ${rail - bend} ${targetY} H ${targetX}`;
    }
    return getSmoothStepPath({
      sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition,
      borderRadius: 12, offset: 10
    })[0];
  });
</script>

<BaseEdge {id} {path} {markerEnd} {style} interactionWidth={0} />
{#if label || id === 'repeat'}
  <EdgeLabel x={data.labelX} y={data.labelY} transparent>
    {#if id === 'projection' || id === 'readout'}
      <WaveformSketch title={label} kind={data.kind} projection={id === 'projection'} />
    {:else}
      <span class="edge-label {data.kind}" class:vertical={data.vertical}>
        {#if id === 'repeat'}<tex>n\gets n+1</tex>{:else}{label}{/if}
      </span>
    {/if}
  </EdgeLabel>
{/if}
