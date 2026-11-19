<script>
  import { SvelteFlow, ViewportPortal } from '@xyflow/svelte';
  import { diagramLayout } from './layout.js';
  import Stage from './Stage.svelte';
  import Connection from './Connection.svelte';

  const nodeTypes = { stage: Stage };
  const edgeTypes = { connection: Connection };
  const model = $state({ force: 'analytic' });
  let width = $state(0);
  let height = $state(0);

  let compact = $derived(width > 0 && width < 700);
  let layout = $derived(diagramLayout(compact, model));
  let zoom = $derived(Math.min(width / layout.width, height / layout.height));
  let viewport = $derived({ x: (width - zoom * layout.width) / 2, y: (height - zoom * layout.height) / 2, zoom });
</script>

<fieldset class="force-model">
  <legend>Nonlinear force</legend>
  <label>
    <input type="radio" name="force-model" value="analytic" bind:group={model.force} />
    Analytic force
  </label>
  <label>
    <input type="radio" name="force-model" value="neural" bind:group={model.force} />
    Neural force
  </label>
</fieldset>

<div class="canvas" class:compact style:aspect-ratio="{layout.width} / {layout.height}" bind:clientWidth={width} bind:clientHeight={height}>
  {#if width && height}
    <SvelteFlow nodes={layout.nodes} edges={layout.edges} {nodeTypes} {edgeTypes}
      {viewport} minZoom={0.1} maxZoom={3} colorMode="dark"
      nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}
      nodesFocusable={false} edgesFocusable={false} deleteKey={null}
      panOnDrag={false} zoomOnScroll={false} zoomOnPinch={false} zoomOnDoubleClick={false}
      preventScrolling={false} aria-label="Continuous string model, discrete sample loop and parameter update">
      <ViewportPortal target="back">
        {#each layout.regions as [kind, name, x, y, w, h] (kind)}
          <div class="region {kind}" style:left="{x}px" style:top="{y}px" style:width="{w}px" style:height="{h}px">
            <h2>{name}</h2>
          </div>
        {/each}
      </ViewportPortal>
    </SvelteFlow>
  {/if}
</div>
