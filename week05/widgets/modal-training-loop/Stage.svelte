<script>
  import { Handle } from '@xyflow/svelte';
  import { pickupSignal, plotTrace, targetSignal } from './example-waveforms.js';

  let { id, data } = $props();
  let neural = $derived(data.model.force === 'neural');
  const waveformPlot = { x: 1, y: 12, width: 238, scale: 12 };
</script>

{#each data.ports as port (port.id)}
  <Handle id={port.id} type={port.type} position={port.side} isConnectable={false} />
{/each}

<article class="stage {id}">
  {#if id === 'pde'}
    <h2>Forced string PDE</h2>
    <div class="formula"><tex>
      \begin{aligned}
        \rho w_{tt} &= T_0 w_{xx} - 2\rho\gamma w_t \\
                    &\quad + f_{\rm ext}(x,t) - f_{\rm nl}(w)
      \end{aligned}
    </tex></div>
    {#if !neural}
      <div class="formula"><tex>f_{\rm nl}(w)=-T_{\rm nl}(w)\,w_{xx}</tex></div>
      <div class="formula"><tex>T_{\rm nl}(w)=\frac{EA}{2L}\int_0^L w_x^2\,\mathrm{d}x</tex></div>
    {/if}
  {:else if id === 'modal'}
    <h2>Modal dynamics</h2>
    {#if neural}
      <div class="formula"><tex>
        \begin{aligned}
          \rho\ddot q_\mu &= -2\rho\gamma\dot q_\mu - T_0 k_\mu^2 q_\mu \\
                          &\quad + \bar f_{\rm ext,\mu}(t) - [\mathcal N_\theta(\mathbf q)]_\mu
        \end{aligned}
      </tex></div>
      <div class="neural-network">
        <p>MLP with weights <tex>\theta</tex></p>
        <div class="formula"><tex>\mathbf q\;\xrightarrow{\;\mathcal N_\theta\;}\;\bar{\mathbf f}_{{\rm nl},\theta}</tex></div>
      </div>
    {:else}
      <div class="formula"><tex>
        \begin{aligned}
          \rho\ddot q_\mu &= -2\rho\gamma\dot q_\mu - T_0 k_\mu^2 q_\mu \\
                          &\quad + \bar f_{\rm ext,\mu}(t) - \bar f_{\rm nl,\mu}(\mathbf q)
        \end{aligned}
      </tex></div>
      <div class="formula secondary"><tex>\bar f_{\rm nl,\mu}(\mathbf q)=\frac{EA}{2L} k_\mu^2 q_\mu \sum_\nu k_\nu^2 q_\nu^2</tex></div>
    {/if}
  {:else if id === 'advance'}
    <h2>Advance one sample</h2>
    {#if neural}
      <div class="formula"><tex>
        \begin{aligned}
          &\rho(1+\gamma h)\,q_\mu^{n+1} \\
          &\quad =(2\rho-h^2 T_0 k_\mu^2)\,q_\mu^n \\
          &\qquad - \rho(1-\gamma h)\,q_\mu^{n-1} \\
          &\qquad + h^2 \bar f_{\rm ext,\mu}(t_n) \\
          &\qquad - h^2 [\mathcal N_\theta(\mathbf q^n)]_\mu
        \end{aligned}
      </tex></div>
    {:else}
      <div class="formula"><tex>
        \begin{aligned}
          &\rho(1+\gamma h)\,q_\mu^{n+1} \\
          &\quad =(2\rho-h^2 T_0 k_\mu^2)\,q_\mu^n \\
          &\qquad - \rho(1-\gamma h)\,q_\mu^{n-1} \\
          &\qquad + h^2 \bar f_{\rm ext,\mu}(t_n) \\
          &\qquad - h^2 \bar f_{\rm nl,\mu}(\mathbf q^n)
        \end{aligned}
      </tex></div>
    {/if}
    <p class="detail">Centred differences, step <tex>h</tex>.</p>
  {:else if id === 'compare'}
    <h2>Compare waveforms</h2>
    <svg class="waveforms" viewBox="0 0 240 24" role="img" aria-label="Illustrative simulated and target pickup waveforms">
      <polyline points={plotTrace(pickupSignal, waveformPlot)} class="line"/>
      <polyline points={plotTrace(targetSignal, waveformPlot)} class="target"/>
    </svg>
    <p class="trace-key"><span>Simulated</span><span>Target</span></p>
    <div class="formula secondary">
      {#if neural}
        <tex>J(T_0,\gamma,\class{network-parameter}{\theta})=\frac{1}{N}\sum_{n=1}^N(\hat y^n-y^n)^2</tex>
      {:else}
        <tex>J(T_0,\gamma)=\frac{1}{N}\sum_{n=1}^N(\hat y^n-y^n)^2</tex>
      {/if}
    </div>
  {:else if neural}
    <h2>Fit physics and network</h2>
    <div class="formula"><tex>
      \begin{aligned}
        T_0 &\gets T_0 - \eta_T\,\partial_{T_0}J \\
        \gamma &\gets \gamma - \eta_\gamma\,\partial_\gamma J \\
        \class{network-parameter}{\theta}
          &\class{network-parameter}{\gets \theta - \eta_\theta\,\nabla_\theta J}
      \end{aligned}
    </tex></div>
    <p class="detail">Update tension, damping and network weights after each rollout.</p>
  {:else}
    <h2>Fit tension and damping</h2>
    <div class="formula"><tex>
      \begin{aligned}
        T_0 &\gets T_0 - \eta_T\,\partial_{T_0}J \\
        \gamma &\gets \gamma - \eta_\gamma\,\partial_\gamma J
      \end{aligned}
    </tex></div>
    <p class="detail">Update the resting tension <tex>T_0</tex> and damping rate <tex>\gamma</tex> after the full waveform.</p>
  {/if}
</article>
