const amplitudes = [0.64, 0.45, 0.32];
const pickupPosition = 0.4;
const sampleCount = 72;
const points = Array.from({ length: sampleCount + 1 }, (_, i) => i / sampleCount);

export const stringModeShapes = amplitudes.map((amplitude, index) =>
  points.map(x => amplitude * Math.sin((index + 1) * Math.PI * x))
);

function modalSignals(frequencyScale, damping) {
  return amplitudes.map((amplitude, index) => {
    const wavenumber = (index + 1) * Math.PI;
    const weight = Math.sin(wavenumber * pickupPosition);
    return points.map(t =>
      weight * amplitude * Math.exp(-damping * t) * Math.cos(2 * frequencyScale * wavenumber * t)
    );
  });
}

function sumTraces(traces) {
  return points.map((_, index) => traces.reduce((sum, trace) => sum + trace[index], 0));
}

export const pickupModeSignals = modalSignals(1, 0.9);
export const stringShape = sumTraces(stringModeShapes);
export const pickupSignal = sumTraces(pickupModeSignals);
export const targetSignal = sumTraces(modalSignals(1.12, 0.65));

export function plotTrace(samples, { x, y, width, scale }) {
  return samples.map((sample, index) =>
    `${(x + points[index] * width).toFixed(2)},${(y - scale * sample).toFixed(2)}`
  ).join(' ');
}
