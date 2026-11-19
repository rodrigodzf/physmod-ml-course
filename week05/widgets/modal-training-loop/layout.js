import { MarkerType } from '@xyflow/svelte';

const colours = { physics: '#87d7bf', time: '#94c5ff', learning: '#ffb683' };

const wide = {
  width: 1436, height: 704,
  regions: [
    ['physics', 'Continuous time', 6, 8, 926, 334],
    ['time', 'Discrete time', 952, 8, 478, 334],
    ['learning', 'Parameter fitting', 6, 407, 926, 279]
  ],
  stages: {
    pde: [22, 52, 300, 268], modal: [597, 52, 312, 268],
    advance: [984, 52, 320, 268],
    compare: [600, 452, 309, 212], update: [22, 452, 288, 212]
  },
  links: [
    ['projection', 'physics', 'pde.right', 'modal.left', 'Sine projection', 460, 204],
    ['discretise', 'physics', 'modal.right', 'advance.left', 'Discretise', 946, 168],
    ['repeat', 'time', 'advance.right.next', 'advance.right.input', null, 1369, 186, { railX: 1369 }],
    ['readout', 'time', 'advance.bottom', 'compare.right', 'Readout', 1024, 574],
    ['gradient', 'learning', 'compare.left', 'update.right', 'Gradient', 455, 540],
    ['parameters', 'learning', 'update.top', 'advance.bottom.parameters', 'Updated tension and damping', 659, 374]
  ]
};

const narrow = {
  width: 390, height: 1713,
  regions: [
    ['physics', 'Continuous time', 7, 8, 376, 662],
    ['time', 'Discrete time', 7, 692, 376, 333],
    ['learning', 'Parameter fitting', 7, 1177, 376, 508]
  ],
  stages: {
    pde: [25, 52, 298, 188], modal: [25, 452, 298, 198],
    advance: [25, 737, 273, 268],
    compare: [25, 1227, 273, 171], update: [25, 1457, 273, 207]
  },
  links: [
    ['projection', 'physics', 'pde.bottom', 'modal.top', 'Sine projection', 174, 346],
    ['discretise', 'physics', 'modal.bottom', 'advance.top'],
    ['repeat', 'time', 'advance.right.next', 'advance.right.input', null, 333, 871, { railX: 333, vertical: true }],
    ['readout', 'time', 'advance.bottom', 'compare.top', 'Readout', 243, 1092],
    ['gradient', 'learning', 'compare.bottom', 'update.top', 'Gradient', 161, 1427],
    ['parameters', 'learning', 'update.right', 'advance.right.parameters', 'Updated tension and damping', 372, 1394, { railX: 372, vertical: true }]
  ]
};

export function diagramLayout(compact, model) {
  const layout = compact ? narrow : wide;
  const ports = Object.fromEntries(Object.keys(layout.stages).map(id => [id, []]));
  const edges = layout.links.map(([id, kind, from, to, label, labelX, labelY, route]) => {
    const [source, sourceSide] = from.split('.');
    const [target, targetSide] = to.split('.');
    ports[source].push({ id: from, side: sourceSide, type: 'source' });
    ports[target].push({ id: to, side: targetSide, type: 'target' });
    return {
      id, source, target, sourceHandle: from, targetHandle: to, type: 'connection',
      data: { kind, label, labelX, labelY, model, ...route },
      markerEnd: { type: MarkerType.ArrowClosed, color: colours[kind], width: 11, height: 11 },
      style: 'stroke:' + colours[kind] + ';stroke-width:2.6px'
    };
  });
  const nodes = Object.entries(layout.stages).map(([id, [x, y, width, height]]) => ({
    id, type: 'stage', position: { x, y }, width, height, data: { ports: ports[id], model }
  }));
  return { ...layout, nodes, edges };
}
