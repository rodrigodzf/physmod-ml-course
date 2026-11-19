import { liteAdaptor } from '@mathjax/src/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from '@mathjax/src/js/handlers/html.js';
import { TeX } from '@mathjax/src/js/input/tex.js';
import '@mathjax/src/js/input/tex/base/BaseConfiguration.js';
import '@mathjax/src/js/input/tex/ams/AmsConfiguration.js';
import '@mathjax/src/js/input/tex/html/HtmlConfiguration.js';
import { mathjax } from '@mathjax/src/js/mathjax.js';
import { SVG } from '@mathjax/src/js/output/svg.js';
import '@mathjax/src/js/util/asyncLoad/esm.js';

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const document = mathjax.document('', {
  InputJax: new TeX({ packages: ['base', 'ams', 'html'], formatError(_parser, error) { throw error; } }),
  OutputJax: new SVG({ fontCache: 'local', linebreaks: { inline: false } })
});

export const typesetMath = {
  name: 'typeset-math',
  async markup({ content }) {
    let code = '';
    let cursor = 0;
    for (const match of content.matchAll(/<tex>([\s\S]*?)<\/tex>/g)) {
      const tex = match[1].trim();
      const equation = await document.convertPromise(tex, { display: false });
      const svg = adaptor.tags(equation, 'svg')[0];
      adaptor.setAttribute(svg, 'aria-label', tex);
      code += content.slice(cursor, match.index) + '{@html ' + JSON.stringify(adaptor.outerHTML(equation)) + '}';
      cursor = match.index + match[0].length;
    }
    return { code: code + content.slice(cursor) };
  }
};
