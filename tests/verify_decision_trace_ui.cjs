// Run with NODE_PATH pointing to Playwright; uses Chrome (or CHROME_PATH).
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const {pathToFileURL} = require('node:url');
const {mkdtempSync, rmSync} = require('node:fs');
const {tmpdir} = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');

(async () => {
  const fixture = mkdtempSync(path.join(tmpdir(), 'carla-trace-ui-'));
  const repo = path.resolve(__dirname, '..');
  execFileSync(path.join(repo, 'venv/bin/python'), ['-c', `
import sys
from pathlib import Path
import numpy as np
import torch
from utils.decision_trace import write_final_decision_trace
class Model(torch.nn.Module):
    def score(self, x):
        values = torch.tensor([-1., 0., 1., 10., 1000., 1., 8., 10000.])
        return {'fused': values[(x[:, 0, 4:].long() - 4) % 8], 'levels': {}, 'signals': {}}
series = np.arange(40, dtype=np.float32)[:, None]
labels = np.zeros(40, dtype=int)
labels[8:11] = 1
labels[30:35] = 1
write_final_decision_trace(Model(), 'cpu', series,
    {'jepa_dir': sys.argv[1], 'wsz': 8, 'stride': 1,
     'decision_trace': {'max_windows': 5}}, 8.,
    options={'wsz': 8, 'stride': 1, 'check_mask': [4, 8]}, labels=labels)
class NonnegativeModel(Model):
    def score(self, x):
        output = super().score(x)
        output['fused'] = output['fused'].clamp_min(0)
        return output
positive = Path(sys.argv[1]) / 'nonnegative'
positive.mkdir()
write_final_decision_trace(NonnegativeModel(), 'cpu', series,
    {'jepa_dir': str(positive), 'wsz': 8, 'stride': 1,
     'decision_trace': {'max_windows': 5}}, 0.,
    options={'wsz': 8, 'stride': 1, 'check_mask': [4, 8]}, labels=labels)
`, fixture], {cwd: repo, env: {...process.env, MPLCONFIGDIR: '/tmp/carla-mpl'}});
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome", args: ['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport: {width: 1200, height: 1100}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(path.join(fixture, 'decision_trace.html')).href);
    await page.waitForSelector('#decision-series circle');
    assert.equal(await page.locator('#decision-truth .ground-truth-band').count(), 2,
      'full-test timeline must show anomaly labels beyond the replay prefix');
    assert.equal(await page.locator('#decision-truth').getAttribute('data-domain-end'), '40');
    assert.match(await page.locator('#decision-truth-summary').textContent(), /2 anomaly intervals/);
    assert.match(await page.locator('#decision-truth-summary').textContent(), /first 5 of 33 windows/);
    assert.equal(await page.locator('#decision-focus-slider').count(), 1, 'missing timestep slider');
    assert.equal(await page.locator('#decision-series .checking-boundary').count(), 2);
    assert.ok(await page.locator('#decision-series .ground-truth-band').count() > 0);
    const verticalGap = async () => page.locator('#decision-series').evaluate(svg => {
      const y = t => Number(svg.querySelector(`circle[data-timestep="${t}"]`).getAttribute('cy'));
      return Math.abs(y(5) - y(6));
    });
    assert.equal(await page.locator('#decision-series').getAttribute('data-y-scale'), 'symlog');
    const logGap = await verticalGap();
    const rawTooltip = await page.locator('#decision-series circle[data-timestep="8"] title').textContent();
    await page.locator('#decision-y-scale').selectOption('linear');
    assert.ok(logGap > 10 * await verticalGap(), 'log scale must reveal small scores despite large outliers');
    assert.equal(await page.locator('#decision-evolution').getAttribute('data-y-scale'), 'linear');
    assert.equal(await page.locator('#decision-series circle[data-timestep="8"] title').textContent(), rawTooltip);
    await page.locator('#decision-y-scale').selectOption('symlog');
    for (const id of ['decision-series', 'decision-evolution']) {
      assert.equal(await page.locator(`#${id}`).getAttribute('data-y-scale'), 'symlog');
      assert.ok(await page.locator(`#${id} circle`).evaluateAll(nodes =>
        nodes.every(node => Number.isFinite(Number(node.getAttribute('cy'))))));
    }
    await page.locator('#decision-focus').fill('10');
    await page.locator('#decision-focus').dispatchEvent('input');
    assert.equal(await page.locator('#decision-focus-slider').inputValue(), '10');
    assert.equal(await page.locator('#decision-evolution').getAttribute('data-domain-start'), '4');
    assert.equal(await page.locator('#decision-evolution').getAttribute('data-domain-end'), '5');
    assert.equal(await page.locator('#decision-evolution circle').count(), 2);
    assert.equal(await page.locator('#decision-series').getAttribute('data-check-start'), '8');
    assert.equal(await page.locator('#decision-series').getAttribute('data-check-end'), '12');
    await page.locator('#decision-focus-slider').focus();
    await page.keyboard.press('ArrowLeft');
    assert.equal(await page.locator('#decision-focus').inputValue(), '9');
    await page.locator('#decision-step').fill('1');
    await page.locator('#decision-step').dispatchEvent('input');
    assert.equal(await page.locator('#decision-focus-slider').getAttribute('min'), '5');
    assert.equal(await page.locator('#decision-focus-slider').getAttribute('max'), '8');
    assert.equal(await page.locator('#decision-focus').inputValue(), '8');
    assert.equal(await page.locator('#decision-evolution circle').count(), 1);
    await page.locator('#decision-series-zoom-in').click();
    const domain = async () => page.locator('#decision-series').evaluate(svg =>
      [Number(svg.dataset.domainStart), Number(svg.dataset.domainEnd)]);
    const zoomed = await domain();
    await page.locator('#decision-series-reset').click();
    const reset = await domain();
    assert.ok(zoomed[1] - zoomed[0] < reset[1] - reset[0]);
    const box = await page.locator('#decision-series').boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -150);
    await page.waitForTimeout(100);
    const wheel = await domain();
    assert.ok(wheel[1] - wheel[0] < reset[1] - reset[0]);
    await page.locator('#decision-step').fill('4');
    await page.locator('#decision-step').dispatchEvent('input');
    await page.locator('#decision-series-checking').click();
    assert.deepEqual(await domain(), [8, 12]);
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 100, box.y + box.height / 2, {steps: 4});
    await page.mouse.up();
    assert.ok((await domain())[0] < 8, 'drag must pan the zoomed series');
    await page.locator('#decision-evolution-zoom-in').click();
    const evolutionSpan = async () => page.locator('#decision-evolution').evaluate(svg =>
      Number(svg.dataset.domainEnd) - Number(svg.dataset.domainStart));
    const evolutionZoom = await evolutionSpan();
    await page.locator('#decision-evolution-reset').click();
    assert.ok(evolutionZoom < await evolutionSpan());
    await page.locator('#decision-series-checking').click();
    await page.screenshot({path: '/tmp/carla-decision-trace-ui.png', fullPage: true});
    await page.goto(pathToFileURL(path.join(fixture, 'nonnegative/decision_trace.html')).href);
    await page.waitForSelector('#decision-series circle');
    for (const scale of ['symlog', 'linear']) {
      await page.locator('#decision-y-scale').selectOption(scale);
      for (const id of ['decision-series', 'decision-evolution']) {
        assert.ok(await page.locator(`#${id} text`).evaluateAll(nodes =>
          nodes.every(node => !/^-\d/.test(node.textContent))),
          'nonnegative scores must never have negative axis tick labels');
      }
    }
    assert.deepEqual(errors, []);
    console.log('Decision trace browser controls, checking boundaries, truth labels, contributing arrivals and zoom: OK');
  } finally {
    await browser.close();
    rmSync(fixture, {recursive: true, force: true});
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
