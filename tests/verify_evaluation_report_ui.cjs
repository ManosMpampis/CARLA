// Self-contained Chrome/CDP checks; no Playwright dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawn, execFileSync} = require('node:child_process');
const {pathToFileURL} = require('node:url');

(async () => {
  const repo = path.resolve(__dirname, '..');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'carla-report-ui-'));
  execFileSync(path.join(repo,'venv/bin/python'), ['-c', `
import sys
import numpy as np
from utils.evaluation_artifacts import write_score_html
scores = np.exp(np.sin(np.arange(8448)/35)*5)
scores[4000] = 10000
labels = np.zeros(8448,dtype=int)
labels[3950:4050] = 1
starts = np.arange(512,8448-256+1,128)
write_score_html(sys.argv[1],scores[512:],labels[512:],8.,'Test_timeseries',
    timestep_indices=np.arange(512,8448),series_labels=labels,
    window_starts=starts,window_ends=starts+256,
    input_starts=starts-512,input_ends=starts+256)
`, path.join(root,'report.html')], {cwd:repo});
  const chrome = spawn(process.env.CHROME_PATH || '/usr/bin/google-chrome',
    ['--headless','--no-sandbox','--disable-gpu','--remote-debugging-pipe',
     `--user-data-dir=${path.join(root,'chrome')}`],
    {stdio:['ignore','ignore','ignore','pipe','pipe']});
  const pending = new Map(), waiters = [];
  let sequence = 0, buffer = '';
  chrome.stdio[4].on('data', chunk => {
    buffer += chunk.toString();
    let end;
    while ((end=buffer.indexOf('\0')) >= 0) {
      const raw=buffer.slice(0,end);buffer=buffer.slice(end+1);
      if (!raw) continue;
      const message=JSON.parse(raw);
      if (message.id) {
        const request=pending.get(message.id);pending.delete(message.id);
        if (message.error) request.reject(Error(JSON.stringify(message.error)));
        else request.resolve(message.result);
      } else {
        for (let i=waiters.length-1;i>=0;i--) if (waiters[i].method===message.method) {
          waiters[i].resolve(message.params);waiters.splice(i,1);
        }
      }
    }
  });
  const send=(method,params={},sessionId)=>new Promise((resolve,reject)=>{
    const id=++sequence;pending.set(id,{resolve,reject});
    chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');
  });
  const timer=setTimeout(()=>{chrome.kill();throw Error('Chrome UI verification timed out');},30000);
  try {
    const {targetId}=await send('Target.createTarget',{url:'about:blank'});
    const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
    await send('Page.enable',{},sessionId);
    await send('Emulation.setDeviceMetricsOverride',{width:1200,height:1100,deviceScaleFactor:1,mobile:false},sessionId);
    await send('Page.bringToFront',{},sessionId);
    const loaded=new Promise(resolve=>waiters.push({method:'Page.loadEventFired',resolve}));
    await send('Page.navigate',{url:pathToFileURL(path.join(root,'report.html')).href},sessionId);
    await loaded;
    const result=await send('Runtime.evaluate',{awaitPromise:true,returnByValue:true,expression:`
(async () => {
 const $=id=>document.getElementById(id),svg=$('decision-series');
 const wait=()=>new Promise(resolve=>setTimeout(resolve,100));
 const domain=()=>[+svg.dataset.domainStart,+svg.dataset.domainEnd];
 const assert=(ok,message)=>{if(!ok)throw Error(message);};
 assert(svg.dataset.yScale==='symlog','default logarithmic scale');
 assert($('decision-evolution').closest('.decision-chart').hidden,'hide replay-only plot');
 const full=domain();
 assert(full[0]===0&&full[1]===8448,'original full timeline including context');
 assert(!$('decision-step').hidden,'checking-window slider is visible');
 assert(+svg.dataset.checkStart===512&&+svg.dataset.checkEnd===768,'actual first checking crop');
 assert(+svg.dataset.inputStart===0&&+svg.dataset.inputEnd===768,'actual input window');
 $('decision-step').value='27';$('decision-step').dispatchEvent(new Event('input'));await wait();
 assert(+svg.dataset.checkStart===3968&&+svg.dataset.checkEnd===4224,'slider glides across actual checking windows');
 assert(+$('decision-truth').dataset.checkStart===3968,'overview tracks checking region');
 assert($('decision-step-label').textContent.includes('width 256 timesteps'),'checking width is explicit');
 assert(svg.querySelectorAll('.checking-boundary').length===2,'checking boundaries visible');
 assert(domain()[0]===0&&domain()[1]===8448,'slider keeps full timeline');
 $('decision-series-checking').click();await wait();
 assert(domain()[0]===3968&&domain()[1]===4224,'zoom to actual checking region');
 $('decision-focus').value='4001';$('decision-focus').dispatchEvent(new Event('input'));await wait();
 assert(svg.querySelector('circle[data-timestep="4001"]'),'zoom reveals an exact timestep');
 assert($('decision-current').textContent.includes('Timestep 4001:'),'selection does not snap to a downsampled point');
 $('decision-series-reset').click();await wait();
 $('decision-y-scale').value='linear';$('decision-y-scale').dispatchEvent(new Event('change'));
 assert(svg.dataset.yScale==='linear','linear toggle');
 $('decision-y-scale').value='symlog';$('decision-y-scale').dispatchEvent(new Event('change'));
 $('decision-series-zoom-in').click();await wait();
 assert(domain()[1]-domain()[0]<full[1]-full[0],'button zoom');
 $('decision-series-reset').click();await wait();
 assert(domain()[1]-domain()[0]===full[1]-full[0],'reset');
 svg.dispatchEvent(new WheelEvent('wheel',{deltaY:-100,clientX:600,bubbles:true,cancelable:true}));await wait();
 assert(domain()[1]-domain()[0]<full[1]-full[0],'wheel zoom');
 $('decision-focus').value='4001';$('decision-focus').dispatchEvent(new Event('input'));await wait();
 assert($('decision-current').textContent.includes('score'),'selected score');
 assert(svg.querySelectorAll('circle').length<1200,'bounded SVG marks');
 assert(!svg.innerHTML.includes('NaN'),'finite coordinates');
 const title=svg.querySelector('circle title').textContent;
 assert(title.includes('score')&&title.includes('ground truth'),'score tooltip');
 return {marks:svg.querySelectorAll('circle').length,scale:svg.dataset.yScale};
})()`},sessionId);
    assert.ok(!result.exceptionDetails,JSON.stringify(result.exceptionDetails));
    // Check pointer drag through actual browser input, including capture.
    const beforePan = await send('Runtime.evaluate', {expression:"+document.getElementById('decision-series').dataset.domainStart",returnByValue:true},sessionId);
    await send('Input.dispatchMouseEvent',{type:'mousePressed',x:600,y:650,button:'left',clickCount:1},sessionId);
    await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:680,y:650,button:'left',buttons:1},sessionId);
    await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:680,y:650,button:'left',clickCount:1},sessionId);
    const afterPan = await send('Runtime.evaluate', {awaitPromise:true,returnByValue:true,expression:`
new Promise(resolve=>setTimeout(()=>resolve(+document.getElementById('decision-series').dataset.domainStart),100))`},sessionId);
    assert.notEqual(afterPan.result.value,beforePan.result.value,'pointer drag must pan the chart');
    const idle = await send('Runtime.evaluate', {awaitPromise:true,returnByValue:true,expression:`
new Promise(resolve=>{
 let changes=0;const observer=new MutationObserver(records=>{changes+=records.length;});
 observer.observe(document.getElementById('decision-series'),{childList:true,subtree:true});
 setTimeout(()=>{observer.disconnect();resolve(changes);},200);
})`},sessionId);
    assert.equal(idle.result.value,0,'idle reports must not redraw continuously');
    const image=await send('Page.captureScreenshot' ,{format:'png'},sessionId);
    fs.writeFileSync('/tmp/carla-evaluation-report-ui.png',Buffer.from(image.data,'base64'));
    console.log('Chrome report log/linear scales, zoom/pan, selection, tooltips, bounded marks and idle stability: OK',result.result.value);
  } finally {
    clearTimeout(timer);
    chrome.kill();
    await new Promise(resolve=>chrome.once('exit',resolve));
    fs.rmSync(root,{recursive:true,force:true});
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
