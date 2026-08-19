(function(){
  /* ===== Workspace navigation ===== */
  const viewButtons = Array.from(document.querySelectorAll('[data-view]'));
  const viewPanels = Array.from(document.querySelectorAll('[data-panel]'));
  function activateView(view){
    viewPanels.forEach(panel => panel.classList.toggle('active', panel.dataset.panel === view));
    viewButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.view === view));
    const target = document.querySelector(`[data-panel="${view}"]`);
    if(target) window.scrollTo({top:0, behavior:'smooth'});
  }
  viewButtons.forEach(btn => btn.addEventListener('click', ()=>activateView(btn.dataset.view)));


  /* ============ Math helpers ============ */
  function randRange(a,b){ return a + Math.random()*(b-a); }
  function sigmoid(z){ return 1/(1+Math.exp(-z)); }
  function reluFn(z){ return Math.max(0,z); }
  function softmaxVec(zs){
    const m = Math.max(...zs);
    const exps = zs.map(z => Math.exp(z-m));
    const s = exps.reduce((a,b)=>a+b,0) || 1;
    return exps.map(e => e/s);
  }
  function activate(zs, type){
    if(type==='softmax') return softmaxVec(zs);
    if(type==='relu') return zs.map(reluFn);
    return zs.map(sigmoid);
  }
  function activDeriv(type, a, z){
    if(type==='relu') return z>0 ? 1 : 0;
    return a*(1-a); // sigmoid, and softmax (simplified elementwise approximation)
  }
  function shuffleIdx(n){
    const a = Array.from({length:n}, (_,i)=>i);
    for(let i=a.length-1;i>0;i--){ const j = Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
    return a;
  }
  function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
  function fmt(n, d){ return (typeof n === 'number' && isFinite(n)) ? n.toFixed(d===undefined?4:d) : '—'; }

  /* ============ Config & state ============ */
  let cfg = {
    // Default MLP Notebook Visualizer configuration: 4 → 2 → 1
    numInputs: 4, hidden: [2], numOutputs: 1,
    hiddenAct: 'sigmoid', outputAct: 'sigmoid',
    lossFn: 'mse', gdType: 'mini-batch', batchSize: 2,
    epochs: 30, lr: 0.8
  };
  let net = null;         // { layerSizes, weights, biases }
  let dataset = [];       // [{x:[...], y:[...]}]
  let history = [];       // [{epoch, loss}]
  let epochsRun = 0;
  let training = false;
  let stopRequested = false;
  let inspect = null;     // { idx, activations, zsAll, deltas, loss, baseError, pred }
  let inspectStage = 0;   // 0 ready, 1 forward, 2 loss, 3 backprop, 4 update

  const LOSS_LABEL = { diff:'Actual − Predicted', mae:'Mean Absolute Error', mse:'Mean Squared Error' };
  const LOSS_NOTE = {
    diff: 'Loss = mean(y minus predicted). A raw, signed gap — the simplest possible error signal.',
    mae: 'Loss = mean(|y minus predicted|). Gradient uses the sign of the gap, so every mistake pulls weights by the same size step regardless of magnitude.',
    mse: 'Loss = mean((y minus predicted) squared). Gradient uses the gap directly, so bigger mistakes pull weights harder — the classic choice for regression.'
  };

  /* ============ Steps bar ============ */
  const STEPS = ['Configure','Forward','Loss','Backward','Update','Train Epochs'];
  let stepDone = new Set();
  function renderSteps(active){
    const bar = document.getElementById('stepsBar');
    bar.innerHTML = STEPS.map((s,i)=>{
      const cls = i===active ? 'active' : (stepDone.has(i) ? 'done' : '');
      return `<div class="step ${cls}"><span class="num">${i+1}</span>${s}</div>`;
    }).join('');
  }
  function markStep(i){ stepDone.add(i); renderSteps(i); }

  /* ============ Network construction (values typed in by the user) ============ */
  function defaultWeight(layer, i, j){
    // Classic 4 → 2 → 1 inspection defaults. Other architectures fall back to 0.10.
    const preset = {
      '0:0:0':0.3, '0:0:1':-0.2, '0:0:2':0.2, '0:0:3':0.1,
      '0:1:0':0.1, '0:1:1':0.4, '0:1:2':-0.3, '0:1:3':0.4,
      '1:0:0':-0.3, '1:0:1':0.2
    };
    return Object.prototype.hasOwnProperty.call(preset, `${layer}:${i}:${j}`) ? preset[`${layer}:${i}:${j}`] : 0.10;
  }
  function defaultBias(layer, i){
    const preset = {'0:0':0.2, '0:1':0.1, '1:0':-0.3};
    return Object.prototype.hasOwnProperty.call(preset, `${layer}:${i}`) ? preset[`${layer}:${i}`] : 0;
  }
  function defaultX(i){ return [1,1,0,1][i] ?? 0.5; }
  function defaultY(i){ return i===0 ? 1 : 0.5; }

  // Builds empty/default-filled input fields (x, y, weights, biases) that match
  // the current architecture in cfg. Called whenever the architecture changes.
  function renderParamInputs(){
    const sizes = [cfg.numInputs, ...cfg.hidden, cfg.numOutputs];

    const xGrid = document.getElementById('xInputsGrid');
    xGrid.innerHTML = Array.from({length:cfg.numInputs}, (_,i)=>`
      <div class="cfg-field">
        <label for="xin_${i}">x${i+1}</label>
        <input type="number" step="any" id="xin_${i}" data-x-idx="${i}" value="${defaultX()}">
      </div>`).join('');

    const yGrid = document.getElementById('yInputsGrid');
    yGrid.innerHTML = Array.from({length:cfg.numOutputs}, (_,i)=>`
      <div class="cfg-field">
        <label for="yin_${i}">y${i+1} (target)</label>
        <input type="number" step="any" id="yin_${i}" data-y-idx="${i}" value="${defaultY()}">
      </div>`).join('');

    const paramWrap = document.getElementById('paramInputs');
    let html = '';
    for(let k=0;k<sizes.length-1;k++){
      const from = sizes[k], to = sizes[k+1];
      const fromLabel = k===0 ? 'x' : 'h'+k+'.';
      const toLabel = (k===sizes.length-2) ? 'y' : 'h'+(k+1)+'.';
      let head = `<th>${toLabel==='y'?'output':'hidden '+(k+1)} \\ ${fromLabel==='x'?'input':'hidden '+k}</th>`;
      for(let j=0;j<from;j++) head += `<th>${fromLabel}${j+1}</th>`;
      head += `<th class="bias-col">bias</th>`;
      let rows = '';
      for(let i=0;i<to;i++){
        let cells = `<td>${toLabel}${i+1}</td>`;
        for(let j=0;j<from;j++){
          cells += `<td><input type="number" step="any" data-w-layer="${k}" data-w-i="${i}" data-w-j="${j}" value="${defaultWeight(k,i,j)}"></td>`;
        }
        cells += `<td class="bias-col"><input type="number" step="any" data-b-layer="${k}" data-b-i="${i}" value="${defaultBias(k,i)}"></td>`;
        rows += `<tr>${cells}</tr>`;
      }
      html += `
        <div class="param-layer">
          <div class="param-layer-title">Layer ${k+1} — ${from} → ${to}</div>
          <div class="table-scroll">
            <table class="param-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>
          </div>
        </div>`;
    }
    paramWrap.innerHTML = html;
  }

  function randomSample(){
    return { x:Array.from({length:cfg.numInputs},()=>Math.random()), y:Array.from({length:cfg.numOutputs},()=>Math.random()) };
  }

  function renderDatasetEditor(){
    const wrap=document.getElementById('datasetEditor'); if(!wrap) return;
    const n=Math.max(1,Math.min(500,parseInt(document.getElementById('cfgDatasetSize').value,10)||10));
    document.getElementById('cfgDatasetSize').value=n;
    let h='<div class="table-scroll" style="max-height:360px"><table><thead><tr><th>#</th>';
    for(let i=0;i<cfg.numInputs;i++) h+=`<th>x${i+1}</th>`;
    for(let i=0;i<cfg.numOutputs;i++) h+=`<th>y${i+1}</th>`;
    h+='</tr></thead><tbody>';
    for(let r=0;r<n;r++){ h+=`<tr><td>sample ${r+1}</td>`; for(let i=0;i<cfg.numInputs;i++){ const xv=(r===0 ? defaultX(i) : 0.5); h+=`<td><input type="number" step="any" data-ds-row="${r}" data-ds-x="${i}" value="${xv}" style="width:82px"></td>`; } for(let i=0;i<cfg.numOutputs;i++){ const yv=(r===0 ? defaultY(i) : 0.5); h+=`<td><input type="number" step="any" data-ds-row="${r}" data-ds-y="${i}" value="${yv}" style="width:82px"></td>`; } h+='</tr>'; }
    wrap.innerHTML=h+'</tbody></table></div>';
  }

  function readDatasetFromEditor(){
    const n=Math.max(1,Math.min(500,parseInt(document.getElementById('cfgDatasetSize').value,10)||1)); const data=[];
    for(let r=0;r<n;r++){
      const x=Array.from({length:cfg.numInputs},(_,i)=>parseFloat(document.querySelector(`[data-ds-row="${r}"][data-ds-x="${i}"]`)?.value)||0);
      const y=Array.from({length:cfg.numOutputs},(_,i)=>parseFloat(document.querySelector(`[data-ds-row="${r}"][data-ds-y="${i}"]`)?.value)||0);
      data.push({x,y});
    } return data;
  }

  function fillDatasetEditor(data){
    const safe=Array.isArray(data)?data:[]; document.getElementById('cfgDatasetSize').value=Math.max(1,Math.min(500,safe.length||1)); renderDatasetEditor();
    safe.forEach((row,r)=>{ (row.x||[]).slice(0,cfg.numInputs).forEach((v,i)=>{const e=document.querySelector(`[data-ds-row="${r}"][data-ds-x="${i}"]`);if(e)e.value=v;}); (row.y||[]).slice(0,cfg.numOutputs).forEach((v,i)=>{const e=document.querySelector(`[data-ds-row="${r}"][data-ds-y="${i}"]`);if(e)e.value=v;}); });
  }

  function prepareDataset(){
    const source=document.getElementById('cfgDatasetSource').value; const n=Math.max(1,Math.min(500,parseInt(document.getElementById('cfgDatasetSize').value,10)||10));
    if(source==='random'){ const data=Array.from({length:n},randomSample); fillDatasetEditor(data); document.getElementById('datasetStatus').textContent=`Generated ${data.length} random samples.`; }
    else if(source==='localStorage'){ try{ const data=JSON.parse(localStorage.getItem('mlp_dataset')||'null'); if(!Array.isArray(data)||!data.length) throw new Error('No saved dataset found.'); fillDatasetEditor(data); document.getElementById('datasetStatus').textContent=`Loaded ${data.length} samples from local storage.`; } catch(err){ document.getElementById('datasetStatus').textContent=`Local storage load failed: ${err.message}`; } }
    else { document.getElementById('cfgDatasetSize').value=n; renderDatasetEditor(); document.getElementById('datasetStatus').textContent=`Manual dataset ready — enter ${n} sample(s).`; }
  }

  function saveDatasetToLocalStorage(){ const data=readDatasetFromEditor(); localStorage.setItem('mlp_dataset',JSON.stringify(data)); document.getElementById('datasetStatus').textContent=`Saved ${data.length} samples to local storage.`; }

  function readSampleFromUI(){ return readDatasetFromEditor(); }

  function buildNetwork(c){
    const sizes = [c.numInputs, ...c.hidden, c.numOutputs];
    const weights = [], biases = [];
    for(let k=0;k<sizes.length-1;k++){
      const from = sizes[k], to = sizes[k+1];
      const W = []; for(let i=0;i<to;i++){
        const row = [];
        for(let j=0;j<from;j++){
          const el = document.querySelector(`[data-w-layer="${k}"][data-w-i="${i}"][data-w-j="${j}"]`);
          row.push(el ? (parseFloat(el.value) || 0) : defaultWeight());
        }
        W.push(row);
      }
      const B = []; for(let i=0;i<to;i++){
        const el = document.querySelector(`[data-b-layer="${k}"][data-b-i="${i}"]`);
        B.push(el ? (parseFloat(el.value) || 0) : defaultBias());
      }
      weights.push(W); biases.push(B);
    }
    return { layerSizes: sizes, weights, biases };
  }

  /* ============ Forward / Backward / Grad ============ */
  function forwardPass(net, c, x){
    const activations = [x.slice()];
    const zsAll = [null];
    let a = x.slice();
    const L = net.weights.length;
    for(let k=0;k<L;k++){
      const W = net.weights[k], B = net.biases[k];
      const to = W.length;
      const z = [];
      for(let i=0;i<to;i++){
        let sum = B[i];
        for(let j=0;j<W[i].length;j++) sum += W[i][j]*a[j];
        z.push(sum);
      }
      const actType = (k===L-1) ? c.outputAct : c.hiddenAct;
      const aNext = activate(z, actType);
      zsAll.push(z);
      activations.push(aNext);
      a = aNext;
    }
    return { activations, zsAll };
  }

  function lossAndBaseError(y, pred, lossFn){
    const n = y.length;
    let baseError = [], loss = 0;
    if(lossFn==='mae'){
      baseError = y.map((yi,i)=>Math.sign(yi-pred[i]));
      loss = y.reduce((s,yi,i)=>s+Math.abs(yi-pred[i]),0)/n;
    } else if(lossFn==='mse'){
      baseError = y.map((yi,i)=>(yi-pred[i]));
      loss = y.reduce((s,yi,i)=>s+(yi-pred[i])*(yi-pred[i]),0)/n;
    } else {
      baseError = y.map((yi,i)=>(yi-pred[i]));
      loss = y.reduce((s,yi,i)=>s+(yi-pred[i]),0)/n;
    }
    return { baseError, loss };
  }

  function backwardPass(net, c, activations, zsAll, y){
    const L = net.weights.length;
    const pred = activations[L];
    const { baseError, loss } = lossAndBaseError(y, pred, c.lossFn);
    const deltas = new Array(L);
    const outDelta = pred.map((p,i)=> baseError[i]*activDeriv(c.outputAct, p, zsAll[L][i]));
    deltas[L-1] = outDelta;
    for(let k=L-2;k>=0;k--){
      const a = activations[k+1], z = zsAll[k+1];
      const nextW = net.weights[k+1], nextDelta = deltas[k+1];
      const size = a.length;
      const d = [];
      for(let i=0;i<size;i++){
        let sum = 0;
        for(let j=0;j<nextDelta.length;j++) sum += nextW[j][i]*nextDelta[j];
        d.push(sum*activDeriv(c.hiddenAct, a[i], z[i]));
      }
      deltas[k] = d;
    }
    return { deltas, loss, pred, baseError };
  }

  function computeGrads(net, activations, deltas){
    const L = net.weights.length;
    const dW = [], dB = [];
    for(let k=0;k<L;k++){
      const from = activations[k], delta = deltas[k];
      const W = net.weights[k];
      const gW = []; for(let i=0;i<W.length;i++){ const row=[]; for(let j=0;j<W[i].length;j++) row.push(delta[i]*from[j]); gW.push(row); }
      dW.push(gW); dB.push(delta.slice());
    }
    return { dW, dB };
  }

  function zeroLikeW(net){ return net.weights.map(W=>W.map(row=>row.map(()=>0))); }
  function zeroLikeB(net){ return net.biases.map(B=>B.map(()=>0)); }
  function addInPlaceW(acc,g){ for(let k=0;k<acc.length;k++) for(let i=0;i<acc[k].length;i++) for(let j=0;j<acc[k][i].length;j++) acc[k][i][j]+=g[k][i][j]; }
  function addInPlaceB(acc,g){ for(let k=0;k<acc.length;k++) for(let i=0;i<acc[k].length;i++) acc[k][i]+=g[k][i]; }
  function applyGrads(net, c, sumW, sumB, count){
    const lr = c.lr;
    for(let k=0;k<net.weights.length;k++){
      for(let i=0;i<net.weights[k].length;i++){
        for(let j=0;j<net.weights[k][i].length;j++) net.weights[k][i][j] += lr*(sumW[k][i][j]/count);
        net.biases[k][i] += lr*(sumB[k][i]/count);
      }
    }
  }

  function trainOneEpoch(net, c, dataset){
    let totalLoss = 0, count = 0;
    const order = shuffleIdx(dataset.length);
    if(c.gdType==='stochastic'){
      order.forEach(idx=>{
        const { x, y } = dataset[idx];
        const { activations, zsAll } = forwardPass(net, c, x);
        const { deltas, loss } = backwardPass(net, c, activations, zsAll, y);
        const { dW, dB } = computeGrads(net, activations, deltas);
        applyGrads(net, c, dW, dB, 1);
        totalLoss += loss; count++;
      });
    } else {
      const bs = Math.max(1, Math.min(c.batchSize, dataset.length));
      for(let start=0; start<order.length; start+=bs){
        const batchIdx = order.slice(start, start+bs);
        let sumW = zeroLikeW(net), sumB = zeroLikeB(net), batchLoss = 0;
        batchIdx.forEach(idx=>{
          const { x, y } = dataset[idx];
          const { activations, zsAll } = forwardPass(net, c, x);
          const { deltas, loss } = backwardPass(net, c, activations, zsAll, y);
          const { dW, dB } = computeGrads(net, activations, deltas);
          addInPlaceW(sumW, dW); addInPlaceB(sumB, dB);
          batchLoss += loss;
        });
        applyGrads(net, c, sumW, sumB, batchIdx.length);
        totalLoss += batchLoss; count += batchIdx.length;
      }
    }
    return totalLoss/count;
  }

  /* ============ Layout / diagram ============ */
  function computeLayout(sizes){
    const maxSize = Math.max(...sizes);
    const H = Math.max(300, maxSize*64+90);
    const W = Math.max(680, sizes.length*180);
    const positions = sizes.map((n,l)=>{
      const x = sizes.length===1 ? W/2 : 70 + l*((W-140)/(sizes.length-1));
      const ys = [];
      for(let i=0;i<n;i++) ys.push(n===1 ? H/2 : 55 + i*((H-115)/(n-1)));
      return { x, ys };
    });
    const r = maxSize<=4 ? 21 : (maxSize<=6 ? 16 : 12);
    return { W, H, positions, r };
  }

  function layerName(l, numLayers){
    if(l===0) return 'input';
    if(l===numLayers-1) return 'output';
    return 'hidden ' + l;
  }
  function nodeLabel(l, i, numLayers){
    if(l===0) return 'x'+(i+1);
    if(l===numLayers-1) return 'y'+(i+1);
    return 'h'+l+'.'+(i+1);
  }

  function renderDiagram(){
    const svg = document.getElementById('netsvg');
    if(!net){ svg.innerHTML=''; return; }
    const sizes = net.layerSizes;
    const { W, H, positions, r } = computeLayout(sizes);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const totalEdges = sizes.slice(0,-1).reduce((s,n,k)=>s+n*sizes[k+1],0);
    const showLabels = totalEdges <= 36;

    let parts = [];
    for(let k=0;k<sizes.length-1;k++){
      for(let i=0;i<sizes[k];i++){
        for(let j=0;j<sizes[k+1];j++){
          const x1 = positions[k].x, y1 = positions[k].ys[i];
          const x2 = positions[k+1].x, y2 = positions[k+1].ys[j];
          const id = `e_${k}_${i}_${j}`;
          parts.push(`<path class="edge" id="${id}" d="M${x1},${y1} L${x2},${y2}"/>`);
          if(showLabels){
            const midx=(x1+x2)/2, midy=(y1+y2)/2;
            const wv = net.weights[k][j][i];
            parts.push(`<rect class="edge-label-bg" x="${midx-24}" y="${midy-8}" width="48" height="12" rx="3"/>`);
            parts.push(`<text class="edge-label" x="${midx}" y="${midy+2}" text-anchor="middle">${wv.toFixed(2)}</text>`);
          }
        }
      }
    }

    const act = inspect ? inspect.activations : null;
    for(let l=0;l<sizes.length;l++){
      parts.push(`<text class="layer-tag" x="${positions[l].x}" y="18">${layerName(l, sizes.length)}</text>`);
      for(let i=0;i<sizes[l];i++){
        const x = positions[l].x, y = positions[l].ys[i];
        const cls = l===0 ? '' : (l===sizes.length-1 ? 'output' : 'hidden');
        const val = act ? act[l][i] : null;
        parts.push(`<g id="n_${l}_${i}">
          <circle class="node-circle ${cls}" cx="${x}" cy="${y}" r="${r}"/>
          <text class="node-label" x="${x}" y="${y+4}" font-size="${r>=18?12:10}">${nodeLabel(l,i,sizes.length)}</text>
          ${val!==null ? `<text class="node-sub" x="${x}" y="${y+r+13}">${val.toFixed(3)}</text>` : ''}
        </g>`);
      }
    }
    svg.innerHTML = parts.join('');
  }

  function flashNodes(ids){
    ids.forEach(id=>{
      const g = document.getElementById(id);
      if(!g) return;
      g.classList.remove('node-flash'); void g.offsetWidth; g.classList.add('node-flash');
    });
  }
  function setLayerStatus(text, active=true){
    const el = document.getElementById('layerStatus');
    if(!el) return;
    el.textContent = text; el.classList.toggle('active', active);
  }
  function pulseLayerEdges(k, dur, reverse){
    const sizes = net.layerSizes;
    const cls = reverse ? 'pulse-back' : 'pulse-fwd';
    for(let i=0;i<sizes[k];i++) for(let j=0;j<sizes[k+1];j++){
      const p = document.getElementById(`e_${k}_${i}_${j}`);
      if(p) p.classList.add(cls);
    }
    setTimeout(()=>{
      for(let i=0;i<sizes[k];i++) for(let j=0;j<sizes[k+1];j++){
        const p = document.getElementById(`e_${k}_${i}_${j}`);
        if(p) p.classList.remove('pulse-fwd','pulse-back');
      }
    }, dur+80);
  }
  async function animateForward(){
    const L = net.weights.length;
    const DUR = Math.max(260, 650 - L*80);
    for(let k=0;k<L;k++){
      setLayerStatus(`① Layer ${k} → Layer ${k+1} — computing z, then activation`);
      pulseLayerEdges(k, DUR, false);
      await sleep(DUR+100);
      flashNodes(net.layerSizes[k+1] ? Array.from({length:net.layerSizes[k+1]}, (_,i)=>`n_${k+1}_${i}`) : []);
    }
    setLayerStatus('Forward pass complete — ŷ ready', false);
  }
  async function animateBackward(){
    const L = net.weights.length;
    const DUR = Math.max(260, 650 - L*80);
    for(let k=L-1;k>=0;k--){
      setLayerStatus(`② Layer ${k+1} → Layer ${k} — propagating error backward`);
      pulseLayerEdges(k, DUR, true);
      await sleep(DUR+100);
      flashNodes(Array.from({length:net.layerSizes[k]}, (_,i)=>`n_${k}_${i}`));
    }
    setLayerStatus('Backpropagation complete — ready to update weights', false);
  }

  /* ============ Calc log ============ */
  function entryHTML(title, expl, line, resultLine){
    return `<div class="entry"><div class="title">${title}</div><div class="expl">${expl}</div><div class="line">${line}</div>${resultLine?`<div class="result">${resultLine}</div>`:''}</div>`;
  }
  function prependLog(html){
    const log = document.getElementById('calcLog');
    log.innerHTML = html + log.innerHTML;
  }

  function logForward(x, activations, zsAll){
    const L = net.weights.length;
    let html = entryHTML('Forward — Input Layer', `Input vector loaded into ${cfg.numInputs} input node(s).`,
      x.map((v,i)=>`x${i+1} = ${v}`).join('\n'));
    for(let k=0;k<L;k++){
      const isOut = k===L-1;
      const actType = isOut ? cfg.outputAct : cfg.hiddenAct;
      const lines = [];
      for(let i=0;i<net.weights[k].length;i++){
        const terms = net.weights[k][i].map((w,j)=>`(${w.toFixed(3)}×${activations[k][j].toFixed(3)})`).join(' + ');
        lines.push(`z_${k+1}.${i+1} = ${terms} + b(${net.biases[k][i].toFixed(3)}) = ${zsAll[k+1][i].toFixed(4)}`);
        lines.push(`a_${k+1}.${i+1} = ${actType}(z) = ${activations[k+1][i].toFixed(4)}`);
      }
      html += entryHTML(`Forward — Layer ${k+1}${isOut?' (Output)':''}`, `Weighted sum plus bias, then ${actType} activation.`, lines.join('\n'));
    }
    prependLog(html);
  }
  function logLoss(y, pred, loss, baseError){
    const lines = y.map((yi,i)=>`output ${i+1}: y=${yi} , ŷ=${pred[i].toFixed(4)} , base-error=${baseError[i].toFixed(4)}`).join('\n');
    const html = entryHTML(`Loss — ${LOSS_LABEL[cfg.lossFn]}`, LOSS_NOTE[cfg.lossFn], lines, `Loss = ${loss.toFixed(5)}`);
    prependLog(html);
  }
  function logBackward(deltas, activations){
    const L = net.weights.length;
    let html = '';
    for(let k=L-1;k>=0;k--){
      const isOut = k===L-1;
      const lines = deltas[k].map((d,i)=>`δ_${k+1}.${i+1} = ${d.toFixed(5)}`).join('\n');
      html += entryHTML(`Backward — Layer ${k+1}${isOut?' (Output)':''}`,
        isOut ? 'δ = (loss error) × activation′(z)' : 'δ = (Σ next-layer weight × next δ) × activation′(z)', lines);
    }
    prependLog(html);
  }
  function logUpdate(dW, dB, lr){
    const L = net.weights.length;
    let html = '';
    for(let k=0;k<L;k++){
      const lines = [];
      for(let i=0;i<dW[k].length;i++){
        for(let j=0;j<dW[k][i].length;j++){
          const delta = lr*dW[k][i][j];
          lines.push(`w[${k}][${i}][${j}] += α·δ·a = ${delta.toFixed(5)} → ${net.weights[k][i][j].toFixed(4)}`);
        }
        lines.push(`b[${k}][${i}] += α·δ = ${(lr*dB[k][i]).toFixed(5)} → ${net.biases[k][i].toFixed(4)}`);
      }
      html += entryHTML(`Update — Layer ${k+1}`, `Weights and biases nudged by α·δ·(input from previous layer).`, lines.join('\n'));
    }
    prependLog(html);
  }

  /* ============ Rendering: config, dashboard, tables, charts ============ */
  function renderHiddenLayerRows(){
    const wrap = document.getElementById('hiddenLayersList');
    wrap.innerHTML = cfg.hidden.map((n,idx)=>`
      <div class="hl-row">
        <label>hidden layer ${idx+1}</label>
        <div class="hl-inline">
          <input type="number" min="1" max="8" value="${n}" data-hl-idx="${idx}">
          <button class="hl-remove" type="button" data-hl-remove="${idx}" title="Remove layer">✕</button>
        </div>
      </div>`).join('');
    wrap.querySelectorAll('input[data-hl-idx]').forEach(inp=>{
      inp.addEventListener('change', e=>{
        const idx = +e.target.dataset.hlIdx;
        let v = parseInt(e.target.value,10);
        if(Number.isNaN(v)) v = 1;
        v = Math.max(1, Math.min(8, v));
        cfg.hidden[idx] = v; e.target.value = v;
      });
    });
    wrap.querySelectorAll('button[data-hl-remove]').forEach(btn=>{
      btn.addEventListener('click', e=>{
        const idx = +e.target.dataset.hlRemove;
        cfg.hidden.splice(idx,1);
        renderHiddenLayerRows();
      });
    });
  }

  function readConfigFromUI(){
    cfg.numInputs = Math.max(1, Math.min(8, parseInt(document.getElementById('cfgInputs').value,10) || 1));
    cfg.numOutputs = Math.max(1, Math.min(6, parseInt(document.getElementById('cfgOutputs').value,10) || 1));
    cfg.hiddenAct = document.getElementById('cfgHiddenAct').value;
    cfg.outputAct = document.getElementById('cfgOutputAct').value;
    cfg.lossFn = document.getElementById('cfgLoss').value;
    cfg.gdType = document.getElementById('cfgGd').value;
    cfg.batchSize = Math.max(1, Math.min(500, parseInt(document.getElementById('cfgBatch').value,10) || 1));
    cfg.epochs = Math.max(1, Math.min(300, parseInt(document.getElementById('cfgEpochs').value,10) || 30));
    cfg.lr = Math.max(0.001, Math.min(2, parseFloat(document.getElementById('cfgLr').value) || 0.15));
    document.getElementById('cfgInputs').value = cfg.numInputs;
    document.getElementById('cfgOutputs').value = cfg.numOutputs;
    document.getElementById('cfgBatch').value = cfg.batchSize;
    document.getElementById('cfgEpochs').value = cfg.epochs;
    document.querySelector('main').classList.toggle('gd-mini', cfg.gdType==='mini-batch');
    document.body.classList.toggle('gd-mini', cfg.gdType==='mini-batch');
  }

  function renderArchTag(){
    const parts = [cfg.numInputs, ...cfg.hidden, cfg.numOutputs].join(' → ');
    const actSym = { sigmoid:'σ', relu:'ReLU', softmax:'softmax' };
    document.getElementById('archTag').textContent = `${parts} · ${actSym[cfg.hiddenAct]}/${actSym[cfg.outputAct]}`;
  }

  function renderDashboard(){
    const grid = document.getElementById('dashGrid');
    const lastLoss = history.length ? history[history.length-1].loss : null;
    const minLoss = history.length ? Math.min(...history.map(h=>h.loss)) : null;
    grid.innerHTML = `
      <div class="dash-cell"><div class="label">Architecture</div><div class="value" style="font-size:15px;">${[cfg.numInputs,...cfg.hidden,cfg.numOutputs].join('-')}</div></div>
      <div class="dash-cell"><div class="label">Optimizer</div><div class="value" style="font-size:15px;">${cfg.gdType==='mini-batch' ? 'mini-batch ('+cfg.batchSize+')' : 'stochastic'}</div></div>
      <div class="dash-cell"><div class="label">Dataset samples</div><div class="value">${dataset.length}</div></div>
      <div class="dash-cell step"><div class="label">Updates / epoch</div><div class="value">${cfg.gdType==='stochastic' ? dataset.length : Math.ceil(dataset.length/Math.max(1,cfg.batchSize))}</div></div>
      <div class="dash-cell"><div class="label">Loss function</div><div class="value" style="font-size:15px;">${LOSS_LABEL[cfg.lossFn]}</div></div>
      <div class="dash-cell step"><div class="label">Epochs run</div><div class="value">${epochsRun}</div></div>
      <div class="dash-cell loss"><div class="label">Latest epoch loss</div><div class="value">${lastLoss!==null?lastLoss.toFixed(5):'—'}</div></div>
      <div class="dash-cell loss"><div class="label">Best epoch loss</div><div class="value">${minLoss!==null?minLoss.toFixed(5):'—'}</div></div>
    `;
  }

  function renderWeightTables(){
    const wrap = document.getElementById('weightTables');
    let html = '';
    for(let k=0;k<net.weights.length;k++){
      const from = net.layerSizes[k], to = net.layerSizes[k+1];
      const isOut = k===net.weights.length-1;
      let rows = '';
      for(let i=0;i<to;i++){
        for(let j=0;j<from;j++){
          rows += `<tr><td>w[${nodeLabel(k,j,net.layerSizes.length)}→${nodeLabel(k+1,i,net.layerSizes.length)}]</td><td>${net.weights[k][i][j].toFixed(4)}</td></tr>`;
        }
        rows += `<tr><td>b[${nodeLabel(k+1,i,net.layerSizes.length)}]</td><td>${net.biases[k][i].toFixed(4)}</td></tr>`;
      }
      html += `<details class="layer-table" ${k===0?'open':''}>
        <summary>Layer ${k+1}${isOut?' (Output)':''} — ${from}→${to} — ${from*to} weights + ${to} biases</summary>
        <div class="table-scroll"><table><thead><tr><th>parameter</th><th>value</th></tr></thead><tbody>${rows}</tbody></table></div>
      </details>`;
    }
    wrap.innerHTML = html;
  }

  function lineChart(svgId, series, opts){
    const svg = document.getElementById(svgId);
    const W = 400, H = 220, padL = 42, padR = 14, padT = 14, padB = 30;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const allPoints = series.flatMap(s=>s.points);
    if(allPoints.length===0){ svg.innerHTML=''; return; }
    const xs = allPoints.map(p=>p.x), ys = allPoints.map(p=>p.y);
    let yMin = opts.yMin!==undefined ? opts.yMin : Math.min(...ys);
    let yMax = opts.yMax!==undefined ? opts.yMax : Math.max(...ys);
    if(yMin===yMax){ yMin-=0.5; yMax+=0.5; }
    const pad=(yMax-yMin)*0.12; yMin-=pad; yMax+=pad;
    const xMin = Math.min(...xs,0), xMax = Math.max(...xs,1);
    const sx = x => padL + (xMax===xMin?0:(x-xMin)/(xMax-xMin))*plotW;
    const sy = y => padT + plotH - ((y-yMin)/(yMax-yMin))*plotH;
    let parts = [];
    parts.push(`<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT+plotH}" stroke="var(--card-edge)" stroke-width="1"/>`);
    parts.push(`<line x1="${padL}" y1="${padT+plotH}" x2="${padL+plotW}" y2="${padT+plotH}" stroke="var(--card-edge)" stroke-width="1"/>`);
    const ticks=4;
    for(let i=0;i<=ticks;i++){
      const val = yMin+(yMax-yMin)*i/ticks, y = sy(val);
      parts.push(`<line x1="${padL-3}" y1="${y}" x2="${padL+plotW}" y2="${y}" stroke="var(--card-edge)" stroke-width="1"/>`);
      parts.push(`<text x="${padL-6}" y="${y+3}" font-size="8" font-family="IBM Plex Mono, monospace" fill="var(--ink-faint)" text-anchor="end">${val.toFixed(3)}</text>`);
    }
    const xSteps = Math.min(xMax-xMin, 8) || 1;
    for(let i=0;i<=xSteps;i++){
      const xv = Math.round(xMin + (xMax-xMin)*i/xSteps);
      const x = sx(xv);
      parts.push(`<text x="${x}" y="${padT+plotH+14}" font-size="8" font-family="IBM Plex Mono, monospace" fill="var(--ink-faint)" text-anchor="middle">${xv}</text>`);
    }
    series.forEach(s=>{
      if(s.points.length===0) return;
      const path = s.points.map((p,i)=>`${i===0?'M':'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
      parts.push(`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2"/>`);
      if(s.points.length<=60) s.points.forEach(p=>parts.push(`<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="2.6" fill="${s.color}" stroke="var(--card)" stroke-width="1"/>`));
    });
    svg.innerHTML = parts.join('');
  }
  function renderLossChart(){
    lineChart('lossChart', [{ points: history.map(h=>({x:h.epoch,y:h.loss})), color:'#00e5ff' }], { yMin:0 });
  }

  function currentPrediction(){
    if(!net || !dataset.length) return null;
    const { activations } = forwardPass(net, cfg, dataset[0].x);
    return activations[net.weights.length].slice();
  }

  function renderHistoryTable(){
    const head = document.getElementById('historyTableHead');
    const body = document.getElementById('historyTableBody');
    const emptyNote = document.getElementById('historyEmptyNote');
    if(!history.length){
      head.innerHTML = ''; body.innerHTML = '';
      emptyNote.style.display = 'block';
      return;
    }
    emptyNote.style.display = 'none';
    const numOutputs = (history[0].pred && history[0].pred.length) || 0;
    let headHtml = '<th>Epoch</th><th>Loss</th>';
    for(let i=0;i<numOutputs;i++) headHtml += `<th>ŷ${i+1}</th>`;
    head.innerHTML = headHtml;
    body.innerHTML = history.map(h=>{
      let row = `<td>${h.epoch}</td><td>${h.loss.toFixed(5)}</td>`;
      (h.pred||[]).forEach(p=> row += `<td>${p.toFixed(4)}</td>`);
      return `<tr>${row}</tr>`;
    }).join('');
    const scrollWrap = body.closest('.table-scroll');
    if(scrollWrap) scrollWrap.scrollTop = scrollWrap.scrollHeight;
  }

  function renderIoBars(){
    const wrap = document.getElementById('ioBars');
    const tag = document.getElementById('sampleTag');
    if(!inspect){ wrap.innerHTML = '<div class="mono status-line">Run ① Forward Propagation on a sample to see this.</div>'; tag.textContent=''; return; }
    tag.textContent = `sample #${inspect.idx+1}`;
    const y = dataset[inspect.idx].y;
    const pred = inspect.pred || inspect.activations[net.weights.length];
    wrap.innerHTML = y.map((yi,i)=>`
      <div class="io-row">
        <div class="io-label mono">output ${i+1}</div>
        <div class="io-bar-track"><div class="io-bar-fill actual" style="width:${Math.min(100,Math.max(0,yi*100))}%"></div></div>
        <div class="io-val mono">y=${yi.toFixed(3)}</div>
      </div>
      <div class="io-row">
        <div class="io-label mono"></div>
        <div class="io-bar-track"><div class="io-bar-fill pred" style="width:${Math.min(100,Math.max(0,pred[i]*100))}%"></div></div>
        <div class="io-val mono">ŷ=${pred[i].toFixed(3)}</div>
      </div>
    `).join('');
  }

  function renderSampleLoss(){
    document.getElementById('lossFnLabel').textContent = LOSS_LABEL[cfg.lossFn];
    document.getElementById('lossFormulaNote').textContent = LOSS_NOTE[cfg.lossFn];
    if(!inspect || inspect.loss===undefined){
      document.getElementById('sampleLossCard').textContent = '—';
      document.getElementById('sampleErrCard').textContent = '—';
      return;
    }
    document.getElementById('sampleLossCard').textContent = inspect.loss.toFixed(5);
    const meanErr = inspect.baseError ? (inspect.baseError.reduce((a,b)=>a+b,0)/inspect.baseError.length) : NaN;
    document.getElementById('sampleErrCard').textContent = isFinite(meanErr) ? meanErr.toFixed(5) : '—';
  }

  function renderSampleSelect(){
    const sel = document.getElementById('sampleSelect');
    sel.innerHTML = dataset.map((d,i)=>`<option value="${i}">#${i+1} — x:[${d.x.join(', ')}]</option>`).join('');
    if(inspect) sel.value = inspect.idx;
  }

  function updateSidebarStatus(){
    const network = [cfg.numInputs,...cfg.hidden,cfg.numOutputs].join(' → ');
    const netEl = document.getElementById('sidebarNetwork');
    const dataEl = document.getElementById('sidebarDataset');
    const modeEl = document.getElementById('sidebarMode');
    if(netEl) netEl.textContent = network;
    if(dataEl) dataEl.textContent = `Dataset: ${dataset.length} samples`;
    if(modeEl) modeEl.textContent = `Mode: ${cfg.gdType==='mini-batch' ? 'Mini-batch' : 'Stochastic'}`;
  }

  function renderAll(){
    updateSidebarStatus();
    renderArchTag();
    renderDashboard();
    renderDiagram();
    renderWeightTables();
    renderLossChart();
    renderHistoryTable();
    renderIoBars();
    renderSampleLoss();
  }

  /* ============ Rebuild / reset actions ============ */
  function rebuildEverything(regenFields){
    readConfigFromUI();
    if(regenFields) renderParamInputs(); // fresh input/weight/bias fields matching the new architecture
    dataset = readDatasetFromEditor();
    net = buildNetwork(cfg);
    history = []; epochsRun = 0; inspect = null; inspectStage = 0;
    stepDone = new Set(); renderSteps(0);
    document.getElementById('calcLog').innerHTML = '<div class="mono status-line">Nothing computed yet — try Forward Propagation on a sample, or press Train.</div>';
    renderSampleSelect();
    renderAll();
    document.getElementById('cfgStatus').innerHTML =
      `Built <b>${[cfg.numInputs,...cfg.hidden,cfg.numOutputs].join('-')}</b> network. Fill in your input, weight &amp; bias values below, then press Apply Values.`;
    setLayerStatus('Idle — press a control to watch it propagate layer by layer', false);
  }

  function applyEnteredValues(){
    readConfigFromUI();
    dataset = readDatasetFromEditor();
    net = buildNetwork(cfg);
    history = []; epochsRun = 0; inspect = null; inspectStage = 0;
    stepDone = new Set(); renderSteps(0);
    document.getElementById('calcLog').innerHTML = '<div class="mono status-line">Nothing computed yet — try Forward Propagation on a sample, or press Train.</div>';
    renderSampleSelect();
    renderAll();
    document.getElementById('valuesStatus').textContent = 'Applied — network weights, biases, input and target values updated from what you entered.';
    setLayerStatus('Idle — press a control to watch it propagate layer by layer', false);
  }

  /* ============ Inspect-sample actions ============ */
  function currentSampleIdx(){ return +document.getElementById('sampleSelect').value || 0; }

  function updateInspectStageNote(message){
    const el = document.getElementById('inspectStageNote');
    if(el) el.textContent = message;
  }

  function resetInspectOnly(){
    net = buildNetwork(cfg);
    inspect = null;
    inspectStage = 0;
    document.getElementById('calcLog').innerHTML = '<div class="mono status-line">Nothing computed yet — choose a sample and start with Forward Propagation.</div>';
    setLayerStatus('Idle — ready for a new inspection.', false);
    renderAll();
    updateInspectStageNote('Step 0 · Ready — choose a sample, then use Next Step to walk through the complete training step.');
  }

  async function nextInspectStep(){
    if(training) return;
    inspectStage = inspectStage >= 4 ? 1 : inspectStage + 1;
    if(inspectStage === 1){
      await doForwardInspect();
      updateInspectStageNote('Step 1 · Forward propagation complete — prediction is now available.');
    } else if(inspectStage === 2){
      doLossInspect();
      updateInspectStageNote('Step 2 · Loss computed — the prediction error is ready for backpropagation.');
    } else if(inspectStage === 3){
      await doBackpropInspect();
      updateInspectStageNote('Step 3 · Backpropagation complete — gradients are ready for the parameter update.');
    } else if(inspectStage === 4){
      doApplySampleUpdate();
      updateInspectStageNote('Step 4 · Parameters updated — press Next Step again for another training step.');
    }
  }

  async function runFullInspectTrainingStep(){
    if(training) return;
    setTrainButtonsDisabled(true);
    try{
      await doForwardInspect();
      doLossInspect();
      await doBackpropInspect();
      doApplySampleUpdate();
      inspectStage = 4;
      updateInspectStageNote('Training step complete · Forward → Loss → Backpropagation → Update.');
    } finally {
      setTrainButtonsDisabled(false);
    }
  }

  async function doForwardInspect(){
    const idx = currentSampleIdx();
    const { x } = dataset[idx];
    setLayerStatus('Running forward pass…');
    await animateForward();
    const { activations, zsAll } = forwardPass(net, cfg, x);
    inspect = { idx, activations, zsAll, pred: activations[net.weights.length] };
    logForward(x, activations, zsAll);
    markStep(1);
    inspectStage = 1;
    renderDiagram(); renderIoBars(); renderSampleLoss();
  }
  function doLossInspect(){
    if(!inspect){ setLayerStatus('Run Forward Propagation first.', true); return; }
    const { y } = dataset[inspect.idx];
    const pred = inspect.pred;
    const { baseError, loss } = lossAndBaseError(y, pred, cfg.lossFn);
    inspect.baseError = baseError; inspect.loss = loss;
    logLoss(y, pred, loss, baseError);
    markStep(2);
    inspectStage = 2;
    renderSampleLoss();
  }
  async function doBackpropInspect(){
    if(!inspect || inspect.loss===undefined){ setLayerStatus('Run Forward + Loss first.', true); return; }
    const { y } = dataset[inspect.idx];
    setLayerStatus('Running backward pass…');
    await animateBackward();
    const { deltas } = backwardPass(net, cfg, inspect.activations, inspect.zsAll, y);
    inspect.deltas = deltas;
    logBackward(deltas, inspect.activations);
    markStep(3);
    inspectStage = 3;
  }
  function doApplySampleUpdate(){
    if(!inspect || !inspect.deltas){ setLayerStatus('Run Forward + Loss + Backprop first.', true); return; }
    const { dW, dB } = computeGrads(net, inspect.activations, inspect.deltas);
    logUpdate(dW, dB, cfg.lr);
    applyGrads(net, cfg, dW, dB, 1);
    markStep(4);
    inspectStage = 4;
    inspect.deltas = null;
    renderDiagram(); renderWeightTables();
    setLayerStatus('Weights updated from this sample — architecture diagram and tables refreshed.', false);
  }

  /* ============ Training ============ */
  function setTrainButtonsDisabled(disabled){
    ['btnTrain','btnStepEpoch','btnResetTraining','btnBuild','btnApplyValues',
     'btnForward','btnLoss','btnBackprop','btnApplySample','btnNextStep','btnRunFullStep','btnResetInspect'].forEach(id=>{
      const el = document.getElementById(id); if(el) el.disabled = disabled;
    });
    document.getElementById('btnStop').disabled = !disabled;
  }

  async function runTraining(){
    if(training) return;
    training = true; stopRequested = false;
    setTrainButtonsDisabled(true);
    markStep(5);
    const target = cfg.epochs;
    const perEpochDelay = target<=20 ? 90 : (target<=80 ? 25 : 4);
    const updatesPerEpoch = cfg.gdType==='stochastic' ? dataset.length : Math.ceil(dataset.length/Math.max(1,cfg.batchSize));
    document.getElementById('trainStatus').textContent = `Training… ${target} epoch(s), ${dataset.length} samples/epoch, ${updatesPerEpoch} update(s)/epoch.`;
    for(let e=1; e<=target; e++){
      if(stopRequested) break;
      const avgLoss = trainOneEpoch(net, cfg, dataset);
      epochsRun += 1;
      history.push({ epoch: epochsRun, loss: avgLoss, pred: currentPrediction() });
      if(e % Math.max(1, Math.floor(target/40)) === 0 || e===target){
        renderDashboard(); renderLossChart(); renderHistoryTable(); renderDiagram(); renderWeightTables();
      }
      if(perEpochDelay>0) await sleep(perEpochDelay);
    }
    setTrainButtonsDisabled(false);
    training = false;
    document.getElementById('trainStatus').textContent = stopRequested
      ? `Stopped after ${epochsRun} total epoch(s) — latest loss ${history.length?history[history.length-1].loss.toFixed(5):'—'}.`
      : `Training complete — ${epochsRun} total epoch(s) run, latest loss ${history.length?history[history.length-1].loss.toFixed(5):'—'}.`;
    renderDashboard(); renderLossChart(); renderHistoryTable(); renderDiagram(); renderWeightTables();
  }

  function stepOneEpoch(){
    if(training) return;
    const avgLoss = trainOneEpoch(net, cfg, dataset);
    epochsRun += 1;
    history.push({ epoch: epochsRun, loss: avgLoss, pred: currentPrediction() });
    markStep(5);
    document.getElementById('trainStatus').textContent = `Ran 1 epoch (epoch ${epochsRun} total) — avg loss ${avgLoss.toFixed(5)}.`;
    renderDashboard(); renderLossChart(); renderHistoryTable(); renderDiagram(); renderWeightTables();
  }

  function resetTraining(){
    net = buildNetwork(cfg); // rebuilds from the weight/bias fields you typed in above
    history = []; epochsRun = 0; inspect = null; inspectStage = 0;
    stepDone = new Set(); renderSteps(0);
    document.getElementById('trainStatus').textContent = 'Weights reset to the values entered above — history cleared.';
    document.getElementById('calcLog').innerHTML = '<div class="mono status-line">Nothing computed yet — try Forward Propagation on a sample, or press Train.</div>';
    renderAll();
  }

  /* ============ Wire up ============ */
  document.getElementById('btnAddHidden').addEventListener('click', ()=>{
    if(cfg.hidden.length>=4) return;
    cfg.hidden.push(3);
    renderHiddenLayerRows();
  });
  document.getElementById('cfgGd').addEventListener('change', e=>{
    document.querySelector('main').classList.toggle('gd-mini', e.target.value==='mini-batch');
  });
  document.getElementById('cfgDatasetSource').addEventListener('change', prepareDataset);
  document.getElementById('cfgDatasetSize').addEventListener('change', ()=>{ if(document.getElementById('cfgDatasetSource').value!=='localStorage') renderDatasetEditor(); });
  document.getElementById('btnGenerateDataset').addEventListener('click', prepareDataset);
  document.getElementById('btnSaveDataset').addEventListener('click', saveDatasetToLocalStorage);
  document.getElementById('btnBuild').addEventListener('click', ()=>rebuildEverything(true));
  document.getElementById('btnApplyValues').addEventListener('click', applyEnteredValues);

  document.getElementById('btnTrain').addEventListener('click', runTraining);
  document.getElementById('btnStepEpoch').addEventListener('click', stepOneEpoch);
  document.getElementById('btnStop').addEventListener('click', ()=>{ stopRequested = true; });
  document.getElementById('btnResetTraining').addEventListener('click', resetTraining);

  document.getElementById('btnForward').addEventListener('click', doForwardInspect);
  document.getElementById('btnLoss').addEventListener('click', doLossInspect);
  document.getElementById('btnBackprop').addEventListener('click', doBackpropInspect);
  document.getElementById('btnApplySample').addEventListener('click', doApplySampleUpdate);
  document.getElementById('btnNextStep').addEventListener('click', nextInspectStep);
  document.getElementById('btnRunFullStep').addEventListener('click', runFullInspectTrainingStep);
  document.getElementById('btnResetInspect').addEventListener('click', resetInspectOnly);
  document.getElementById('sampleSelect').addEventListener('change', ()=>{ inspect=null; inspectStage=0; updateInspectStageNote('Step 0 · Ready — selected sample changed. Start with Forward Propagation or Next Step.'); renderIoBars(); renderSampleLoss(); renderDiagram(); });

  /* ============ Init ============ */
  renderHiddenLayerRows();
  renderDatasetEditor();
  rebuildEverything(true);
})();