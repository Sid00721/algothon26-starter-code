#!/usr/bin/env python3
"""Build a self-contained interactive market dashboard from prices.txt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "output" / "interactive_dashboard.html"


def clean(values: np.ndarray, decimals: int = 6) -> list:
    array = np.asarray(values, dtype=float)
    rounded = np.round(array, decimals)
    return [[None if not np.isfinite(value) else float(value) for value in row] for row in rounded]


def rolling_structure(returns: np.ndarray, window: int) -> dict[str, list[float | None]]:
    n_assets, n_days = returns.shape
    average_correlation: list[float | None] = [None] * n_days
    median_volatility: list[float | None] = [None] * n_days
    first_pc: list[float | None] = [None] * n_days
    for day in range(window, n_days):
        sample = returns[:, day - window + 1 : day + 1]
        correlation = np.corrcoef(sample)
        off_diagonal = correlation[~np.eye(n_assets, dtype=bool)]
        average_correlation[day] = round(float(np.nanmean(off_diagonal)), 5)
        median_volatility[day] = round(float(np.nanmedian(np.std(sample, axis=1, ddof=1) * np.sqrt(250))), 5)
        eigenvalues = np.linalg.eigvalsh(np.nan_to_num(correlation, nan=0.0))
        first_pc[day] = round(float(eigenvalues[-1] / np.maximum(eigenvalues.sum(), 1e-12)), 5)
    return {
        "averageCorrelation": average_correlation,
        "medianVolatility": median_volatility,
        "firstPC": first_pc,
    }


def main() -> None:
    prices_frame = pd.read_csv(ROOT / "prices.txt", sep=r"\s+")
    names = prices_frame.columns.tolist()
    prices = prices_frame.values.T.astype(float)
    n_assets, n_days = prices.shape
    log_prices = np.log(prices)
    returns = np.full_like(prices, np.nan)
    returns[:, 1:] = np.diff(log_prices, axis=1)
    normalized = prices / prices[:, [0]]
    drawdowns = prices / np.maximum.accumulate(prices, axis=1) - 1

    valid_returns = returns[:, 1:]
    annual_return = np.mean(valid_returns, axis=1) * 250
    annual_volatility = np.std(valid_returns, axis=1, ddof=1) * np.sqrt(250)
    sharpe = annual_return / np.maximum(annual_volatility, 1e-12)
    full_correlation = np.corrcoef(valid_returns)
    eigenvalues, eigenvectors = np.linalg.eigh(full_correlation)
    order = np.argsort(eigenvalues)[::-1]

    payload = {
        "names": names,
        "days": list(range(1, n_days + 1)),
        "prices": clean(prices, 4),
        "normalized": clean(normalized, 5),
        "returns": clean(returns, 6),
        "drawdowns": clean(drawdowns, 5),
        "annualReturn": np.round(annual_return, 5).tolist(),
        "annualVolatility": np.round(annual_volatility, 5).tolist(),
        "sharpe": np.round(sharpe, 4).tolist(),
        "pcaVariance": np.round(eigenvalues[order] / eigenvalues.sum(), 6).tolist(),
        "pcaLoadings": np.round(eigenvectors[:, order[:3]], 5).tolist(),
        "rolling": {str(window): rolling_structure(returns, window) for window in (20, 60, 120)},
    }

    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Algothon 2026 Quant Market Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.33.0/plotly.min.js"></script>
<style>
  :root { color-scheme: light dark; --gap: 18px; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 20px; background: Canvas; color: CanvasText; font-family: ui-sans-serif, system-ui, sans-serif; }
  #quant-market-dashboard { width: 100%; max-width: 1500px; margin: 0 auto; }
  h1, h2 { font-weight: 500; margin: 0 0 10px; }
  h1 { font-size: clamp(1.45rem, 3vw, 2.2rem); }
  h2 { margin-top: 30px; font-size: 1.2rem; }
  p { margin: 0 0 16px; color: GrayText; }
  .controls { display: flex; flex-wrap: wrap; align-items: end; gap: 14px; margin: 10px 0 8px; }
  label { display: grid; gap: 5px; font-size: 0.9rem; }
  select, input { font: inherit; color: CanvasText; background: Field; border: 1px solid GrayText; padding: 7px 9px; border-radius: 6px; }
  input[type="range"] { width: min(420px, 80vw); padding: 0; }
  .plot { width: 100%; min-height: 360px; }
  .plot.large { min-height: 650px; }
  .plot.medium { min-height: 470px; }
  .pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--gap); }
  .metric-line { display: flex; flex-wrap: wrap; gap: 18px; font-variant-numeric: tabular-nums; margin: 5px 0 10px; }
  .metric-line span { color: GrayText; }
  .metric-line strong { color: CanvasText; font-weight: 500; }
  @media (max-width: 900px) { .pair { grid-template-columns: 1fr; } body { padding: 12px; } }
</style>
</head>
<body>
<main id="quant-market-dashboard">
  <h1>Algothon 2026 market structure</h1>
  <p>Prices, returns, risk, dependence, regimes and technical context for all 51 instruments across 500 observed days.</p>

  <h2>All normalized price series</h2>
  <div id="normalized-chart" class="plot medium" role="img" aria-label="All 51 price series normalized to one on day one"></div>

  <h2>All daily log returns</h2>
  <div id="returns-chart" class="plot medium" role="img" aria-label="Daily log returns for all 51 instruments"></div>

  <h2>Rolling covariance and correlation</h2>
  <div class="controls">
    <label>Matrix
      <select id="matrix-type"><option value="correlation">Correlation</option><option value="covariance">Annualized covariance</option></select>
    </label>
    <label>Rolling window
      <select id="matrix-window"><option value="20">20 days</option><option value="60" selected>60 days</option><option value="120">120 days</option></select>
    </label>
    <label>As-of day <strong id="asof-label">500</strong>
      <input id="asof-day" type="range" min="121" max="500" value="500" step="1">
    </label>
  </div>
  <div class="metric-line" id="matrix-metrics"></div>
  <div id="matrix-chart" class="plot large" role="img" aria-label="Interactive rolling dependence matrix"></div>

  <h2>Dependence and volatility regimes</h2>
  <div class="controls">
    <label>Regime window
      <select id="regime-window"><option value="20">20 days</option><option value="60" selected>60 days</option><option value="120">120 days</option></select>
    </label>
  </div>
  <div id="regime-chart" class="plot medium" role="img" aria-label="Rolling average correlation, first principal component share and median volatility"></div>

  <h2>Rolling volatility across every asset</h2>
  <div class="controls">
    <label>Volatility window
      <select id="vol-window"><option value="20" selected>20 days</option><option value="60">60 days</option><option value="120">120 days</option></select>
    </label>
  </div>
  <div id="volatility-chart" class="plot large" role="img" aria-label="Rolling annualized volatility heatmap for all instruments"></div>

  <h2>Factor structure and risk-return map</h2>
  <div class="pair">
    <div id="pca-chart" class="plot medium" role="img" aria-label="Principal component explained variance"></div>
    <div id="risk-return-chart" class="plot medium" role="img" aria-label="Annualized mean return versus volatility for every asset"></div>
  </div>

  <h2>Drawdowns across every asset</h2>
  <div id="drawdown-chart" class="plot medium" role="img" aria-label="Drawdowns from running peaks for all instruments"></div>

  <h2>Selected-asset technical context</h2>
  <div class="controls">
    <label>Instrument
      <select id="asset-select"></select>
    </label>
  </div>
  <div id="technical-chart" class="plot large" role="img" aria-label="Price, moving averages, Bollinger bands and RSI for selected asset"></div>
</main>
<script>
const DATA = __PAYLOAD__;
const root = document.getElementById('quant-market-dashboard');
const config = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']};
const colors = ['#2563eb','#dc2626','#059669','#9333ea','#d97706','#0891b2'];
const baseLayout = (title, yTitle) => ({
  title: {text: title, font: {size: 16}},
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {color: getComputedStyle(document.body).color},
  margin: {l: 62, r: 28, t: 48, b: 52},
  hovermode: 'x unified',
  xaxis: {title: 'Day', gridcolor: 'rgba(128,128,128,.16)'},
  yaxis: {title: yTitle, gridcolor: 'rgba(128,128,128,.16)'},
  legend: {orientation: 'h', y: -0.18}
});

function lineTraces(values, width=0.8) {
  return DATA.names.map((name, i) => ({x: DATA.days, y: values[i], name, type: 'scattergl', mode: 'lines', line: {width}, hovertemplate: `${name}: %{y:.4f}<extra></extra>`}));
}

Plotly.newPlot('normalized-chart', lineTraces(DATA.normalized), {...baseLayout('', 'Growth of $1'), showlegend: false}, config);
Plotly.newPlot('returns-chart', lineTraces(DATA.returns, 0.65), {...baseLayout('', 'Daily log return'), showlegend: false}, config);
Plotly.newPlot('drawdown-chart', lineTraces(DATA.drawdowns, 0.7), {...baseLayout('', 'Drawdown'), showlegend: false, yaxis: {...baseLayout('', '').yaxis, tickformat: '.0%'}}, config);

function matrixFor(asOf, window, type) {
  const end = asOf - 1;
  const start = Math.max(1, end - window + 1);
  const samples = DATA.returns.map(row => row.slice(start, end + 1).filter(v => v !== null));
  const means = samples.map(row => row.reduce((a,b)=>a+b,0) / row.length);
  const std = samples.map((row,i) => Math.sqrt(row.reduce((s,v)=>s+(v-means[i])**2,0) / Math.max(1,row.length-1)));
  const matrix = samples.map((row,i) => samples.map((other,j) => {
    let cov = 0;
    for (let k=0;k<row.length;k++) cov += (row[k]-means[i])*(other[k]-means[j]);
    cov /= Math.max(1,row.length-1);
    return type === 'covariance' ? cov*250 : cov / Math.max(1e-12,std[i]*std[j]);
  }));
  return {matrix, std, start: start+1, end: end+1};
}

function updateMatrix() {
  const type = document.getElementById('matrix-type').value;
  const window = +document.getElementById('matrix-window').value;
  const asOf = +document.getElementById('asof-day').value;
  document.getElementById('asof-label').textContent = asOf;
  const result = matrixFor(asOf, window, type);
  const off = [];
  for (let i=0;i<DATA.names.length;i++) for (let j=i+1;j<DATA.names.length;j++) off.push(type === 'correlation' ? result.matrix[i][j] : result.matrix[i][j]);
  const avg = off.reduce((a,b)=>a+b,0)/off.length;
  const sorted = [...result.std].sort((a,b)=>a-b);
  const medianVol = sorted[Math.floor(sorted.length/2)]*Math.sqrt(250);
  document.getElementById('matrix-metrics').innerHTML = `<span>Sample <strong>days ${result.start}–${result.end}</strong></span><span>Average pair correlation <strong>${type==='correlation' ? avg.toFixed(3) : '—'}</strong></span><span>Median annualized volatility <strong>${(100*medianVol).toFixed(1)}%</strong></span>`;
  const limit = type === 'correlation' ? 1 : Math.max(...result.matrix.flat().map(Math.abs));
  Plotly.react('matrix-chart', [{z: result.matrix, x: DATA.names, y: DATA.names, type: 'heatmap', colorscale: 'RdBu', reversescale: true, zmin: -limit, zmax: limit, hovertemplate: '%{y} × %{x}<br>%{z:.4f}<extra></extra>'}], {
    ...baseLayout('', ''), margin: {l: 80,r:30,t:20,b:90}, xaxis: {tickangle:-90, tickfont:{size:9}}, yaxis:{autorange:'reversed',tickfont:{size:9}}, hovermode:'closest'
  }, config);
}
['matrix-type','matrix-window','asof-day'].forEach(id => document.getElementById(id).addEventListener('input', updateMatrix));
updateMatrix();

function updateRegime() {
  const window = document.getElementById('regime-window').value;
  const series = DATA.rolling[window];
  const traces = [
    {x:DATA.days,y:series.averageCorrelation,name:'Average pair correlation',mode:'lines',line:{color:colors[0]}},
    {x:DATA.days,y:series.firstPC,name:'PC1 variance share',mode:'lines',line:{color:colors[3]}},
    {x:DATA.days,y:series.medianVolatility,name:'Median annualized volatility',mode:'lines',yaxis:'y2',line:{color:colors[4]}}
  ];
  const layout = {...baseLayout('', 'Dependence'), yaxis2:{title:'Annualized volatility',overlaying:'y',side:'right',tickformat:'.0%',gridcolor:'rgba(0,0,0,0)'}, legend:{orientation:'h',y:-.2}};
  Plotly.react('regime-chart',traces,layout,config);
}
document.getElementById('regime-window').addEventListener('change',updateRegime);
updateRegime();

function rollingVol(window) {
  return DATA.returns.map(row => row.map((_,day) => {
    if (day < window) return null;
    const sample = row.slice(day-window+1,day+1).filter(v=>v!==null);
    const mean = sample.reduce((a,b)=>a+b,0)/sample.length;
    const variance = sample.reduce((s,v)=>s+(v-mean)**2,0)/Math.max(1,sample.length-1);
    return Math.sqrt(variance*250);
  }));
}
function updateVolatility() {
  const window=+document.getElementById('vol-window').value;
  Plotly.react('volatility-chart',[{z:rollingVol(window),x:DATA.days,y:DATA.names,type:'heatmap',colorscale:'Viridis',hovertemplate:'%{y}, day %{x}<br>ann. vol %{z:.1%}<extra></extra>'}],{
    ...baseLayout('', ''),margin:{l:80,r:35,t:20,b:52},yaxis:{autorange:'reversed',tickfont:{size:9}},hovermode:'closest'
  },config);
}
document.getElementById('vol-window').addEventListener('change',updateVolatility);
updateVolatility();

Plotly.newPlot('pca-chart',[{x:DATA.pcaVariance.slice(0,15).map((_,i)=>i+1),y:DATA.pcaVariance.slice(0,15),type:'bar',marker:{color:colors[0]},hovertemplate:'PC%{x}: %{y:.1%}<extra></extra>'}],{
  ...baseLayout('PCA eigenvalue spectrum','Explained variance'),xaxis:{title:'Principal component',dtick:1},yaxis:{title:'Explained variance',tickformat:'.0%',gridcolor:'rgba(128,128,128,.16)'},showlegend:false
},config);

Plotly.newPlot('risk-return-chart',[{x:DATA.annualVolatility,y:DATA.annualReturn,text:DATA.names,mode:'markers+text',type:'scatter',textposition:'top center',textfont:{size:9},marker:{size:9,color:DATA.sharpe,colorscale:'RdBu',cmin:-1.5,cmax:1.5,colorbar:{title:'Sharpe'}},hovertemplate:'%{text}<br>vol %{x:.1%}<br>return %{y:.1%}<extra></extra>'}],{
  ...baseLayout('Full-sample risk-return map','Annualized mean log return'),xaxis:{title:'Annualized volatility',tickformat:'.0%',gridcolor:'rgba(128,128,128,.16)'},yaxis:{title:'Annualized mean log return',tickformat:'.0%',gridcolor:'rgba(128,128,128,.16)'},showlegend:false
},config);

const assetSelect=document.getElementById('asset-select');
DATA.names.forEach((name,i)=>{const option=document.createElement('option');option.value=i;option.textContent=name;assetSelect.appendChild(option);});

function sma(values,window){return values.map((_,i)=>i+1<window?null:values.slice(i-window+1,i+1).reduce((a,b)=>a+b,0)/window);}
function rollingStd(values,window){return values.map((_,i)=>{if(i+1<window)return null;const s=values.slice(i-window+1,i+1),m=s.reduce((a,b)=>a+b,0)/window;return Math.sqrt(s.reduce((a,b)=>a+(b-m)**2,0)/(window-1));});}
function rsi(values,window=14){const out=Array(values.length).fill(null);let gain=0,loss=0;for(let i=1;i<values.length;i++){const d=values[i]-values[i-1],g=Math.max(d,0),l=Math.max(-d,0);gain=i===1?g:(gain*(window-1)+g)/window;loss=i===1?l:(loss*(window-1)+l)/window;if(i>=window)out[i]=loss===0?100:100-100/(1+gain/loss);}return out;}
function updateTechnical(){
  const i=+assetSelect.value,p=DATA.prices[i],ma20=sma(p,20),ma50=sma(p,50),ma100=sma(p,100),sd20=rollingStd(p,20),r=rsi(p);
  const upper=ma20.map((v,j)=>v===null?null:v+2*sd20[j]),lower=ma20.map((v,j)=>v===null?null:v-2*sd20[j]);
  const traces=[
    {x:DATA.days,y:p,name:'Price',mode:'lines',line:{color:colors[0]}},
    {x:DATA.days,y:ma20,name:'MA20',mode:'lines',line:{color:colors[4],width:1}},
    {x:DATA.days,y:ma50,name:'MA50',mode:'lines',line:{color:colors[2],width:1}},
    {x:DATA.days,y:ma100,name:'MA100',mode:'lines',line:{color:colors[1],width:1}},
    {x:DATA.days,y:upper,name:'Bollinger upper',mode:'lines',line:{width:.5,color:'rgba(128,128,128,.5)'},hoverinfo:'skip'},
    {x:DATA.days,y:lower,name:'Bollinger lower',mode:'lines',fill:'tonexty',fillcolor:'rgba(128,128,128,.12)',line:{width:.5,color:'rgba(128,128,128,.5)'},hoverinfo:'skip'},
    {x:DATA.days,y:r,name:'RSI(14)',mode:'lines',yaxis:'y2',line:{color:colors[3]}}
  ];
  Plotly.react('technical-chart',traces,{...baseLayout(`${DATA.names[i]} technical context`,'Price'),yaxis:{domain:[.34,1],title:'Price',gridcolor:'rgba(128,128,128,.16)'},yaxis2:{domain:[0,.23],title:'RSI',range:[0,100],gridcolor:'rgba(128,128,128,.16)'},shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:70,y1:70,line:{dash:'dot',color:colors[1]}},{type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:30,y1:30,line:{dash:'dot',color:colors[2]}}],legend:{orientation:'h',y:1.08},hovermode:'x unified'},config);
}
assetSelect.addEventListener('change',updateTechnical);
updateTechnical();
</script>
</body>
</html>'''

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(template.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))))
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
