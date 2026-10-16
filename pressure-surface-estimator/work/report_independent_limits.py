"""Plot and package the measured exploratory depth/spatial envelope."""
from pathlib import Path
import json, shutil, html, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
depth=json.loads((OUT/'independent-depth/results.json').read_text())
space=json.loads((OUT/'wider-surface/results.json').read_text())
assert len(depth['runs'])==28
assert depth['frozen_method_and_original_pressure_integrity_pass']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
colors={'upstream':'#147d92','gaussian':'#b35b27'}
names={'upstream':'Package irregular spectrum','gaussian':'Gaussian frequency shape'}
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for kind in colors:
    rows=[r for r in depth['runs'] if r['case'].startswith(kind)]
    ds=sorted({r['centre_depth_m'] for r in rows})
    for ax,key,scale in [(axes[0],'pooled_R2',1),(axes[1],'pooled_RMSE_m',100)]:
        vals=np.array([[r[key]*scale for r in rows if r['centre_depth_m']==d] for d in ds])
        ax.plot(ds,vals.mean(axis=1),'-o',color=colors[kind],label=names[kind],ms=4)
        ax.fill_between(ds,vals.min(axis=1),vals.max(axis=1),color=colors[kind],alpha=.16)
        ax.grid(alpha=.18);ax.set_xlabel('Submarine centre depth (m)')
axes[0].set_ylabel('Surface R², all 403 locations pooled');axes[0].set_ylim(0,1.025)
axes[0].axhline(.95,color='#667085',ls=':',lw=1)
axes[0].text(149,.958,'0.95',ha='right',color='#667085',fontsize=9)
axes[1].set_ylabel('Surface height RMSE (cm)');axes[1].set_ylim(bottom=0)
axes[0].legend(frameon=False,loc='lower left',fontsize=9)
fig.suptitle('Depth test • 60 × 12 m surface • six pressures • no measurement noise',fontsize=13)
fig.savefig(OUT/'independent-depth/depth-accuracy.png');plt.close(fig)
if '--depth-plot-only' in sys.argv:sys.exit(0)
assert len(space['runs'])==4 and space['frozen_method_and_original_pressure_integrity_pass']

fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained',sharey=True)
for ax,kind in zip(axes,['upstream','gaussian']):
    files=sorted((OUT/'wider-surface').glob(kind+'*_map.npz'))
    maps=[np.load(f) for f in files];grid=maps[0]['grid'];ids=maps[0]['grid_indices']
    val=np.minimum.reduce([m['R2'][ids] for m in maps]).reshape(31,61)
    im=ax.pcolormesh(np.arange(-150,151,5),np.arange(-75,76,5),val,cmap='viridis',vmin=0,vmax=1,shading='nearest',rasterized=True)
    cs=ax.contour(np.arange(-150,151,5),np.arange(-75,76,5),val,levels=[.8,.95,.99],colors='white',linewidths=.8)
    ax.clabel(cs,fmt='%.2f',fontsize=8)
    ax.add_patch(Ellipse((0,0),70,10,fill=False,color='#b8440d',lw=2))
    ax.add_patch(Rectangle((-30,-6),60,12,fill=False,color='#b8440d',lw=1.1,ls='--'))
    ax.scatter([35,-35,0,0,0],[0,0,5,-5,0],s=11,color='#b8440d',zorder=4)
    ax.annotate('',xy=(43,25),xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#b8440d','lw':1.4})
    ax.set_title(names[kind]);ax.set_xlabel('Along hull x (m)');ax.set_aspect('equal')
axes[0].set_ylabel('Across hull y (m)')
cb=fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.07,pad=.12,shrink=.8)
cb.set_label('Pointwise surface R²: lower of the two phase seeds (values below 0 clipped to 0)')
fig.suptitle('Spatial reach at 15 m centre depth • orange: hull, ports, 60 × 12 m patch, travel direction',fontsize=12)
fig.savefig(OUT/'wider-surface/spatial-accuracy.png');plt.close(fig)

def interval(rows,key,scale=1,digits=3):
    a=[r[key]*scale for r in rows]
    return f'{min(a):.{digits}f}–{max(a):.{digits}f}'
depth_rows=[]
for d in depth['protocol']['prespecified_depths_m']:
    rows=[r for r in depth['runs'] if r['centre_depth_m']==d]
    depth_rows.append([str(d),interval(rows,'pooled_R2'),interval(rows,'pooled_RMSE_m',100,1),interval(rows,'minimum_point_R2')])
area_rows=[]
for length,width in space['protocol']['prespecified_centred_rectangles_m']:
    rows=[a for r in space['runs'] for a in r['areas'] if a['length_m']==length and a['width_m']==width]
    area_rows.append([f'{length} × {width}',interval(rows,'pooled_R2'),interval(rows,'pooled_RMSE_m',100,1),interval(rows,'minimum_point_R2')])
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
headers=['Centre depth (m)','Surface R²','Height RMSE (cm)','Worst-location R²']
depth_table=table(headers,depth_rows)
area_table=table(['Centred patch (m)','Surface R²','Height RMSE (cm)','Worst-location R²'],area_rows)
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Independent wave reconstruction: depth and reach</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#183044;background:#f1f5f7;margin:0}main{max-width:1120px;margin:auto;padding:32px 22px 65px}h1{font-size:32px;line-height:1.2;max-width:850px}h2{font-size:23px;margin-top:38px}p{max-width:930px}.tag{font-size:13px;font-weight:700;color:#147d92;letter-spacing:.05em}section{background:white;border-radius:14px;padding:24px;margin:24px 0;border:1px solid #d8e2e8}img{display:block;width:100%;height:auto}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums;font-size:15px}td,th{text-align:left;padding:11px 12px;border-bottom:1px solid #d8e2e8}th{background:#f2f6f8;font-size:13px}a{color:#086e87}.note{color:#4b6070;font-size:14px}.callout{border-left:4px solid #147d92;padding:12px 18px;background:#eaf5f7}code{font-size:13px}@media(max-width:600px){main{padding:20px 12px}section{padding:14px}h1{font-size:27px}td,th{padding:9px 8px}}</style><main>
<p class="tag">MDM3 · EXPLORATORY SIMULATION RESULTS · 15 OCTOBER 2026</p>
<h1>How deep, and how far beyond the hull, can six pressures reconstruct individual waves?</h1>
<p>Measured using the unchanged hybrid estimator and the independent CFDwavemaker source. Two single-system frequency shapes, two phase seeds each, no measurement noise, no crossing seas, no IMU. These are paired tests on the four existing sea realizations.</p>
<div class="callout"><strong>Read the two tests separately.</strong> The depth sweep holds the surface patch at 60 × 12 m. The spatial sweep holds centre depth at 15 m. They do not establish the combined accuracy of a large patch at a large depth.</div>
<section><h2>Depth: the recovered surface gradually loses detail</h2>
<img src="depth-accuracy.png" alt="Surface R squared falls and surface height error increases with submarine depth, with different curves for the two wave spectra.">
DEPTH_TABLE
<p class="note">Each range covers all four case/seed combinations. Worst-location R² is the least accurate of the 403 scored surface locations in each run. Shaded plot bands span two phase seeds per spectral shape; they are not confidence intervals. Depth means hull centre: the top port is 5 m shallower and the bottom port 5 m deeper.</p>
<p>There is no single physical maximum depth in this result. Short waves leave exponentially weaker pressure signatures at depth. The unchanged numerical stabilisation then suppresses information it cannot reliably invert; errors in the pressure-learned spectrum can also contribute. Lower stabilisation or a longer history may recover more in an ideal calculation, but these results do not demonstrate that improvement.</p>
<p>For an isolated deep-water wave component, the pressure amplitude relative to its surface value is <code>exp(−4π² d / (g T²))</code>. At 60 m, an 8-second wave retains about 2.3% while a 4-second wave retains only 0.000028%. This explains why a smooth reconstruction can still miss short-wave detail. This is the incident wave signal, before hull-motion contributions.</p></section>
<section><h2>Spatial reach: accuracy varies strongly with location</h2>
<img src="../wider-surface/spatial-accuracy.png" alt="Two pointwise accuracy maps over a 300 by 150 metre region, with the submarine and baseline surface patch marked.">
AREA_TABLE
<p class="note">All rectangles are centred on the hull and aligned with it. The 60 × 12 m benchmark retains its original 2 m along-hull × 1 m across-hull grid (403 points). Larger rectangles use a 5 m grid. The map uses the lower R² of the two phase seeds at each grid point, separately for each sea shape. Values between grid points are not independently checked. Pooled accuracy alone cannot certify every location.</p>
<p>The sensors constrain waves passing their limited footprint. Known dispersion and a pressure-learned statistical prior extend the estimate beyond that footprint, but distant lateral combinations remain ambiguous. Past measurements help some downstream locations more than upstream locations. A larger rendering is easy; a larger accurately measured region requires evidence like this map.</p></section>
<section><h2>What entered the estimator</h2>
<p>Only six current/past pressure readings, their timestamps, the assumed port geometry and the known nominal hull model/depth. The first 1,024 readings (8.5 minutes at 2 Hz) learn a statistical prior. The subsequent current-time estimate uses a 128-tap, 64-second pressure history. At every depth, statistics are learned again from that depth’s pressures. The spectrum family remains JONSWAP with fixed gamma and one directional system; the Gaussian source therefore tests a frequency-shape mismatch.</p>
<p>For spatial extension, the pressure record and its learned prior stay fixed. Only the requested output locations change. The evaluator obtains the true surface from the native external package; it supplies no phases or surface samples to the inverse. Causal batch predictions were checked against packet-by-packet streaming. The original 60 × 12 m spatial predictions and native reference surface were reproduced to numerical precision. All frozen source hashes and original pressure records were verified unchanged.</p>
<h2>Limits of this evidence</h2>
<p>CFDwavemaker supplies an independent implementation of linear Airy waves and incident pressure. It is not a full CFD solver. The forward simulation and inverse retain shared linear-wave, nominal hull-force, vehicle-dynamics and instrument assumptions. Sensor placement is assumed, not sponsor-verified. This does not measure real-hardware performance, nonlinear seas, noise robustness, changing seas, forward speed or wave-hull scattering. The two phase seeds do not establish a universal operating envelope.</p>
<p>These tests map the current method; no parameters were retuned for a better score. The next research step is to extend the causal wave-state estimator and compare it on fresh realizations, with a location-specific error estimate. A credible depth limit for hardware also needs actual sensor noise, resolution and model error.</p>
<p><a href="https://github.com/oystelan/CFDwavemaker">CFDwavemaker source</a> · <a href="https://ocw.mit.edu/courses/2-017j-design-of-electromechanical-robotic-systems-fall-2009/41e4cc0e5a0ebd845ea4f5e7c055fdf7_MIT2_017JF09_coursetext.pdf">MIT: linear wave pressure attenuation</a> · <a href="https://www.sciencedirect.com/science/article/pii/S0165212517301518">Phase-resolved wave predictable zones</a></p>
<p><a href="results.json">Depth results</a> · <a href="protocol.json">Depth protocol</a> · <a href="../wider-surface/results.json">Spatial results</a> · <a href="../wider-surface/protocol.json">Spatial protocol</a> · <a href="../independent-wave-package/index.html">Original independent-wave replay</a> · <a href="source-snapshot.zip">Experiment scripts</a></p></section></main></html>'''
page=page.replace('DEPTH_TABLE',depth_table).replace('AREA_TABLE',area_table)
(OUT/'independent-depth/index.html').write_text(page)
lines=['# Independent-wave depth and spatial envelope','', 'Noise-free, six pressures only. Four existing independent sea realizations, frozen estimator. Exploratory paired envelope tests; not fresh held-out validation.','', '## Depth: 60 x 12 m patch','', '| '+' | '.join(headers)+' |','|---|---|---|---|']
lines+=['| '+' | '.join(row)+' |' for row in depth_rows]
lines+=['','## Spatial reach: centre depth 15 m','','| Patch (m) | Surface R2 | Height RMSE (cm) | Worst-location R2 |','|---|---|---|---|']
lines+=['| '+' | '.join(row)+' |' for row in area_rows]
lines+=['','Ranges across both spectra and both phase seeds. Depth is hull centre, top port depth = centre minus 5 m. No combined large-area/deep-water result claimed.','', 'Read index.html for assumptions, grid resolution, causal regression checks and limitations. All estimator sources and original pressure records passed hash integrity checks. Results do not establish a fundamental physical maximum depth or real-hardware performance.','']
(OUT/'independent-depth/README.md').write_text('\n'.join(lines))
import zipfile
with zipfile.ZipFile(OUT/'independent-depth/source-snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['map_independent_limits.py','report_independent_limits.py','independent_wave_adapter.py']:
        z.write(ROOT/'work'/name,'work/'+name)
    z.write(OUT/'independent-wave-package/protocol.json','prior-experiment-protocol.json')
print('\n'.join(lines))
