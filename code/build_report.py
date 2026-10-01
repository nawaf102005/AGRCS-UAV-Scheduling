"""Generate publication-ready figures directly from executed results."""
import json,argparse,io,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=ROOT/'results')
args=parser.parse_args();RES=args.results.resolve();FIG=ROOT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
summary=json.loads((RES/'summary.json').read_text())
S={(r['case'],r['method']):r for r in summary}
manifest=json.loads((RES/'manifest.json').read_text())['cases']
bulk={r['method']:r for r in json.loads((RES/'bulk_summary.json').read_text())}
records=json.loads((RES/'per_seed.json').read_text())
ack_pairs=[]
for baseline in ['A-GRCS','Frozen-GRCS','DDQN-Match']:
    aa={r['seed']:r for r in records if r['case']=='ack_abrupt' and r['method']=='A-GRCS-ACK'}
    bb={r['seed']:r for r in records if r['case']=='ack_abrupt' and r['method']==baseline}
    for metric in ['goodput','failure','rf']:
        v=np.array([aa[s][metric]-bb[s][metric] for s in aa])
        ack_pairs.append(dict(baseline=baseline,metric=metric,difference=float(v.mean()),ci95=float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))))
(RES/'ack_paired.json').write_text(json.dumps(ack_pairs,indent=2))
lat=json.loads((RES/'isolated_latency.json').read_text())
COL={'A-GRCS':'#006b8f','A-GRCS-ACK':'#009e73','Frozen-GRCS':'#d55e00','Rolling-GRCS':'#cc79a7',
     'ACI-d2':'#6b5b95','ACI-Full':'#555555','DDQN-Match':'#b58900','PF-Power':'#7e8d23','Nominal':'#aaa'}
plt.rcParams.update({
    'font.family':'serif',
    'font.serif':['Times New Roman','Tinos','Liberation Serif','DejaVu Serif'],
    'font.size':8.0,
    'axes.labelsize':9.0,
    'xtick.labelsize':8.0,
    'ytick.labelsize':8.0,
    'legend.fontsize':8.0,
    'axes.spines.top':True,
    'axes.spines.right':True,
    'axes.spines.bottom':True,
    'axes.spines.left':True,
    'axes.edgecolor':'black',
    'axes.linewidth':0.8,
    'lines.linewidth':1.2,
    'lines.markersize':4.0,
    'axes.grid':True,
    'grid.alpha':.18,
    'pdf.fonttype':42,
    'ps.fonttype':42,
})

def trace(case,method,key):
    rel = Path(manifest[case].get('path') or '.')
    directory = (RES / rel / 'traces').resolve()
    files = sorted(directory.glob(f'{case}_*_{method}.npz'))
    if not files:
        raise FileNotFoundError(
            f'No trace files for case={case!r}, method={method!r} in {directory}. '
            'Run `python code/reproduce.py --profile paper` first, or pass --results '
            'to a complete reproduced-results directory containing traces/.'
        )
    arrays=[]
    for path in files:
        data=np.load(path)
        if key not in data:
            raise KeyError(f'{key!r} not found in {path}')
        arrays.append(data[key])
    return np.stack(arrays)
def box_axes(ax):
    """Force a complete rectangular border around every plot."""
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis='both', which='both', direction='out', width=0.8)

def save(fig,name):
    # Apply the four-sided box to every axes, including supplementary figures.
    for ax in fig.axes:
        box_axes(ax)
    fig.tight_layout()
    for ext in ['pdf','png']:
        buf=io.BytesIO()
        fig.savefig(buf,format=ext,dpi=600,bbox_inches='tight',pad_inches=0.03)
        temporary=FIG/f'.{name}.{ext}.tmp'
        temporary.write_bytes(buf.getvalue())
        os.replace(temporary,FIG/f'{name}.{ext}')
    plt.close(fig)

# -------------------------------------------------------------------------
# Main/supplementary figures
# Each panel is saved as an independent file with suffix _a, _b, or _c.
# The (a)/(b)/(c) labels should be supplied by the manuscript caption/subfigure,
# not drawn inside the plot.
# -------------------------------------------------------------------------

# Shift: (a) failed-slot moving mean, (b) cumulative graph miscoverage
window=100
fig,ax=plt.subplots(figsize=(5,3.15))
for m in ['A-GRCS','Frozen-GRCS','Rolling-GRCS','ACI-d2']:
    raw=trace('abrupt',m,'failure').mean(0)
    ax.plot(np.arange(window-1,len(raw))*.1,
            np.convolve(raw,np.ones(window)/window,'valid')*100,
            label=m,color=COL[m])
ax.axvline(105,color='grey',ls=':')
ax.axhline(10,color='black',lw=.6,ls=':')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Failed slots, 100-slot mean (%)')
ax.legend(fontsize=9,ncol=2)
save(fig,'shift_a')

fig,ax=plt.subplots(figsize=(5,3.15))
for m in ['A-GRCS','ACI-d2']:
    raw=trace('abrupt',m,'miscoverage').mean(0)
    t=np.arange(1,len(raw)+1)
    ax.plot(t*.1,np.cumsum(raw)/t*100,label=m,color=COL[m])
t=np.arange(100,3001)
ax.plot(t*.1,100*(.1+17.7/t),'k--',label='A-GRCS bound')
ax.axvline(105,color='grey',ls=':')
ax.axhline(10,color='black',lw=.6,ls=':')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Cumulative graph miscoverage (%)')
ax.set_ylim(0,35)
ax.legend(fontsize=9)
save(fig,'shift_b')

# Stress: (a) goodput, (b) failed slots
cases=['gradual','recurrent','interference','correlation95']
labels=['Gradual','Recurrent','Interference','High correlation']
methods=['A-GRCS','Frozen-GRCS','Rolling-GRCS','ACI-d2']

fig,ax=plt.subplots(figsize=(5,3.1))
for i,m in enumerate(methods):
    v=[S[c,m]['goodput'] for c in cases]
    er=[S[c,m]['goodput_ci'] for c in cases]
    ax.errorbar(np.arange(4)+(i-1.5)*.15,v,yerr=er,fmt='o',
                capsize=2,color=COL[m],label=m)
ax.set_xticks(range(4),labels)
ax.set_ylabel('Goodput (Mbit/s)')
ax.legend(fontsize=9)
save(fig,'stress_a')

fig,ax=plt.subplots(figsize=(5,3.1))
for i,m in enumerate(methods):
    v=[100*S[c,m]['failure'] for c in cases]
    er=[100*S[c,m]['failure_ci'] for c in cases]
    ax.errorbar(np.arange(4)+(i-1.5)*.15,v,yerr=er,fmt='o',
                capsize=2,color=COL[m],label=m)
ax.set_xticks(range(4),labels)
ax.set_ylabel('Failed slots (%)')
ax.axhline(10,ls=':',color='black')
ax.legend(fontsize=9)
save(fig,'stress_b')

# Scaling: (a) goodput, (b) scheduling latency
ks=[24,50,100]
cs=['abrupt','users_50','users_100']

fig,ax=plt.subplots(figsize=(5,3.1))
for m in ['A-GRCS','Frozen-GRCS','ACI-d2','ACI-Full']:
    ax.errorbar(ks,[S[c,m]['goodput'] for c in cs],
                yerr=[S[c,m]['goodput_ci'] for c in cs],
                marker='o',capsize=3,color=COL[m],label=m)
ax.set_xlabel('Users')
ax.set_xticks(ks)
ax.set_ylabel('Goodput (Mbit/s)')
ax.legend(fontsize=9)
save(fig,'scaling_a')

fig,ax=plt.subplots(figsize=(5,3.1))
for m in ['A-GRCS','ACI-d2','DDQN-Match']:
    rr=[next(r for r in lat if r['users']==K and r['method']==m) for K in ks]
    ax.plot(ks,[r['mean_ms'] for r in rr],'-o',color=COL[m],label=m+' mean')
    ax.plot(ks,[r['p95_ms'] for r in rr],':',color=COL[m],label=m+' p95')
ax.set_xlabel('Users')
ax.set_xticks(ks)
ax.set_ylabel('Serial scheduling latency (ms)')
ax.legend(fontsize=9,ncol=2)
save(fig,'scaling_b')

# Energy: (a) goodput-energy operating points, (b) budget feedback
cs=['eta_0.1','abrupt','eta_2.0','eta_10.0']

fig,ax=plt.subplots(figsize=(5,3.1))
for m in ['A-GRCS','ACI-d2','ACI-Full']:
    ax.plot([S[c,m]['rf'] for c in cs],
            [S[c,m]['goodput'] for c in cs],
            '-o',color=COL[m],label=m)
    if m=='A-GRCS':
        for c,label in zip(cs,['0.1','0.5','2','10']):
            ax.annotate(label,(S[c,m]['rf'],S[c,m]['goodput']),
                        xytext=(3,4),textcoords='offset points',fontsize=9)
ax.set_xlabel('RF energy (J/slot)')
ax.set_ylabel('Goodput (Mbit/s)')
ax.legend(fontsize=9)
save(fig,'energy_a')

fig,ax=plt.subplots(figsize=(5,3.1))
for m,color in [('A-GRCS',COL['A-GRCS']),('A-GRCS-Budget','#009e73')]:
    raw=trace('budget',m,'rf').mean(0)
    ax.plot(np.arange(99,3000)*.1,
            np.convolve(raw,np.ones(100)/100,'valid'),
            label=m,color=color)
ax.axhline(.35,color='black',ls=':',label='RF target')
ax.set_xlabel('Time (s)')
ax.set_ylabel('RF energy, 100-slot mean (J)')
ax.legend(fontsize=9)
save(fig,'energy_b')

# Selected graph degree: already a single panel
fig,ax=plt.subplots(figsize=(4.8,2.8))
cases=['stationary','abrupt','recurrent','interference']
bottom=np.zeros(4)
for d,color in zip([1,2,4],['#8dd3c7','#006b8f','#6b5b95']):
    v=np.array([(trace(c,'A-GRCS','degree')==d).mean() for c in cases])
    ax.bar(np.arange(4),v,bottom=bottom,label=f'd={d}',color=color)
    bottom+=v
ax.set_xticks(range(4),['Stationary','Abrupt','Recurrent','Interference'])
ax.set_ylabel('Selected-slot fraction')
ax.legend(ncol=3,fontsize=8)
save(fig,'degree')

# Feedback comparison: (a) goodput, (b) failed slots
cases=['ack_abrupt','ack_interference','ack_load']
labels=['Abrupt','Interference','High traffic']
methods=['A-GRCS','A-GRCS-ACK','Frozen-GRCS']

fig,ax=plt.subplots(figsize=(5,3.1))
for i,m in enumerate(methods):
    ax.bar(np.arange(3)+(i-1)*.24,
           [S[c,m]['goodput'] for c in cases],
           width=.23,
           yerr=[S[c,m]['goodput_ci'] for c in cases],
           capsize=2,color=COL[m],label=m)
ax.set_xticks(range(3),labels)
ax.set_ylabel('Goodput (Mbit/s)')
ax.legend(fontsize=9)
save(fig,'feedback_a')

fig,ax=plt.subplots(figsize=(5,3.1))
for i,m in enumerate(methods):
    ax.bar(np.arange(3)+(i-1)*.24,
           [100*S[c,m]['failure'] for c in cases],
           width=.23,
           yerr=[100*S[c,m]['failure_ci'] for c in cases],
           capsize=2,color=COL[m],label=m)
ax.set_xticks(range(3),labels)
ax.set_ylabel('Failed slots (%)')
ax.axhline(10,color='black',ls=':')
ax.legend(fontsize=9)
save(fig,'feedback_b')

# DDQN training: already a single panel
fig,ax=plt.subplots(figsize=(5,3))
for seed in range(3):
    logs=json.loads((ROOT/'models'/f'training_seed{seed}.json').read_text())['logs']
    logs=[r for r in logs if 'validation_reward' in r]
    ax.plot([r['step'] for r in logs],
            [r['validation_reward'] for r in logs],
            '-o',label=f'Training seed {seed}')
ax.set_xlabel('Training transitions')
ax.set_ylabel('Independent validation reward')
ax.legend(fontsize=8)
save(fig,'training')

print('Generated separated boxed figures from saved results.')
