"""Generate supplementary review figures and 30-seed paired checks."""
import json,io,argparse
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';OUT=RES/'reviewer';FIG=ROOT/'figures'/'supplementary';FIG.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--baseline-results',type=Path,default=RES);args=parser.parse_args();BASE=args.baseline_results
old=json.loads((BASE/'per_seed.json').read_text());new=json.loads((OUT/'per_seed.json').read_text());summary=json.loads((BASE/'summary.json').read_text());S={(r['case'],r['method']):r for r in summary}
def stats(v):
 v=np.asarray(v);return dict(mean=float(v.mean()),ci=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))),n=len(v))
def subset(case,method):return sorted([r for r in new if r['case']==case and r['method']==method],key=lambda r:r['seed'])
def mean_ci(rows,key):return stats([r[key] for r in rows])
rows30={}
for m in ['A-GRCS-ACK','Frozen-GRCS']:
 rows30[m]=sorted([r for r in old if r['case']=='ack_abrupt' and r['method']==m]+subset('extra_seeds',m),key=lambda r:r['seed'])
 assert [r['seed'] for r in rows30[m]]==list(range(30))
paired={}
for key in ['goodput','failure','rf']:
 v=np.array([a[key]-b[key] for a,b in zip(rows30['A-GRCS-ACK'],rows30['Frozen-GRCS'])]);st=stats(v)
 st['leave_one_out_mean_range']=[float(np.min([(v.sum()-x)/29 for x in v])),float(np.max([(v.sum()-x)/29 for x in v]))]
 paired[key]=st
(OUT/'paired30.json').write_text(json.dumps(paired,indent=2))
(OUT/'combined30.json').write_text(json.dumps(rows30,indent=2))
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
 'ps.fonttype':42
})
def box_axes(ax):
 for side in ['top','bottom','left','right']:
  ax.spines[side].set_visible(True)
  ax.spines[side].set_color('black')
  ax.spines[side].set_linewidth(0.8)
 ax.tick_params(axis='both',which='both',direction='out',width=0.8)

def save(fig,name):
 for ax in fig.axes:
  box_axes(ax)
 fig.tight_layout()
 for ext in ['pdf','png']:
  buf=io.BytesIO()
  fig.savefig(buf,format=ext,dpi=600,bbox_inches='tight',pad_inches=0.03)
  (FIG/f'{name}.{ext}').write_bytes(buf.getvalue())
 plt.close(fig)
# ---------------------------------------------------------------------
# Supplementary/reviewer figures.
# Every panel is exported separately. The manuscript should supply
# subfigure labels (a), (b), (c), while the image itself stays clean.
# ---------------------------------------------------------------------

# Risk sensitivity: (a) goodput, (b) regulated-risk metrics.
aa=[.05,.1,.2]

fig,ax=plt.subplots(figsize=(5,3.5))
for m,col,mark in [('A-GRCS','#0072B2','o'),('A-GRCS-ACK','#009E73','s')]:
 rr=[subset('alpha',f'{m}@{a}') if a!=.1 else
     [r for r in old if r['case']=='ack_abrupt' and r['method']==m]
     for a in aa]
 st=[mean_ci(r,'goodput') for r in rr]
 ax.errorbar(aa,[x['mean'] for x in st],
             yerr=[x['ci'] for x in st],
             color=col,marker=mark,capsize=3,label=m)
ax.set_xticks(aa)
ax.set_xlabel('Controller target α')
ax.set_ylabel('Goodput (Mbit/s)')
ax.legend(fontsize=8)
save(fig,'risk_sensitivity_a')

fig,ax=plt.subplots(figsize=(5,3.5))
for m,col,mark in [('A-GRCS','#0072B2','o'),('A-GRCS-ACK','#009E73','s')]:
 rr=[subset('alpha',f'{m}@{a}') if a!=.1 else
     [r for r in old if r['case']=='ack_abrupt' and r['method']==m]
     for a in aa]
 st=[mean_ci(r,'failure') for r in rr]
 ax.errorbar(aa,[100*x['mean'] for x in st],
             yerr=[100*x['ci'] for x in st],
             color=col,marker=mark,capsize=3,
             label=m+' delivery')
 if m=='A-GRCS':
  stg=[mean_ci(r,'miscoverage') for r in rr]
  ax.errorbar(aa,[100*x['mean'] for x in stg],
              yerr=[100*x['ci'] for x in stg],
              color='#D55E00',marker='^',capsize=3,
              label='A-GRCS graph error')
ax.plot(aa,[100*(a+3*(a+.02*(1-a))/(.02*3000)) for a in aa],
        'k--',lw=1,label='Controller bound')
ax.set_xticks(aa)
ax.set_xlabel('Controller target α')
ax.set_ylabel('Slots (%)')
ax.legend(fontsize=8)
save(fig,'risk_sensitivity_b')


# Existing one-factor sensitivity:
# (a) ACI step, (b) reference fraction, (c) reference-error bound.
sens_specs = [
 ([.005,.02,.05],
  ['gamma_0.005','abrupt','gamma_0.05'],
  'ACI step γ',
  'review_sensitivity_existing_a'),
 ([0,.025],
  ['no_overhead','abrupt'],
  'Reference fraction ζ',
  'review_sensitivity_existing_b'),
 ([.5,2],
  ['abrupt','reference2'],
  'Reference-error bound (dB)',
  'review_sensitivity_existing_c')
]

for xx,cases,label,name in sens_specs:
 fig,ax=plt.subplots(figsize=(5,3.3))
 for metric,col,mark,lab in [
     ('failure','#0072B2','o','Delivery failure'),
     ('miscoverage','#D55E00','^','Graph error')]:
  rr=[S[c,'A-GRCS'] for c in cases]
  ax.errorbar(xx,[100*r[metric] for r in rr],
              yerr=[100*r[metric+'_ci'] for r in rr],
              color=col,marker=mark,capsize=3,label=lab)
 ax.set_xlabel(label)
 ax.set_xticks(xx)
 ax.set_ylabel('Slots (%)')
 ax.legend(fontsize=8)
 save(fig,name)


# Thirty-seed paired differences:
# (a) goodput difference, (b) failed-slot difference.
paired_specs = [
 ('goodput',1,'Goodput difference (Mbit/s)','paired30_a'),
 ('failure',100,'Failed-slot difference (percentage points)','paired30_b')
]

for key,scale,label,name in paired_specs:
 fig,ax=plt.subplots(figsize=(5,3.3))
 v=np.array([a[key]-b[key] for a,b in
             zip(rows30['A-GRCS-ACK'],rows30['Frozen-GRCS'])])*scale
 ax.scatter(range(30),v,color='#0072B2',s=22)
 ax.axhline(0,color='black',lw=.8)
 st=paired[key]
 ax.axhline(st['mean']*scale,color='#D55E00',label='Paired mean')
 ax.axhspan((st['mean']-st['ci'])*scale,
            (st['mean']+st['ci'])*scale,
            color='#D55E00',alpha=.15,
            label='95% t interval of mean')
 ax.set_xlabel('Evaluation seed')
 ax.set_ylabel(label)
 ax.legend(fontsize=8)
 save(fig,name)

print(json.dumps(dict(paired=paired),indent=2))
