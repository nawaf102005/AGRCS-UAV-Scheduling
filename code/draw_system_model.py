"""Publication vector schematic; illustrative geometry, not experimental data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
f,ax=plt.subplots(figsize=(10,3.25));f.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98);ax.set(xlim=(0,10),ylim=(0,3.3));ax.axis('off')
b='#0072B2';g='#00845A';o='#C55A11';grey='#60717C'
def arr(a,z,c=grey,ls='-'):ax.add_patch(FancyArrowPatch(a,z,arrowstyle='-|>',mutation_scale=10,color=c,lw=1.2,linestyle=ls))
def box(x,y,w,h,label,c=b):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03',edgecolor=c,facecolor='#F6F9FC',lw=1));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10.5)
ax.text(.1,3.1,'(a) Orthogonal UAV downlink',weight='bold',fontsize=10);ax.text(4.05,3.1,'(b) Scheduling and selected-graph feedback',weight='bold',fontsize=10)
u=[(.65,2.45),(1.8,2.65),(2.95,2.45)];v=[(.4,.9),(1.15,.65),(2,.9),(2.8,.65),(3.4,1.05)]
for j,(x,y) in enumerate(u):
 for k in [j,j+1]:ax.plot([x,v[k][0]],[y,v[k][1]],'--',lw=.9,color='#AAB7C4',zorder=0)
for j,k in [(0,0),(1,2),(2,4)]:arr(u[j],v[k],b)
for j,(x,y) in enumerate(u):
 ax.plot([x-.11,x+.11],[y-.08,y+.08],color=b,lw=2);ax.plot([x-.11,x+.11],[y+.08,y-.08],color=b,lw=2)
 for dx,dy in [(-.11,-.08),(.11,.08),(-.11,.08),(.11,-.08)]:ax.add_patch(Circle((x+dx,y+dy),.055,fill=False,edgecolor=b))
 ax.text(x,y+.17,f'UAV {j+1}',ha='center',fontsize=9.5)
for k,(x,y) in enumerate(v):ax.plot(x,y,'o',color=grey,ms=4);ax.text(x,y-.18,f'User {k+1}',ha='center',fontsize=9)
ax.text(.08,1.55,'External\ninterferer',color=o,fontsize=9.5);arr((.5,1.5),v[1],o,':')
ax.plot([.15,.5],[.27,.27],'--',color='#AAB7C4');ax.text(.57,.27,'Candidate edge',va='center',fontsize=9.5)
ax.plot([2,2.35],[.27,.27],color=b);ax.text(2.42,.27,'Scheduled edge',va='center',fontsize=9.5)
ax.text(1.9,.02,'Illustrative geometry; orthogonal inter-UAV RBs',ha='center',fontsize=8,color=grey);ax.plot([3.85,3.85],[.1,2.95],color='#CDD6DF')
box(4.15,2.13,2.1,.65,'Forecasts, geometry,\nqueues and history');box(7,2.13,2.65,.65,'Graph-specific states\nand score buffers')
box(4.15,1.1,2.1,.65,'Evaluate graphs:\npower + assignment');box(7,1.1,2.65,.65,'Choose graph; execute\nassociation and payload')
box(4.15,.08,2.1,.65,'Selected-controller\nupdate only',g);box(7,.08,2.65,.65,'Graph references OR\nactual delivery outcomes',g)
arr((5.2,2.13),(5.2,1.75));arr((7,2.37),(6.25,1.7));arr((6.25,1.42),(7,1.42));arr((8.3,1.1),(8.3,.73),g);arr((7,.4),(6.25,.4),g)
ax.plot([4.15,4,4,6.65,6.65],[.4,.4,2.98,2.98,2.45],color=g,lw=1);arr((6.65,2.45),(7,2.45),g)
OUT=ROOT/'figures';OUT.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','png']:f.savefig(OUT/f'system_model.{ext}',dpi=600,bbox_inches='tight',pad_inches=.04)
plt.close(f)
