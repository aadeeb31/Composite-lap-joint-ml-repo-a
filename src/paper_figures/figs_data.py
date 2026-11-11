from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import gaussian_kde
from style import SERIF
plt.rcParams.update({'font.family':SERIF,'font.size':14,'axes.labelsize':15,'axes.titlesize':15,'axes.linewidth':1.2,'xtick.labelsize':13,'ytick.labelsize':13})
P=str(ROOT/'data/processed')+'/'; FIGD=str(ROOT/'figures')+'/'
C1,C2='#1565E0','#FF6B00'
cf=pd.read_csv(P+'CFRP_CFRP/CFRP_CFRP__MaxLoad.csv'); cc=pd.read_csv(P+'Composite_Composite/Composite_Composite__MaxLoad.csv')
ls=pd.read_csv(P+'CFRP_CFRP/CFRP_CFRP__LSS.csv'); cl=pd.read_csv(P+'Composite_Composite/Composite_Composite__LSS.csv')
fc=pd.read_csv(P+'CFRP_CFRP/CFRP_CFRP__FailureMode.csv'); fk=pd.read_csv(P+'Composite_Composite/Composite_Composite__FailureMode.csv')

def kde_line(ax,x,bins,color,log=False):
    # smooth curve on the count scale of the histogram (n x bin width x density)
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    z=np.log10(x) if log else x
    k=gaussian_kde(z); g=np.linspace(z.min(),z.max(),300)
    bw=np.diff(np.log10(bins) if log else bins).mean()
    y=k(g)*len(x)*bw
    ax.plot(10**g if log else g,y,color=color,lw=2.2)

def stacked(ax,a,b,bins,log=False):
    ax.hist(a,bins=bins,color=C2,edgecolor='k',lw=0.6,label='Composite-composite'); ax.hist(b,bins=bins,color=C1,edgecolor='k',lw=0.6,label='CFRP-CFRP')
    kde_line(ax,a,bins,'#B34700',log); kde_line(ax,b,bins,'#0B3E91',log)
    if log: ax.set_xscale('log')

# ---- Fig 21 with KDE curves
fig,ax=plt.subplots(1,3,figsize=(15.5,4.9))
x2=cc['MaxLoad_kN'].dropna(); x2=x2[x2>0]; x1=cf['MaxLoad_kN'].dropna(); x1=x1[x1>0]
bl=np.logspace(np.log10(min(x1.min(),x2.min())),np.log10(max(x1.max(),x2.max())),40)
stacked(ax[0],x2,x1,bl,True)
ax[0].axvline(25,color='k',ls='--',lw=1.2); ax[0].text(27,ax[0].get_ylim()[1]*0.55,'25 kN',fontsize=12)
ax[0].set_xlabel('Maximum load (kN)'); ax[0].set_ylabel('records'); ax[0].set_title('(a) Maximum load',loc='left',fontweight='bold'); ax[0].legend(fontsize=11.5,frameon=False,loc='upper left')
bb=np.linspace(0,36,31); stacked(ax[1],cl['LSS_MPa'],ls['LSS_MPa'],bb)
ax[1].set_xlabel('Lap-shear strength (MPa)'); ax[1].set_ylabel('records'); ax[1].set_title('(b) Lap-shear strength',loc='left',fontweight='bold'); ax[1].legend(fontsize=11.5,frameon=False,loc='upper right')
names=['Adhesive','Cohesive','Mixed','Substrate']; a=ax[2]; x=np.arange(4)
a.bar(x,[(fk.Failure_Mode==i).sum() for i in range(4)],0.7,color=C2,edgecolor='k',lw=0.8,label='Composite-composite')
a.bar(x,[(fc.Failure_Mode==i).sum() for i in range(4)],0.7,color=C1,edgecolor='k',lw=0.8,label='CFRP-CFRP')
a.set_xticks(x); a.set_xticklabels(names); a.set_ylabel('records'); a.set_title('(c) Failure mode',loc='left',fontweight='bold'); a.set_ylim(0,380); a.legend(frameon=False,fontsize=11.5,loc='upper left')
for a in ax.flat: a.tick_params(direction='in',top=True,right=True)
plt.tight_layout(); plt.savefig(FIGD+'fig21_dataset_overview.png',dpi=200); plt.savefig(FIGD+'fig21_dataset_overview.pdf'); plt.close()

# ---- Fig 22: distribution of main continuous inputs (composite-composite load table = superset of CFRP-CFRP)
cols=[('Overlap_Length_mm','Overlap length (mm)'),('Adhesive_Thickness_mm','Adhesive thickness (mm)'),('Adherend1_Thickness_mm','Adherend thickness (mm)'),
      ('Joint_Width_mm','Joint width (mm)'),('Cure_Temp_C','Cure temperature (°C)'),('Test_Temp_C','Test temperature (°C)')]
fig,ax=plt.subplots(2,3,figsize=(15.5,8.2))
for a,(c,lab),t in zip(ax.flat,cols,'abcdef'):
    v2=cc[c].dropna().values; v1=cf[c].dropna().values
    lo,hi=min(v1.min(),v2.min()),max(v1.max(),v2.max()); lg=(c=='Overlap_Length_mm')
    bins=np.logspace(np.log10(lo),np.log10(hi),26) if lg else np.linspace(lo,hi,26)
    stacked(a,v2,v1,bins,lg)
    if lg:
        a.set_xticks([10,20,50,100,200]); a.set_xticklabels(['10','20','50','100','200']); a.minorticks_off()
    a.set_xlabel(lab); a.set_ylabel('records'); a.set_title(f'({t}) {lab.split(" (")[0]}  (n = {len(v2)})',loc='left',fontweight='bold',fontsize=13.5)
    a.tick_params(direction='in',top=True,right=True)
ax[0,0].legend(fontsize=11.5,frameon=False,loc='upper right')
plt.tight_layout(); plt.savefig(FIGD+'fig22_input_distributions.png',dpi=200); plt.savefig(FIGD+'fig22_input_distributions.pdf'); plt.close()

# ---- Fig 23: most used categories (composite-composite table), decoded with the label maps
mp=pd.read_csv(P+'Label_Encoder_Maps.csv')
dec=lambda col: dict(zip(mp[mp.Column==col].Code,mp[mp.Column==col].Category))
surf=[c for c in cc.columns if c.startswith('Surf_') and c not in ('Surf_Unknown','Surf_Ultrasonic_vibration_assisted')]
fig,ax=plt.subplots(1,3,figsize=(15.5,4.8))
def hbar(a,s1,s2,title,top=6):
    s2=s2[s2>0]; order=s2.sort_values(ascending=True).tail(top).index
    y=np.arange(len(order)); a.barh(y,s2[order].values,color=C2,edgecolor='k',lw=0.6,label='Composite-composite'); a.barh(y,s1.reindex(order).fillna(0).values,color=C1,edgecolor='k',lw=0.6,label='CFRP-CFRP')
    a.set_yticks(y); a.set_yticklabels(order,fontsize=12); a.set_xlabel('records'); a.set_title(title,loc='left',fontweight='bold',fontsize=13.5); a.tick_params(direction='in',top=True,right=True)
for a,(col,ttl) in zip(ax[:2],[('Adhesive_Class','(a) Adhesive class'),('Joint_Type_Class','(b) Joint type')]):
    d=dec(col); d={k:v.replace(' (PC/PEI/PES/PSU)','') for k,v in d.items()}; hbar(a,cf[col].map(d).value_counts(),cc[col].map(d).value_counts(),ttl)
s2=pd.Series({c.replace('Surf_','').replace('_',' '):int(cc[c].sum()) for c in surf}); s1=pd.Series({c.replace('Surf_','').replace('_',' '):int(cf[c].sum()) for c in surf})
hbar(ax[2],s1,s2,'(c) Surface treatment')
ax[0].legend(fontsize=11.5,frameon=False,loc='lower right')
plt.tight_layout(); plt.savefig(FIGD+'fig23_category_counts.png',dpi=200); plt.savefig(FIGD+'fig23_category_counts.pdf'); plt.close()
