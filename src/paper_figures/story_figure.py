import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pickle
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle, FancyArrow
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
names={f.name for f in fm.fontManager.ttflist}
plt.rcParams['font.family']='Liberation Sans' if 'Liberation Sans' in names else 'DejaVu Sans'
NAVY='#1F3864'; STEEL='#4A6FA5'; LIGHT='#E6EDF6'; INK='#1B1F27'; ACC='#C8642B'; GREY='#9AA5B5'; MID='#B9C7DC'
W,H=100,82
fig=plt.figure(figsize=(7.2,5.9),dpi=300); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off'); ax.set_aspect('auto')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; V=str(ROOT/'results/paper')
def panel(x,y,w,h,num,title,hc=NAVY):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.2',fc='white',ec=hc,lw=1.0))
    ax.add_patch(FancyBboxPatch((x,y+h-4.4),w,4.4,boxstyle='round,pad=0,rounding_size=1.2',fc=hc,ec=hc,lw=1.0))
    ax.add_patch(Rectangle((x,y+h-4.4),w,2.0,fc=hc,ec=hc,lw=0))
    ax.add_patch(Circle((x+2.9,y+h-2.2),1.55,fc='white',ec='white'))
    ax.text(x+2.9,y+h-2.2,str(num),ha='center',va='center',fontsize=8.16,fontweight='bold',color=hc)
    ax.text(x+5.5,y+h-2.2,title,ha='left',va='center',fontsize=8.88,fontweight='bold',color='white')
def arr(x1,y1,x2,y2,c=STEEL,lw=1.6,ms=10): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=ms,lw=lw,color=c,shrinkA=0,shrinkB=0))
def inset(x,y,w,h):
    a=fig.add_axes([x/W,y/H,w/W,h/H]); a.tick_params(labelsize=5,length=2,width=0.5,pad=1.5); 
    for s in a.spines.values(): s.set_linewidth(0.6)
    a.patch.set_alpha(0); return a
bw=31.4; gap=2.0; x0=1.0; xs=[x0,x0+bw+gap,x0+2*(bw+gap)]
ph=38.6; y1=H-1.0-ph; y2=y1-2.6-ph
# ---------- panel 1 : joint + database
panel(xs[0],y1,bw,ph,1,'Joints and database')
X=xs[0]; Y=y1
# single-lap joint drawing
ax.add_patch(Rectangle((X+3,Y+26.2),17,3.0,fc=MID,ec=NAVY,lw=0.6,hatch='////'))   # upper adherend left part
ax.add_patch(Rectangle((X+12,Y+23.2),17,3.0,fc=MID,ec=NAVY,lw=0.6,hatch='////'))  # lower adherend right part
ax.add_patch(Rectangle((X+12,Y+26.2),8,1.0,fc=ACC,ec=ACC,lw=0))                   # adhesive layer
ax.add_patch(Rectangle((X+12,Y+25.2),8,1.0,fc='#F2B48A',ec='none'))
ax.add_patch(Rectangle((X+12,Y+26.2),8,0.0,fc=ACC))
arr(X+3,Y+27.7,X+0.2,Y+27.7,c=INK,lw=1.1,ms=7); arr(X+29,Y+24.7,X+31.2,Y+24.7,c=INK,lw=1.1,ms=7)
ax.annotate('',xy=(X+12,Y+30.8),xytext=(X+20,Y+30.8),arrowprops=dict(arrowstyle='<->',lw=0.7,color=INK,shrinkA=0,shrinkB=0))
ax.plot([X+12,X+12],[Y+29.3,Y+31.1],color=INK,lw=0.5); ax.plot([X+20,X+20],[Y+29.3,Y+31.1],color=INK,lw=0.5)
ax.text(X+16,Y+31.5,'overlap length',ha='center',va='bottom',fontsize=6.48,color=INK)
ax.text(X+21.8,Y+28.6,'adhesive layer',ha='left',va='center',fontsize=6.48,color=ACC,fontweight='bold')
ax.plot([X+20.1,X+21.5],[Y+26.7,Y+28.3],color=ACC,lw=0.5)
ax.text(X+6,Y+24.3,'composite\nadherends',ha='center',va='top',fontsize=6.48,color=NAVY)
# three outputs
ax.text(X+bw/2,Y+20.3,'Outputs: load, strength, failure mode',ha='center',va='center',fontsize=6.72,color=INK,fontweight='bold')
# failure mode icons
def fm_icon(cx,cy,kind,label):
    ax.add_patch(Rectangle((cx-3.1,cy+0.2),6.2,1.7,fc=MID,ec=NAVY,lw=0.4))
    ax.add_patch(Rectangle((cx-3.1,cy-1.9),6.2,1.7,fc=MID,ec=NAVY,lw=0.4))
    ax.add_patch(Rectangle((cx-3.1,cy-0.2),6.2,0.4,fc='#F2B48A',ec='none'))
    if kind=='adh': ax.plot([cx-3.1,cx+3.1],[cy+0.25,cy+0.25],color='k',lw=1.0,ls=(0,(1.5,1)))
    if kind=='coh': ax.plot([cx-3.1,cx+3.1],[cy,cy],color='k',lw=1.0,ls=(0,(1.5,1)))
    if kind=='mix': ax.plot([cx-3.1,cx-0.2,cx+0.2,cx+3.1],[cy+0.25,cy+0.25,cy-0.2,cy-0.2],color='k',lw=1.0,ls=(0,(1.5,1)))
    if kind=='sub': ax.plot([cx-3.1,cx-1,cx+0.5,cx+3.1],[cy+0.3,cy+0.9,cy+1.5,cy+1.0],color='k',lw=1.0,ls=(0,(1.5,1)))
    ax.text(cx,cy-3.0,label,ha='center',va='top',fontsize=6.00,color=INK)
for i,(k,l) in enumerate([('adh','adhesive'),('coh','cohesive'),('mix','mixed'),('sub','substrate')]): fm_icon(X+4.7+i*7.5,Y+15.5,k,l)
# literature -> records
ax.plot([X+1.5,X+bw-1.5],[Y+10.6,Y+10.6],color=LIGHT,lw=0.8)
for i in range(4):
    ax.add_patch(Rectangle((X+3.0+i*0.9,Y+3.3+i*0.6),4.3,5.3,fc='white',ec=NAVY,lw=0.6))
    for j in range(3): ax.plot([X+3.8+i*0.9,X+6.5+i*0.9],[Y+7.4+i*0.6-j*1.1,Y+7.4+i*0.6-j*1.1],color=GREY,lw=0.5)
ax.text(X+6.4,Y+2.1,'74 studies',ha='center',va='top',fontsize=6.72,color=INK,fontweight='bold')
arr(X+10.4,Y+6.5,X+16.0,Y+6.5,lw=1.2,ms=8)
# grid icon
rng=np.random.default_rng(1)
for r in range(6):
    for c in range(8):
        ax.add_patch(Rectangle((X+17.2+c*1.45,Y+3.4+r*1.0),1.3,0.85,fc=STEEL if (r+c)%3 else '#8EA9CF',ec='none'))
ax.text(X+22.9,Y+2.1,'1,516 records',ha='center',va='top',fontsize=6.72,color=INK,fontweight='bold')
# ---------- panel 2 : models
panel(xs[1],y1,bw,ph,2,'22 models, 8 families')
X=xs[1]; Y=y1
cw=bw/4; 
def glyph(i,j,kind,label):
    cx=X+cw*(i+0.5); cy=Y+ph-13.0-j*14.6
    ax.add_patch(FancyBboxPatch((cx-3.5,cy-3.4),7.0,6.8,boxstyle='round,pad=0,rounding_size=0.7',fc=LIGHT,ec='none'))
    t=np.linspace(-3,3,50)
    if kind=='lin': ax.plot([cx-3,cx+3],[cy-2.4,cy+2.4],color=NAVY,lw=1.3); [ax.plot(cx-2.6+k*1.3,cy-2.0+k*0.95+(0.5 if k%2 else -0.6),'o',ms=1.4,color=ACC) for k in range(5)]
    if kind=='ker': ax.plot(cx+t,cy+1.8*np.sin(t*1.1),color=NAVY,lw=1.3); [ax.plot(cx+u,cy+1.8*np.sin(u*1.1)+0.5*(-1)**k,'o',ms=1.4,color=ACC) for k,u in enumerate([-2.5,-1.4,-0.2,1.0,2.2])]
    if kind=='nn':
        pts=[[(-2.6,-1.4),(-2.6,0),(-2.6,1.4)],[(-0.9,-2),(-0.9,-0.7),(-0.9,0.7),(-0.9,2)],[(0.8,-1.4),(0.8,0),(0.8,1.4)],[(2.5,0)]]
        for L in range(3):
            for a in pts[L]:
                for b in pts[L+1]: ax.plot([cx+a[0],cx+b[0]],[cy+a[1],cy+b[1]],color=GREY,lw=0.4)
        for L in pts:
            for a in L: ax.add_patch(Circle((cx+a[0],cy+a[1]),0.55,fc=NAVY,ec='none'))
    def tree(cx,cy,s=1.0,c=NAVY):
        for (a,b,c2,d) in [(0,2.4,-1.6,0.6),(0,2.4,1.6,0.6),(-1.6,0.6,-2.4,-1.4),(-1.6,0.6,-0.8,-1.4),(1.6,0.6,0.8,-1.4),(1.6,0.6,2.4,-1.4)]:
            ax.plot([cx+a*s,cx+c2*s],[cy+b*s-0.2,cy+d*s-0.2],color=c,lw=0.9)
        for (px,py) in [(0,2.4),(-1.6,0.6),(1.6,0.6)]: ax.add_patch(Circle((cx+px*s,cy+py*s-0.2),0.5*s,fc=c,ec='none'))
        for px in (-2.4,-0.8,0.8,2.4): ax.add_patch(Rectangle((cx+px*s-0.4*s,cy-1.4*s-0.6),0.8*s,0.8*s,fc=ACC,ec='none'))
    if kind=='tree': tree(cx,cy)
    if kind=='for': tree(cx-1.9,cy+0.6,0.55); tree(cx+1.9,cy+0.6,0.55); tree(cx,cy-1.0,0.55,STEEL)
    if kind=='boost':
        for k in range(4): ax.add_patch(Rectangle((cx-2.8+k*1.45,cy-2.6),1.25,0.9+k*1.05,fc=NAVY if k<3 else ACC,ec='none'))
        ax.plot([cx-3,cx+3],[cy-2.7,cy-2.7],color=INK,lw=0.4)
    if kind=='knn':
        rg=np.random.default_rng(3)
        for k in range(9): ax.plot(cx+rg.uniform(-2.8,2.8),cy+rg.uniform(-2.4,2.4),'o',ms=1.6,color=GREY)
        ax.add_patch(Circle((cx+0.2,cy+0.1),1.1,fc=ACC,ec='none')); ax.add_patch(Circle((cx+0.2,cy+0.1),2.2,fc='none',ec=NAVY,lw=0.7,ls='--'))
    if kind=='gp':
        ax.fill_between(cx+t,cy+1.6*np.sin(t*0.9)-0.9-0.25*np.abs(t),cy+1.6*np.sin(t*0.9)+0.9+0.25*np.abs(t),color=MID,lw=0)
        ax.plot(cx+t,cy+1.6*np.sin(t*0.9),color=NAVY,lw=1.2)
    ax.text(cx,cy-4.1,label,ha='center',va='top',fontsize=6.00,color=INK)
for idx,(k,l) in enumerate([('lin','Linear'),('ker','Kernel'),('nn','Neural net'),('tree','Tree'),('for','Forest'),('boost','Boosting'),('knn','k-NN'),('gp','Bayesian')]):
    glyph(idx%4,idx//4,k,l)
# ---------- panel 3 : predict (real parity)
panel(xs[2],y1,bw,ph,3,'Predict',hc=STEEL)
P=pickle.load(open(V+'/preds.pkl','rb'))[('CFRP_CFRP__MaxLoad','LGBM')]
a=inset(xs[2]+6.8,y1+6.4,bw-9.0,ph-14.4)
a.scatter(P['ytr'],P['ptr'],s=2.2,c='#B9C7DC',lw=0,label='train')
a.scatter(P['yte'],P['pte'],s=4.2,c=ACC,lw=0,label='test')
lo,hi=0.3,250; a.plot([lo,hi],[lo,hi],color=INK,lw=0.7); a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo,hi); a.set_ylim(lo,hi)
a.set_xlabel('Measured load (kN)',fontsize=6.48,labelpad=1); a.set_ylabel('Predicted load (kN)',fontsize=6.48,labelpad=1)
a.legend(fontsize=5.76,frameon=False,loc='upper left',handletextpad=0.2,borderpad=0.1,markerscale=1.6)
a.text(0.97,0.06,'R² = 0.961',transform=a.transAxes,ha='right',fontsize=7.20,fontweight='bold',color=NAVY)
ax.text(xs[2]+bw/2,y1+2.2,'CFRP–CFRP maximum load, LightGBM, test split',ha='center',va='center',fontsize=6.00,color=INK)
# ---------- panel 4 : SHAP beeswarm
panel(xs[0],y2,bw,ph,4,'Explain: which inputs matter',hc=STEEL)
D=pickle.load(open(V+'/ana/CFRP_CFRP__LSS.pkl','rb')); sv=np.asarray(D['shap']['values']); Xs=np.asarray(D['shap']['Xs']); feats=list(D['features'])
order=np.argsort(-np.abs(sv).mean(0))[:6]
lab={'Adhesive_Thickness_mm':'Adhesive thickness','Overlap_Length_mm':'Overlap length','Aging_Days':'Ageing time','Fillet':'Fillet','Cure_Temp_C':'Cure temperature','Filler1_wtpct':'Filler content','Adherend1_Thickness_mm':'Adherend thickness','Surf_Plasma_corona':'Plasma treatment','PostCure_Temp_C':'Post-cure temp.','Adherend2_Thickness_mm':'Adherend 2 thickness'}
a=inset(xs[0]+13.0,y2+6.0,bw-14.5,ph-13.2)
cm=LinearSegmentedColormap.from_list('bl',['#2F6BD8','#8E7CC3','#E0344D'])
rg=np.random.default_rng(7)
for r,j in enumerate(order):
    x=sv[:,j]; c=Xs[:,j]; c=(c-np.nanmin(c))/(np.nanmax(c)-np.nanmin(c)+1e-9)
    yy=r+rg.normal(0,0.13,len(x)); a.scatter(x,-yy,s=1.6,c=c,cmap=cm,lw=0,vmin=0,vmax=1)
a.set_yticks(-np.arange(6)); a.set_yticklabels([lab.get(feats[j],feats[j].replace('_',' ')) for j in order],fontsize=6.24)
a.axvline(0,color=GREY,lw=0.5); a.set_xlabel('Effect on predicted strength (MPa)',fontsize=6.24,labelpad=1); a.set_ylim(-5.7,0.7)
ax.text(xs[0]+bw/2,y2+2.2,'CFRP–CFRP strength; blue low, red high value',ha='center',va='center',fontsize=6.00,color=INK)
# ---------- panel 5 : PDP
panel(xs[1],y2,bw,ph,5,'Explain: how inputs act',hc=STEEL)
pd_=D['pdp']['data']['Adhesive_Thickness_mm']; g=np.asarray(pd_['grid']); cur=pd_['curves']
a=inset(xs[1]+6.8,y2+6.4,bw-9.0,ph-14.4)
M=np.array([np.asarray(v[1]) for k,v in cur.items() if k!='Linear'])
a.fill_between(g,M.min(0),M.max(0),color=MID,alpha=0.8,lw=0)
for k,v in cur.items():
    if k=='Linear': continue
    a.plot(g,np.asarray(v[1]),color=STEEL,lw=0.6,alpha=0.9)
a.set_xscale('log'); a.set_xlabel('Adhesive thickness (mm)',fontsize=6.48,labelpad=1); a.set_ylabel('Predicted strength (MPa)',fontsize=6.48,labelpad=1)
a.text(0.97,0.95,'thin bondline →\nhigher strength',transform=a.transAxes,ha='right',va='top',fontsize=6.48,color=ACC,fontweight='bold')
ax.text(xs[1]+bw/2,y2+2.2,'partial dependence, one line per model family',ha='center',va='center',fontsize=6.00,color=INK)
# ---------- panel 6 : design messages
panel(xs[2],y2,bw,ph,6,'Use: design insight',hc=ACC)
X=xs[2]; Y=y2
def row(cy,title,sub): 
    ax.text(X+12.2,cy+1.2,title,ha='left',va='center',fontsize=7.56,fontweight='bold',color=INK); ax.text(X+12.2,cy-1.8,sub,ha='left',va='center',fontsize=6.36,color=INK,linespacing=1.25)
# icon 1 : bondline thin vs thick
def lay(x,y,w,t):
    ax.add_patch(Rectangle((x,y+t/2),w,1.2,fc=MID,ec=NAVY,lw=0.4)); ax.add_patch(Rectangle((x,y-t/2-1.2),w,1.2,fc=MID,ec=NAVY,lw=0.4)); ax.add_patch(Rectangle((x,y-t/2),w,t,fc='#F2B48A',ec=ACC,lw=0.3))
cy=Y+ph-11.5
lay(X+2.2,cy,4.2,0.5); lay(X+7.0,cy,4.2,2.2)
row(cy,'Thin bondlines','predicted strength about\nhalves from 0.15 to 2 mm')
# icon 2 : overlap
cy=Y+ph-22.2
ax.add_patch(Rectangle((X+2.2,cy+0.4),7.0,1.2,fc=MID,ec=NAVY,lw=0.4)); ax.add_patch(Rectangle((X+5.0,cy-0.8),7.0,1.2,fc=MID,ec=NAVY,lw=0.4)); ax.add_patch(Rectangle((X+5.0,cy+0.4),4.2,0.3,fc=ACC,ec='none'))
ax.annotate('',xy=(X+5.0,cy+3.2),xytext=(X+9.2,cy+3.2),arrowprops=dict(arrowstyle='<->',lw=0.7,color=INK,shrinkA=0,shrinkB=0))
row(cy,'Longer overlap','predicted load rises, but\nless than in proportion')
# icon 3 : ageing
cy=Y+ph-32.9
t=np.linspace(0,1,30)
ax.plot(X+2.8+t*7,cy+2.2-3.4*t**0.9,color=ACC,lw=1.4); ax.plot([X+2.5,X+2.5],[cy-1.6,cy+3],color=INK,lw=0.5); ax.plot([X+2.5,X+10.2],[cy-1.6,cy-1.6],color=INK,lw=0.5)
ax.text(X+6.3,cy-2.6,'time',ha='center',va='top',fontsize=5.52,color=INK)
row(cy,'Ageing','about 20% lower predicted\nstrength after 60 days')
# ---------- arrows between panels
for i in (0,1):
    arr(xs[i]+bw+0.1,y1+ph/2,xs[i+1]-0.1,y1+ph/2)
    arr(xs[i]+bw+0.1,y2+ph/2,xs[i+1]-0.1,y2+ph/2)
# connector from panel 3 down to 4 (elbow)
cx=xs[2]+bw/2
ax.plot([cx,cx],[y1,y1-1.3],color=STEEL,lw=1.6,solid_capstyle='butt'); ax.plot([cx,xs[0]+bw/2],[y1-1.3,y1-1.3],color=STEEL,lw=1.6)
arr(xs[0]+bw/2,y1-1.3,xs[0]+bw/2,y2+ph+0.1)
fig.savefig(str(ROOT/'figures/fig00_story.png'),dpi=300); print('ok')
