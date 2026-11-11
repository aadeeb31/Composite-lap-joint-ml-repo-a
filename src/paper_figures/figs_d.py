"""Figures for nested validation, ranking, heavy tail, explanation stability, sensitivity."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *
import pickle
ST = pickle.load(open(str(ROOT / 'results/paper/stats.pkl'), 'rb')); DUM = pickle.load(open(str(ROOT / 'results/paper/dummy.pkl'), 'rb'))
names = [n for n, *_ in DS if n in ST]
def lab(n): return [t for k, _, t, _ in DS if k == n][0]

# fig15: random test vs nested unseen-paper score, per model
fig, axes = plt.subplots(3, 3, figsize=(22, 16.0), gridspec_kw=dict(height_ratios=[1, 1, 0.34])); axes = axes.ravel()
for ax, n in zip(axes, names):
    T = ST[n]['table'].sort_values('Nested_mean'); y = np.arange(len(T)); LO = -1.2 if not n.endswith('FailureMode') else 0.0
    ax.set_xlim(LO - 0.12 if LO < 0 else 0.05, 1.02)
    for i, (_, r) in enumerate(T.iterrows()):
        if r.Nested_mean < LO:
            ax.annotate(f'{r.Nested_mean:.1f} ◄', (LO - 0.1, i), fontsize=11, va='center', ha='left', color=S.FAMC[r.Family], fontweight='bold'); ax.plot([LO - 0.1, r.Random_test], [i, i], color='#999999', lw=1.6, zorder=1); ax.scatter(r.Random_test, i, s=95, marker='o', facecolor='white', edgecolor=S.FAMC[r.Family], linewidth=2.2, zorder=3); continue
        r = r.copy(); r['Nested_min'] = max(r.Nested_min, LO - 0.1)
        c = S.FAMC[r.Family]; ax.plot([r.Nested_mean, r.Random_test], [i, i], color='#999999', lw=1.6, zorder=1)
        ax.plot([r.Nested_min, r.Nested_max], [i, i], color=c, lw=5, alpha=.55, solid_capstyle='butt', zorder=2)
        ax.scatter(r.Random_test, i, s=95, marker='o', facecolor='white', edgecolor=c, linewidth=2.2, zorder=3)
        ax.scatter(r.Nested_mean, i, s=95, marker=S.FAMM[r.Family], facecolor=c, edgecolor='k', linewidth=1.1, zorder=4)
    ax.axvline(max(DUM[n]['score'], LO - 0.1), color='k', ls=':', lw=1.6); ax.set_yticks(y); ax.set_yticklabels(T.Model, fontsize=12.5)
    ax.set_xlabel('Macro-F1' if ST[n]['table'].shape and n.endswith('FailureMode') else 'R²'); ax.set_title(lab(n), fontsize=15); ax.grid(axis='x'); S.panel(ax, 'abcdef'[names.index(n)])
axes[6].axis('off'); axes[7].axis('off'); axes[8].axis('off')
h = [Line2D([], [], marker='o', color='w', markerfacecolor='white', markeredgecolor='#444', markeredgewidth=2.2, markersize=11, label='Random 80:20 test'),
     Line2D([], [], marker='s', color='w', markerfacecolor='#777', markeredgecolor='k', markersize=11, label='Unseen-study (nested), mean of repeats'),
     Line2D([], [], color='#777', lw=5, alpha=.55, label='Range over repeats'), Line2D([], [], color='k', ls=':', lw=1.6, label='Training-mean / majority baseline')]
axes[6].legend(handles=h, loc='center', fontsize=14, title='Markers', title_fontsize=14.5)
family_legend(axes[7], loc='center', ncol=2, title='Family', fs=13.5)
fig.tight_layout(); save(fig, 'fig15_random_vs_nested_all_models')

# fig16: family mean ranks (best model of each family per paper) with Nemenyi-style annotation
fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 6.2), sharex=False)
for ax, n in zip(np.atleast_1d(axes), names):
    fr = ST[n]['fam_meanrank'].sort_values(ascending=False)
    ax.barh(range(len(fr)), fr.values, color=[S.FAMC[f] for f in fr.index], edgecolor='k', height=.72)
    ax.set_yticks(range(len(fr))); ax.set_yticklabels(fr.index, fontsize=12); ax.set_xlabel('Mean rank over papers (1 = best)')
    ax.set_title(lab(n).replace(' · ', '\n'), fontsize=13); ax.grid(axis='x'); ax.set_xlim(0, 8.2)
    ax.text(.98, .03, f"Friedman p={ST[n]['fam_friedman'][1]:.1e}", transform=ax.transAxes, ha='right', fontsize=11)
fig.tight_layout(); save(fig, 'fig16_family_mean_rank')

# fig17: model mean ranks with Nemenyi critical-difference band from the best model
fig, axes = plt.subplots(3, 3, figsize=(22, 18.2), gridspec_kw=dict(height_ratios=[1, 1, 0.62])); axes = axes.ravel()
for ax, n in zip(axes, names):
    rk = ST[n]['ranks'].sort_values(); cd = ST[n]['cd']; fam = lambda m: S.MODEL_FAMILY[m]
    ax.axvspan(rk.iloc[0], rk.iloc[0] + cd, color='#FFE08A', alpha=.45, zorder=0)
    for i, (m, v) in enumerate(rk.items()):
        ax.scatter(v, i, s=120, marker=S.FAMM[fam(m)], facecolor=S.FAMC[fam(m)], edgecolor='k', linewidth=1.1, zorder=3, alpha=1 if v < rk.iloc[0] + cd else .55)
    ax.set_yticks(range(len(rk))); ax.set_yticklabels(rk.index, fontsize=12.5); ax.invert_yaxis(); ax.set_xlabel('Mean rank over papers (1 = best)')
    ax.set_title(f"{lab(n)}\nCD = {cd:.2f}, {ST[n]['indist_frac']*100:.0f}% of pairs within CD", fontsize=13.5); ax.grid(axis='x'); S.panel(ax, 'abcdef'[names.index(n)])
axes[6].axis('off'); axes[8].axis('off'); ax = axes[7]; ax.axis('off'); family_legend(ax, loc='upper left', ncol=2, title='Family', fs=13)
ax.text(.02, .35, 'Shaded band: models whose mean rank lies within the Nemenyi\ncritical difference of the best-ranked model\n(not statistically separable from it at 5%).', fontsize=13, transform=ax.transAxes, va='top')
fig.tight_layout(); save(fig, 'fig17_nemenyi_rank_bands')

# fig18: heavy-tail effect on the two load datasets
ln = [n for n in names if n.endswith('MaxLoad')]
if ln:
    fig, axes = plt.subplots(2, len(ln), figsize=(11.5 * len(ln), 12.5)); axes = np.atleast_2d(axes).reshape(2, -1)
    for j, n in enumerate(ln):
        H = ST[n]['heavy'].copy(); H['Family'] = H.Model.map(S.MODEL_FAMILY); H = H.sort_values('MAE_le25'); ax = axes[0, j]
        ax.barh(range(len(H)), H.MAE_le25, color=[S.FAMC[f] for f in H.Family], edgecolor='k', height=.75); ax.invert_yaxis()
        ax.axvline(DUM[n]['mae_le25_median'], color='k', ls='--', lw=2); ax.text(DUM[n]['mae_le25_median'], -0.9, ' training-median baseline', fontsize=11.5, va='bottom'); ax.set_ylim(len(H) - .4, -1.6)
        ax.set_yticks(range(len(H))); ax.set_yticklabels(H.Model, fontsize=11.5); ax.set_xlabel('MAE on joints ≤ 25 kN (kN), unseen papers'); ax.set_title(lab(n), fontsize=15); ax.grid(axis='x'); S.panel(ax, 'ab'[j])
        ax = axes[1, j]; H = H.sort_values('R2_all', ascending=False)
        LO2 = -6; yy = np.arange(len(H)); H = H.assign(R2_le25_c=H.R2_le25.clip(lower=LO2)); ax.set_xlim(LO2 - .5, 1.2)
        for i in yy:
            if H.R2_le25.iloc[i] < LO2: ax.annotate(f'{H.R2_le25.iloc[i]:.0f} ◄', (LO2 + .1, i), fontsize=10.5, va='center', color='#333')
        ax.scatter(H.R2_all, yy, s=90, c=[S.FAMC[f] for f in H.Family], edgecolor='k', zorder=3, label='All rows')
        ax.scatter(H.R2_le25_c, yy, s=90, marker='X', c=[S.FAMC[f] for f in H.Family], edgecolor='k', zorder=3, label='Rows ≤ 25 kN')
        for i in yy: ax.plot([H.R2_le25_c.iloc[i], H.R2_all.iloc[i]], [i, i], color='#999', lw=1.2, zorder=1)
        ax.axvline(0, color='k', lw=1); ax.set_yticks(yy); ax.set_yticklabels(H.Model, fontsize=11.5); ax.invert_yaxis(); ax.set_xlabel('R² under nested unseen-paper validation'); ax.grid(axis='x')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=12); S.panel(ax, 'cd'[j])
    fig.tight_layout(); save(fig, 'fig18_heavy_tail')

# fig19: explanation stability across unseen-paper folds
fig, axes = plt.subplots(2, 3, figsize=(16.5, 12.5), sharex=True); axes = axes.ravel()
for ax, n in zip(axes, names):
    s = ST[n]['stab'].copy(); s['Family'] = [S.MODEL_FAMILY[m] for m in s.index]; s = s.sort_values('fold_rho')
    ax.barh(range(len(s)), s.fold_rho, color=[S.FAMC[f] for f in s.Family], edgecolor='k', height=.75); ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=11)
    ax.set_xlabel('Mean Spearman ρ between folds'); ax.set_title(lab(n).replace(' · ', '\n'), fontsize=14.5); ax.grid(axis='x'); ax.axvline(0, color='k', lw=1); S.panel(ax, 'abcdef'[names.index(n)], dx=-0.02)
fig.tight_layout(); save(fig, 'fig19_explanation_stability')

# fig20: sensitivity to row choice
import os
if os.path.exists(str(ROOT / 'results/paper/sensitivity.csv')):
    SE = pd.read_csv(str(ROOT / 'results/paper/sensitivity.csv')); SE['Family'] = SE.Model.map(S.MODEL_FAMILY)
    fig, axes = plt.subplots(3, 3, figsize=(22, 16.9), gridspec_kw=dict(height_ratios=[1, 1, 0.62])); axes = axes.ravel()
    for ax, n in zip(axes, names):
        d = SE[SE.File == n]
        if d.empty: ax.axis('off'); continue
        subs = list(dict.fromkeys(d.Subset)); fm = d.groupby(['Family', 'Subset']).Paperwise.max().unstack()[subs]; w = .8 / len(subs)
        cols = ['#1F3A5F', '#4A6FA5', '#9DB7D9', '#C9D6EA']
        for k, sname in enumerate(subs): ax.bar(np.arange(len(fm)) + k * w, fm[sname].values, w, color=cols[k], edgecolor='k', label=sname)
        ax.set_xticks(np.arange(len(fm)) + .4 - w / 2); ax.set_xticklabels([f.replace('Nearest neighbour', 'k-NN').replace('Neural net', 'MLP') for f in fm.index], rotation=35, ha='right', fontsize=11.5)
        ax.set_ylabel('Best paper-wise score in family'); ax.set_title(lab(n), fontsize=14); ax.grid(axis='y'); ax.axhline(0, color='k', lw=1); S.panel(ax, 'abcdef'[names.index(n)])
    axes[0].legend(fontsize=11, loc='lower left'); axes[6].axis('off'); axes[8].axis('off'); axes[7].axis('off'); fig.tight_layout(); save(fig, 'fig20_row_choice_sensitivity')
