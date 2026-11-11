from figlib import *
from matplotlib.colors import Normalize


def fig01_leaderboard():
    import pickle
    ST = pickle.load(open(str(ROOT / 'results/paper/stats.pkl'), 'rb'))
    NEST = {n: ST[n]['table'].set_index('Model').Nested_mean for n in ST}
    names = [n for n, *_ in DS if (RES.File == n).any() and n in ST]
    mods = ordered_models(RES[RES.File == names[0]]) if False else None
    # rows: union of models in family order (regression-only and classification-only models both appear; blanks stay blank)
    allm = RES[['Model', 'Family']].drop_duplicates(); allm['fi'] = allm.Family.map(FIDX)
    sc = RES.groupby('Model').Score.mean(); allm['m'] = allm.Model.map(sc); allm = allm.sort_values(['fi', 'm'], ascending=[True, False])
    rows = list(allm.Model); fams = list(allm.Family)
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 8.0), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, col, ttl, vmin in [(axs[0], 'Score', 'Random 80:20 test score (paper protocol)', 0.0), (axs[1], 'OOF', 'Unseen-study score (nested, grouped by study)', 0.0)]:
        ax.grid(False)
        for j, n in enumerate(names):
            for i, m in enumerate(rows):
                r = RES[(RES.File == n) & (RES.Model == m)]
                if r.empty: ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc='#F2F2F2', ec='white', lw=1.6, hatch='////')); continue
                v = float(r[col].iloc[0]) if col == 'Score' else float(NEST[n].get(m, np.nan))
                if np.isnan(v): ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc='#F2F2F2', ec='white', lw=1.6, hatch='////')); continue
                c = S.SCORE(np.clip((v - vmin) / (1 - vmin), 0, 1))
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=c, ec='white', lw=1.6))
                ax.text(j, i, f'{v:.2f}' if v > -9 else '<-9', ha='center', va='center', fontsize=12.5, fontweight='bold', color='white' if (v - vmin) / (1 - vmin) < .18 or (v - vmin) / (1 - vmin) > .86 else INK_ if False else 'k')
            best = RES[RES.File == n].sort_values(col).iloc[-1].Model if col == 'Score' else NEST[n].idxmax()
            ax.add_patch(Rectangle((j - .5, rows.index(best) - .5), 1, 1, fc='none', ec='k', lw=2.6))
        ax.set_xlim(-.5, len(names) - .5); ax.set_ylim(len(rows) - .5, -.5)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([SHORT[n].replace('Composite-composite', 'Composite-\ncomposite').replace('CFRP-CFRP', 'CFRP-\nCFRP').replace('\nmax load', '\nload').replace('\nfailure mode', '\nfailure\nmode') for n in names], fontsize=11.5, rotation=0, ha='center', va='bottom')
        ax.xaxis.tick_top(); ax.xaxis.set_label_position('top'); ax.set_title(ttl, pad=62, fontsize=15)
        for s in ax.spines.values(): s.set_linewidth(1.6)
        ax.tick_params(length=0)
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels(rows, fontsize=13.7, fontweight='bold'); axs[0].tick_params(axis='y', pad=22)
    axs[1].yaxis.set_visible(False)
    # family colour band
    i0 = 0
    for f in S.FAM:
        idx = [i for i, x in enumerate(fams) if x == f]
        if not idx: continue
        axs[0].add_patch(Rectangle((-0.5 - 0.14, idx[0] - .5), 0.14, len(idx), fc=S.FAMC[f], ec='k', lw=1.2, clip_on=False))
        axs[1].add_patch(Rectangle((len(names) - 0.5, idx[0] - .5), 0.14, len(idx), fc=S.FAMC[f], ec='k', lw=1.2, clip_on=False))
        axs[1].annotate(f, xy=(1, np.mean(idx)), xycoords=('axes fraction', 'data'), xytext=(34, 0), textcoords='offset points', ha='left', va='center', fontsize=14.3, fontweight='bold')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=S.SCORE), ax=axs, orientation='horizontal', fraction=0.035, pad=0.045, aspect=45)
    cb.set_label('score: R² for load / LSS, macro-F1 for failure mode (same colour scale in both panels)', fontsize=13.0); cb.outline.set_linewidth(1.2)
    fig.text(0.5, 0.005, 'Black frame = best model in the column. Hatched = model not run for that task type (regressors only for load/LSS, classifiers only for failure mode).', ha='center', fontsize=12.1, style='italic')
    save(fig, 'fig01_leaderboard_random_vs_unseen')


def fig02_gap():
    fig, axs = plt.subplots(3, 3, figsize=(22, 20.2), gridspec_kw=dict(height_ratios=[1, 1, 0.62])); axs = axs.ravel(); fig.subplots_adjust(hspace=0.30, wspace=0.26)
    for ax, (n, rel, ttl, kind) in zip(axs, DS):
        r = ordered_models(RES[RES.File == n]).sort_values('Score')
        y = np.arange(len(r)); lo = -0.35
        for yi, (_, q) in zip(y, r.iterrows()):
            c = S.FAMC[q.Family]; xs = [max(q.OOF, lo), q.Rand, q.Score]
            ax.plot([min(xs), max(xs)], [yi, yi], color='#8c8c8c', lw=2.2, zorder=1, solid_capstyle='round')
            ax.scatter(max(q.OOF, lo), yi, marker='D', s=105, fc='white', ec=c, lw=2.5, zorder=3)
            ax.scatter(q.Rand, yi, marker='s', s=82, fc=c, ec='k', lw=1.1, zorder=3, alpha=.75)
            ax.scatter(q.Score, yi, marker='o', s=125, fc=c, ec='k', lw=1.6, zorder=4)
            if q.OOF < lo: ax.annotate('◄', (lo, yi), ha='center', va='center', fontsize=11.7, color=c, zorder=5)
        ax.set_yticks(y); ax.set_yticklabels(r.Model, fontsize=12.5); ax.set_xlim(lo - 0.03, 1.03)
        for t, f in zip(ax.get_yticklabels(), r.Family): t.set_color(S.FAMC[f] if f not in ('Forest', 'Kernel') else '#222'); t.set_fontweight('bold')
        ax.set_xlabel('R²' if kind == 'r2' else 'macro-F1'); ax.set_title(ttl.replace(' · ', '\n'), fontsize=15.0)
        S.panel(ax, 'abcdef'[[d[0] for d in DS].index(n)], dx=-0.14)
    for a_ in axs[6:]: a_.axis('off')
    h = [Line2D([], [], marker='D', color='w', mfc='white', mec='#444', mew=2.5, ms=11, label='Unseen-paper (grouped 5-fold)'),
         Line2D([], [], marker='s', color='w', mfc='#999', mec='k', ms=10, label='Random 5-fold (training part)'),
         Line2D([], [], marker='o', color='w', mfc='#999', mec='k', ms=12, label='Random 80:20 test (paper protocol)')]
    axs[6].legend(handles=h, loc='center', title='What each marker measures', fontsize=13.7, title_fontsize=14.3)
    hf = [Patch(fc=S.FAMC[f], ec='k', label=f) for f in S.FAM]
    axs[7].legend(handles=hf, loc='center', ncol=2, title='Model family (label colour)', fontsize=13.7, title_fontsize=14.3)
    axs[8].text(0.5, 0.5, 'The length of each bar is the\noptimism of a random split.\n◄ = unseen-paper score below -0.35', ha='center', va='center', transform=axs[8].transAxes, fontsize=13.7, style='italic')
    save(fig, 'fig02_optimism_gap_dumbbell')


def fig03_bubble():
    fig, axs = plt.subplots(3, 3, figsize=(22, 20.2), gridspec_kw=dict(height_ratios=[1, 1, 0.62])); axs = axs.ravel(); fig.subplots_adjust(hspace=0.30, wspace=0.26)
    for ax, (n, rel, ttl, kind) in zip(axs, DS):
        r = RES[RES.File == n]; lo = 0.0
        TX = []
        for _, q in r.iterrows():
            s = 40 + 330 * np.clip(q.OOF, 0, 1) ** 1.2
            ax.scatter(max(q.Fit_s, 0.003), q.Score, s=s, marker=S.FAMM[q.Family], fc=S.FAMC[q.Family], ec='k', lw=1.4, alpha=.9, zorder=3)
            TX.append(ax.text(max(q.Fit_s, .003), q.Score, q.Model, fontsize=13, fontweight='bold', zorder=6))
        ax.set_xscale('log'); ax.set_xlabel('Training time, one fit (s, log scale)'); ax.set_ylabel('R² (random 80:20 test)' if kind == 'r2' else 'macro-F1 (random 80:20 test)')
        ax.set_title(ttl.replace(' · ', '\n'), fontsize=15.0); S.panel(ax, 'abcdef'[[d[0] for d in DS].index(n)], dx=-0.14)
        ax.set_ylim(max(0.35, r.Score.min() - 0.09), min(1.01, r.Score.max() + 0.05)); ax.margins(x=0.2)
        from adjustText import adjust_text; adjust_text(TX, ax=ax, expand=(1.6, 2.2), force_text=(0.6, 0.9), force_static=(0.4, 0.6), max_move=(60, 60), iter_lim=600, arrowprops=dict(arrowstyle='-', color='#555', lw=.9))
    axs[6].axis('off'); axs[8].axis('off'); ax = axs[7]; ax.axis('off'); family_legend(axs[6], loc='center', ncol=2, title='Model family')
    for v, y in [(0, 0.38), (0.4, 0.27), (0.8, 0.14)]:
        ax.scatter([0.30], [y], s=40 + 330 * v ** 1.2, marker='o', fc='#DDD', ec='k', lw=1.4, transform=ax.transAxes); ax.text(0.44, y, f'unseen-paper score {v:.1f}', transform=ax.transAxes, va='center', fontsize=13.7)
    ax.text(0.5, 0.52, 'Bubble area = unseen-paper score\n(grouped 5-fold, pooled)', ha='center', transform=ax.transAxes, fontsize=13.7, style='italic')
    save(fig, 'fig03_accuracy_vs_cost_bubble')


def fig04_arena():
    fig, axs = plt.subplots(3, 3, figsize=(22, 22.1), gridspec_kw=dict(height_ratios=[1, 1, 0.62])); axs = axs.ravel(); fig.subplots_adjust(hspace=0.22, wspace=0.42)
    for ax, (n, rel, ttl, kind) in zip(axs, DS):
        r = RES[RES.File == n]; y = 0; yt = []; yl = []; cols = []
        for f in S.FAM:
            q = r[r.Family == f].sort_values('Score', ascending=False)
            if q.empty: continue
            for _, z in q.iterrows():
                ax.scatter(z.Score, y, s=90 + 2600 * z.Test5seeds_sd, marker=S.FAMM[f], fc=S.FAMC[f], ec='k', lw=1.3, zorder=3)
                ax.errorbar(z.Score, y, xerr=z.Test5seeds_sd, color='k', lw=1.3, capsize=3, zorder=2)
                yt.append(y); yl.append(z.Model); cols.append(S.FAMC[f]); y += 1
            ax.plot([q.Score.mean()] * 2, [y - len(q) - .35, y - .65], color=S.FAMC[f], lw=3.2, zorder=1, solid_capstyle='butt')
            y += 0.7
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontweight='bold', fontsize=12.5); ax.invert_yaxis(); ax.set_ylim(y - .5, -.9)
        for t, c in zip(ax.get_yticklabels(), cols): t.set_color(c if c not in ('#F59E0B', '#7BC62D') else '#222')
        ax.set_title(SHORT[n].replace('\n', ' · '), fontsize=16); ax.set_xlabel('R² (random test)' if kind == 'r2' else 'macro-F1 (random test)')
        rng = r.Score.max() - r.Score.min(); ax.set_xlim(r.Score.min() - 0.06 * rng - 0.02, min(1.03, r.Score.max() + 0.06 * rng + 0.02)); S.panel(ax, 'abcdef'[[d[0] for d in DS].index(n)], dx=-0.30)
    axs[6].axis('off'); axs[8].axis('off'); ax = axs[7]; ax.axis('off'); family_legend(axs[6], loc='center', ncol=2, title='Model family')
    for lab, v, yy in [('0.00', 0.00, 0.52), ('0.02', 0.02, 0.40), ('0.05', 0.05, 0.27)]:
        ax.scatter([0.25], [yy], s=90 + 2600 * v, fc='#DDD', ec='k', lw=1.4, transform=ax.transAxes); ax.text(0.38, yy, f'5-seed s.d. = {lab}', transform=ax.transAxes, va='center', fontsize=15)
    ax.text(0.5, 0.62, 'Marker size and error bar = s.d. of the score over five random splits', ha='center', va='center', transform=ax.transAxes, fontsize=13.5, style='italic', wrap=True)
    ax.text(0.5, 0.12, 'Coloured bar = family mean.', ha='center', va='center', transform=ax.transAxes, fontsize=14, style='italic')
    save(fig, 'fig04_family_arena')


if __name__ == '__main__':
    for f in (fig01_leaderboard, fig02_gap, fig03_bubble, fig04_arena): f()
