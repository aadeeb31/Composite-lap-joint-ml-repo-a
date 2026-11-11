from figlib import *
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

FV = LinearSegmentedColormap.from_list('fv', ['#1E88FF', '#B04FD8', '#FF2D6A'])
VT_MARK = {'Lower': 'v', 'Mean': 'o', 'Upper': '^', 'Single': 's'}
DER_C = '#00A676'
CLS_C = ['#E8293B', '#F59E0B', '#1565E0', '#0E9F8E', '#8B3FD9', '#FF4FA3', '#7BC62D', '#00A6C8', '#8A9A00']


def parity_hero(name):
    d = load_ds(name); tgt = d.columns[-1]; r = RES[RES.File == name]; best = r.sort_values('Score').iloc[-1]; P = PRED[(name, best.Model)]
    tr, te = P['tr'], P['te']; ytr, yte, ptr, pte = P['ytr'], P['yte'], P['ptr'], P['pte']
    vt = d.Value_Type.values; der = d.Target_Is_Derived.values.astype(bool); unit = UNIT[name]
    fig = plt.figure(figsize=(8.8, 8.8)); gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1:, :3]); at = fig.add_subplot(gs[0, :3], sharex=ax); ar = fig.add_subplot(gs[1:, 3], sharey=ax)
    mx = max(ytr.max(), yte.max(), ptr.max(), pte.max()) * 1.05; lo = min(0, pte.min(), yte.min())
    TRC, TEC = '#2F6DB5', '#E8710A'
    ax.scatter(ytr, ptr, s=34, marker='D', fc=TRC, ec='k', lw=0.5, alpha=.55, zorder=1, label='Training')
    ax.scatter(yte, pte, s=70, marker='x', c=TEC, linewidths=2.4, zorder=3, label='Testing')
    ax.plot([lo, mx], [lo, mx], color='k', lw=2.4, zorder=2); ax.plot([lo, mx], [lo * 1.1, mx * 1.1], color='k', lw=1.1, ls='--', zorder=2); ax.plot([lo, mx], [lo * .9, mx * .9], color='k', lw=1.1, ls='--', zorder=2)
    ax.set_xlim(lo, mx); ax.set_ylim(lo, mx * 1.42); ax.set_xlabel(f'Measured {tgt.split("_")[0]} ({unit})'.replace('MaxLoad', 'maximum load')); ax.set_ylabel(f'Predicted ({unit})')
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    low = yte <= 25
    extra = f'\nR² on rows ≤ 25 kN = {r2_score(yte[low], pte[low]):.2f}  (n = {low.sum()})' if (unit == 'kN') else ''
    st = (f'{best.Model} (best of 17)\nR² test = {best.Test_r2:.3f}\nMAE = {best.Test_mae:.2f} {unit}   RMSE = {np.sqrt(best.Test_mse):.2f} {unit}\n'
          f'5 seeds: {best.Test5seeds_mean:.3f} ± {best.Test5seeds_sd:.3f}\nn(test) = {len(te)},  n(train) = {len(tr)}')
    ax.text(0.03, 0.97, st, transform=ax.transAxes, va='top', fontsize=13.0, bbox=dict(boxstyle='square,pad=0.5', fc='white', ec='k', lw=1.3), zorder=6)
    # inset zoom for heavy-tailed targets
    q90 = np.percentile(np.r_[ytr, yte], 90)
    if np.r_[ytr, yte].max() / q90 > 3:
        z = q90 * 1.35; ia = ax.inset_axes([0.62, 0.07, 0.34, 0.26]); ia.set_facecolor('white')
        ia.scatter(ytr, ptr, s=14, marker='D', fc=TRC, ec='k', lw=0.3, alpha=.55)
        ia.scatter(yte, pte, s=36, marker='x', c=TEC, linewidths=1.8)
        ia.plot([0, z], [0, z], 'k', lw=1.6); ia.set_xlim(0, z); ia.set_ylim(0, z); ia.tick_params(labelsize=10.4); ia.grid(False)
        ia.set_title(f'Zoom: 0 to {z:.0f} {unit} (90 % of data)', fontsize=11.4, pad=3)
        for sp in ia.spines.values(): sp.set_linewidth(1.6)
        ax.indicate_inset_zoom(ia, ec='k', lw=1.1)
    # marginals
    bins = np.linspace(lo, mx, 36)
    at.hist(ytr, bins, color=TRC, ec='k', lw=.6, label='Training'); at.hist(yte, bins, color=TEC, ec='k', lw=.6, label='Testing'); at.set_yscale('log'); at.tick_params(labelbottom=False); at.set_ylabel('count'); at.set_ylim(at.get_ylim()[0], at.get_ylim()[1] * 12); at.legend(fontsize=11.1, loc='upper right')
    ar.hist(ptr, bins, orientation='horizontal', color=TRC, ec='k', lw=.6); ar.hist(pte, bins, orientation='horizontal', color=TEC, ec='k', lw=.6); ar.set_xscale('log'); ar.tick_params(labelleft=False); ar.set_xlabel('count')
    ax.legend(loc='upper left', fontsize=13, markerscale=1.3, bbox_to_anchor=(0.03, 0.66))
    fig.suptitle([t for n, _, t, _ in DS if n == name][0], fontweight='bold', fontsize=16.9, y=0.96)
    save(fig, f'fig05_parity_hero_{name}')


def parity_family(name):
    from matplotlib.ticker import LogLocator, NullFormatter
    r = RES[RES.File == name]; reps = r.loc[r.groupby('Family').Score.idxmax()].set_index('Family'); unit = UNIT[name]
    fig = plt.figure(figsize=(14.2, 15.0)); gs = fig.add_gridspec(3, 3, left=0.075, right=0.985, top=0.885, bottom=0.06, hspace=0.62, wspace=0.34)
    axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(len(S.FAM))]
    allv = np.concatenate([PRED[(name, reps.loc[f].Model)]['yte'] for f in S.FAM])
    lo0, hi0 = max(allv.min() * 0.4, 0.04), allv.max() * 3.0
    lo, hi = 10 ** np.floor(np.log10(lo0)), 10 ** np.ceil(np.log10(hi0))
    for k, (ax, f) in enumerate(zip(axs, S.FAM)):
        q = reps.loc[f]; P = PRED[(name, q.Model)]; y, p = P['yte'], P['pte']; neg = p <= lo0 * 1.05; pp = np.where(neg, lo0 * 1.25, p); lr = np.log(pp / y)
        sc = ax.scatter(y, pp, s=40 + 230 * np.clip(np.abs(lr) / 1.0, 0, 1.2), c=lr, cmap=S.DIV, norm=TwoSlopeNorm(0, -1, 1), marker=S.FAMM[f], ec='k', lw=1.0, zorder=3)
        ax.plot([lo, hi], [lo, hi], 'k', lw=2.2); ax.plot([lo, hi], [lo * 1.25, hi * 1.25], 'k--', lw=1); ax.plot([lo, hi], [lo / 1.25, hi / 1.25], 'k--', lw=1)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, numticks=12))
            axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=60)); axis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='major', length=9, width=1.6, labelsize=14.5); ax.tick_params(which='minor', length=5, width=1.0)
        mape = np.median(np.abs(p - y) / y) * 100
        ax.set_title(f'{f} ({q.Model})\nR² = {q.Test_r2:.3f}, MAE = {q.Test_mae:.2f} {unit}\nmedian |err| = {mape:.0f}%', fontsize=15.5, linespacing=1.45, pad=9)
        for sp in ax.spines.values(): sp.set_edgecolor(S.FAMC[f]); sp.set_linewidth(3.4)
        ax.set_xlabel(f'Measured ({unit})', fontsize=15.5)
        if k % 3 == 0: ax.set_ylabel(f'Predicted ({unit})', fontsize=15.5)
    al = fig.add_subplot(gs[2, 2]); al.axis('off')
    cax = al.inset_axes([0.04, 0.60, 0.92, 0.07]); cb = fig.colorbar(sc, cax=cax, orientation='horizontal'); cb.ax.tick_params(labelsize=13.5, length=5)
    al.text(0.5, 0.40, 'ln(predicted / measured)\nblue = under-prediction\nred = over-prediction', ha='center', va='top', fontsize=14.5, linespacing=1.5, transform=al.transAxes)
    al.text(0.5, 0.10, 'marker area ~ |ln ratio|\ndashed lines = ±25 %', ha='center', va='top', fontsize=14.5, linespacing=1.5, transform=al.transAxes)
    fig.suptitle(f'{[t for n, _, t, _ in DS if n == name][0]}: best model of each family', fontweight='bold', fontsize=19, y=0.992)
    save(fig, f'fig06_parity_by_family_{name}')


def _swarm(vals, ymid, width=0.42):
    h, e = np.histogram(vals, bins=40); b = np.clip(np.digitize(vals, e[1:-1]), 0, len(h) - 1); c = h[b] / max(h.max(), 1)
    return ymid + np.random.RandomState(0).uniform(-1, 1, len(vals)) * width * c


def shap_dash(name, top=14):
    A = pickle.load(open(f'{ANA}/{name}.pkl', 'rb')); sv = A['shap']['values']; feats = A['features']; Xraw = A['shap']['Xraw']
    ttl = [t for n, _, t, _ in DS if n == name][0]
    if not A['clf']:
        m = np.abs(sv).mean(0); o = np.argsort(m)[::-1][:top][::-1]
        fig = plt.figure(figsize=(14.2, 7.6)); gs = GridSpec(2, 2, height_ratios=[0.09, 1], width_ratios=[1, 1.55], hspace=0.12, wspace=0.06)
        a0 = fig.add_subplot(gs[1, 0]); a1 = fig.add_subplot(gs[1, 1], sharey=a0); ag = fig.add_subplot(gs[0, 0]); ag.grid(False)
        for k, i in enumerate(o):
            c = S.GROUPC[S.group_of(feats[i])]; a0.barh(k, m[i], color=c, ec='k', lw=1.4, height=.74, zorder=2)
            a0.text(m[i] + m[o].max() * 0.035, k, f'{m[i]:.2f}', va='center', fontsize=11.7, fontweight='bold')
        a0.set_yticks(range(len(o))); a0.set_yticklabels([S.nice(feats[i]) for i in o], fontsize=13.4); a0.set_xlabel('mean |SHAP|', labelpad=8); a0.set_xlim(0, m[o].max() * 1.22)
        for k, i in enumerate(o):
            x = sv[:, i]; xr = Xraw[:, i]; lo, hi = np.nanpercentile(xr, [5, 95]); cv = np.clip((xr - lo) / (hi - lo + 1e-12), 0, 1); cv = np.where(np.isnan(cv), .5, cv)
            a1.scatter(x, _swarm(x, k), s=15, c=cv, cmap=FV, ec='k', lw=.2, alpha=.9, zorder=3, rasterized=True)
        a1.axvline(0, color='k', lw=1.4); a1.set_xlabel('SHAP value (impact on prediction)', labelpad=8); a1.tick_params(labelleft=False)
        lo_, hi_ = np.percentile(sv[:, o], [0.3, 99.7]); pad = 0.12 * (hi_ - lo_); a1.set_xlim(lo_ - pad, hi_ + pad)
        nout = int(((sv[:, o] < lo_ - pad) | (sv[:, o] > hi_ + pad)).sum())
        if nout: a1.text(0.99, 0.015, f'{nout} points beyond the axis (|SHAP| up to {np.abs(sv[:, o]).max():.0f})', transform=a1.transAxes, ha='right', va='bottom', fontsize=11.7, style='italic', bbox=dict(fc='white', ec='none', pad=1.5)); a1.tick_params(labelleft=False); a1.set_ylim(-.7, len(o) - .3)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=FV), ax=a1, fraction=0.025, pad=0.01); cb.set_ticks([0, 1]); cb.set_ticklabels(['low', 'high']); cb.set_label('feature value', fontsize=13.0, labelpad=-8); cb.outline.set_linewidth(1.2)
        tot = np.abs(sv).mean(0); gsum = {}
        for i, f in enumerate(feats): gsum[S.group_of(f)] = gsum.get(S.group_of(f), 0) + tot[i]
        s0 = sum(gsum.values()); left = 0
        for gname, v in sorted(gsum.items(), key=lambda t: -t[1]):
            w = v / s0; ag.barh(0, w, left=left, color=S.GROUPC[gname], ec='k', lw=1.3, height=.9)
            if w > .06: ag.text(left + w / 2, 0, (f'{gname}\n{w*100:.0f}%' if w > .3 else f'{w*100:.0f}%'), ha='center', va='center', fontsize=11.2, fontweight='bold', color='white')
            left += w
        ag.set_xlim(0, 1); ag.axis('off'); ag.set_title(f'Share of total SHAP by feature group: {A["shap"]["model"]}', fontsize=13.7, loc='left')
        fig.suptitle(ttl, fontweight='bold', fontsize=17.6, y=0.99)
    else:
        cn = class_names(); ncl = sv.shape[2]; mc = np.abs(sv).mean(0); tot = mc.sum(1); o = np.argsort(tot)[::-1][:top][::-1]
        import joblib; classes = list(joblib.load(f'{ROOT}/results/models/{name}__{A["shap"]["model"]}.joblib').classes_)
        fig = plt.figure(figsize=(14.2, 7.6)); gs = GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.04); a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharey=a0)
        left = np.zeros(len(o))
        for ci in range(ncl):
            a0.barh(range(len(o)), mc[o, ci], left=left, color=CLS_C[ci], ec='k', lw=1.0, height=.72, label=cn.get(classes[ci], classes[ci])); left += mc[o, ci]
        a0.set_yticks(range(len(o))); a0.set_yticklabels([S.nice(feats[i]) for i in o], fontsize=13.4); a0.set_xlabel('mean |SHAP|, stacked by class'); a0.legend(loc='lower right', fontsize=11.7, title='Failure mode', title_fontsize=12.3)
        for k, i in enumerate(o):
            for ci in range(ncl):
                v = sv[:, i, ci]; xr = np.nan_to_num(Xraw[:, i], nan=np.nanmedian(Xraw[:, i])); dirn = np.corrcoef(xr, v)[0, 1] if v.std() > 0 and xr.std() > 0 else 0
                a1.scatter(ci, k, s=30 + 1250 * (mc[i, ci] / mc.max()), c=[dirn], cmap=S.DIV, norm=TwoSlopeNorm(0, -1, 1), ec='k', lw=1.1, zorder=3)
        a1.set_xticks(range(ncl)); a1.set_xticklabels([cn.get(c, c) for c in classes], rotation=0, ha='center', fontsize=12.8); a1.tick_params(labelleft=False); a1.set_xlim(-.6, ncl - .4); a1.set_ylim(-.7, len(o) - .3)
        a1.set_title('Class fingerprint: bubble area = mean |SHAP|, colour = direction', fontsize=14.0)
        lax = a1.inset_axes([0.14, -0.36, 0.72, 0.21]); lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.set_xticks([]); lax.set_yticks([]); lax.grid(False)
        for sp in lax.spines.values(): sp.set_linewidth(1.2)
        for xx, frac in zip((0.2, 0.5, 0.8), (0.25, 0.5, 1.0)):
            lax.scatter([xx], [0.56], s=30 + 1250 * frac, c='#DDDDDD', ec='k', lw=1.0, clip_on=False); lax.text(xx, 0.1, f'{frac * mc.max():.2f}', ha='center', va='center', fontsize=11.5)
        lax.set_title('mean |SHAP|', fontsize=12, pad=4)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -1, 1), cmap=S.DIV), ax=a1, fraction=0.03, pad=0.01); cb.set_ticks([-1, 0, 1]); cb.set_ticklabels(['high value\nlowers prob.', '0', 'high value\nraises prob.']); cb.outline.set_linewidth(1.2)
        fig.suptitle(f'{ttl}: {A["shap"]["model"]}', fontweight='bold', fontsize=17.6, y=0.97)
    h = [Patch(fc=c, ec='k', label=g) for g, c in S.GROUPC.items()]
    fig.legend(handles=h, loc='lower center', ncol=6, fontsize=12.3, bbox_to_anchor=(0.5, -0.115), title='Feature group (bar colour)' if not A['clf'] else 'Feature groups: see variable names', title_fontsize=12.3) if not A['clf'] else None
    save(fig, f'fig07_shap_dashboard_{name}')


def shap_interaction(name, top=11):
    A = pickle.load(open(f'{ANA}/{name}.pkl', 'rb'))
    if 'shap_int' not in A: return
    M = A['shap_int']['mean_abs']; feats = A['features']; main = np.diag(M).copy(); o = np.argsort(main)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8.6, 7.6)); ax.grid(False)
    off = M.copy(); np.fill_diagonal(off, 0); vmax = (2 * off[np.ix_(o, o)]).max()
    for a, i in enumerate(o):
        for b, j in enumerate(o):
            if b > a: continue
            if a == b: ax.scatter(b, a, s=120 + 1300 * main[i] / main[o].max(), marker='s', fc='#FFD60A', ec='k', lw=1.6, zorder=3)
            else:
                v = 2 * M[i, j]; ax.scatter(b, a, s=80 + 2000 * v / vmax, marker='o', c=[v], cmap=S.IMP, vmin=0, vmax=vmax, ec='k', lw=1.3, zorder=3)
                if v > .5 * vmax: ax.text(b, a, f'{v:.2f}', ha='center', va='center', fontsize=10.1, fontweight='bold', color='white')
    ax.set_xticks(range(top)); ax.set_xticklabels([S.nice(feats[i]) for i in o], rotation=90, ha='center', fontsize=12.5); ax.set_yticks(range(top)); ax.set_yticklabels([S.nice(feats[i]) for i in o], fontsize=12.5)
    ax.set_xlim(-.6, top - .4); ax.set_ylim(top - .4, -.6); [s.set_linewidth(1.5) for s in ax.spines.values()]
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=S.IMP), ax=ax, fraction=0.04, pad=0.02); cb.set_label('mean |SHAP interaction| (pair, both triangles summed)'); cb.outline.set_linewidth(1.2)
    ax.set_title(f'{[t for n, _, t, _ in DS if n == name][0]}\nSquares (gold) = main effect · circles = pairwise interaction ({A["shap_int"]["model"]})', fontsize=15.0)
    save(fig, f'fig08_shap_interactions_{name}')


if __name__ == '__main__':
    import sys
    for n in [a for a in sys.argv[1:]] or [d[0] for d in DS]:
        if n.endswith('MaxLoad') or n.endswith('LSS'): parity_hero(n); parity_family(n)
        if os.path.exists(f'{ANA}/{n}.pkl'): shap_dash(n); shap_interaction(n)
