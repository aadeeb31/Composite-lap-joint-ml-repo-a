import sys, os, pickle, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import style as S
S.setup()
SRC = str(ROOT / 'data/processed'); Z = str(ROOT / 'results/paper'); ANA = str(ROOT / 'results/paper/ana'); FIG = str(ROOT / 'figures')
os.makedirs(FIG, exist_ok=True)
META = ['Row_ID', 'Paper_ID', 'Value_Type', 'Target_Is_Derived']
DS = [('CFRP_CFRP__MaxLoad', 'CFRP_CFRP', 'CFRP-CFRP · maximum load', 'r2'),
      ('Composite_Composite__MaxLoad', 'Composite_Composite', 'Composite-composite · maximum load', 'r2'),
      ('CFRP_CFRP__LSS', 'CFRP_CFRP', 'CFRP-CFRP · lap-shear strength', 'r2'),
      ('CFRP_CFRP__FailureMode', 'CFRP_CFRP', 'CFRP-CFRP · failure mode', 'f1'),
      ('Composite_Composite__FailureMode', 'Composite_Composite', 'Composite-composite · failure mode', 'f1'),
      ('Composite_Composite__LSS', 'Composite_Composite', 'Composite-composite · lap-shear strength', 'r2')]
SHORT = {'CFRP_CFRP__MaxLoad': 'CFRP-CFRP\nmax load', 'Composite_Composite__MaxLoad': 'Composite-composite\nmax load', 'CFRP_CFRP__LSS': 'CFRP-CFRP\nLSS',
         'CFRP_CFRP__FailureMode': 'CFRP-CFRP\nfailure mode', 'Composite_Composite__FailureMode': 'Composite-composite\nfailure mode', 'Composite_Composite__LSS': 'Composite-composite\nLSS'}
UNIT = {'CFRP_CFRP__MaxLoad': 'kN', 'Composite_Composite__MaxLoad': 'kN', 'CFRP_CFRP__LSS': 'MPa', 'Composite_Composite__LSS': 'MPa'}
RES = pd.read_csv(Z + '/zoo_results.csv')
PRED = pickle.load(open(Z + '/preds.pkl', 'rb'))
RES['Family'] = RES.Model.map(S.MODEL_FAMILY)
RES['Score'] = np.where(RES.Target == 'Failure_Mode', RES.Test_f1, RES.Test_r2)
RES['OOF'] = RES.GroupOOF_score
RES['Rand'] = RES.Random5fold_score
FIDX = {f: i for i, f in enumerate(S.FAM)}


def load_ds(name):
    rel = [r for n, r, _, _ in DS if n == name][0]
    d = pd.read_csv(f'{SRC}/{rel}/{name}.csv'); return d


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', dpi=300, facecolor='white'); fig.savefig(f'{FIG}/{name}.pdf', facecolor='white'); plt.close(fig); print('saved', name, flush=True)


def ordered_models(df):
    df = df.copy(); df['fi'] = df.Family.map(FIDX); return df.sort_values(['fi', 'Score'], ascending=[True, False])


def class_names(col='Failure_Mode_Class'):
    m = pd.read_csv(f'{SRC}/Label_Encoder_Maps.csv'); m = m[m.Column == col]; return dict(zip(m.Code, m.Category))


def family_legend(ax, loc='best', ncol=1, title=None, fs=9.5):
    h = [Line2D([], [], marker=S.FAMM[f], color='w', markerfacecolor=S.FAMC[f], markeredgecolor='k', markeredgewidth=1.1, markersize=9.5, label=f) for f in S.FAM]
    return ax.legend(handles=h, loc=loc, ncol=ncol, title=title, fontsize=fs, title_fontsize=fs + 0.5, handletextpad=0.3, borderpad=0.6)
