import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# Times New Roman if installed; Liberation Serif is the metric-compatible drop-in used in this workspace
_avail = {f.name for f in fm.fontManager.ttflist}
SERIF = next((f for f in ['Times New Roman', 'Liberation Serif', 'Tinos', 'STIXGeneral', 'DejaVu Serif'] if f in _avail), 'serif')

FAM = ['Linear', 'Kernel', 'Neural net', 'Tree', 'Forest', 'Boosting', 'Nearest neighbour', 'Bayesian']
FAMC = {'Linear': '#E8293B', 'Kernel': '#F59E0B', 'Neural net': '#8B3FD9', 'Tree': '#0E9F8E', 'Forest': '#7BC62D',
        'Boosting': '#1565E0', 'Nearest neighbour': '#FF4FA3', 'Bayesian': '#00A6C8'}
FAMM = {'Linear': 's', 'Kernel': 'D', 'Neural net': 'p', 'Tree': '^', 'Forest': 'P', 'Boosting': 'o', 'Nearest neighbour': 'v', 'Bayesian': 'h'}
MODEL_FAMILY = {'OLS': 'Linear', 'Ridge': 'Linear', 'Lasso': 'Linear', 'ElasticNet': 'Linear', 'LogReg-L2': 'Linear', 'LogReg-L1': 'Linear',
                'SVR': 'Kernel', 'SVC': 'Kernel', 'MLP': 'Neural net', 'DT': 'Tree', 'RF': 'Forest', 'ET': 'Forest', 'GB': 'Boosting', 'XGB': 'Boosting',
                'LGBM': 'Boosting', 'AB': 'Boosting', 'CatBoost': 'Boosting', 'KNN': 'Nearest neighbour', 'BayesRidge': 'Bayesian', 'GPR': 'Bayesian',
                'GPC': 'Bayesian', 'GNB': 'Bayesian'}
# feature groups for SHAP / importance figures
GROUPC = {'Geometry': '#FF6B00', 'Adherend': '#6A00F4', 'Adhesive & filler': '#00B37A', 'Cure & process': '#D81B60',
          'Surface treatment': '#0096D6', 'Test & environment': '#8A9A00'}
INK = '#000000'; GRID = '#D9D9D9'


def group_of(f):
    if f.startswith('Surf_'): return 'Surface treatment'
    if f in ('Test_Temp_C', 'Loading_Rate_mm_min', 'Humidity_pctRH', 'Aging_Days'): return 'Test & environment'
    if f.startswith(('Cure_', 'PostCure_')) or f in ('Primer_Used', 'Bonding_Route'): return 'Cure & process'
    if f.startswith(('Adhesive_Modulus', 'Adhesive_Strength', 'Adhesive_Elong', 'Adhesive_Class', 'Adhesive_Form', 'Filler', 'Mix_')): return 'Adhesive & filler'
    if f in ('Overlap_Length_mm', 'Joint_Width_mm', 'Adhesive_Thickness_mm', 'Fillet', 'Joint_Type_Class'): return 'Geometry'
    return 'Adherend'


NICE = {'Adhesive_Strength_MPa': 'Adhesive material strength (MPa)', 'Adherend1_Strength_MPa': 'Adherend 1 material strength (MPa)', 'Adherend2_Strength_MPa': 'Adherend 2 material strength (MPa)', 'Loading_Rate_mm_min': 'Loading rate (mm/min)', 'Humidity_pctRH': 'Humidity (%RH)', 'No_Plies': 'Number of plies', 'Mix_Ratio_A_per_B': 'Mix ratio (A:B)',
        'Surf_Abrasion_sanding_grinding_brushing': 'Surface: abrasion', 'Surf_Solvent_detergent_clean': 'Surface: solvent clean', 'Surf_Grit_sand_blasting': 'Surface: grit blasting',
        'Surf_Chemical_electrochemical': 'Surface: chemical/electrochem.', 'Surf_Ultrasonic_vibration_assisted': 'Surface: ultrasonic', 'Surf_None_as_received': 'Surface: none (as received)', 'Surf_Other_modifier': 'Surface: other modifier'}


def nice(f):
    import re
    if f in NICE: return NICE[f]
    suf = [('_mm_min', ' (mm/min)'), ('_mm', ' (mm)'), ('_GPa', ' (GPa)'), ('_MPa', ' (MPa)'), ('_pctRH', ' (%RH)'), ('_wtpct', ' (wt%)'), ('_pct', ' (%)'), ('_C', ' (°C)'), ('_h', ' (h)')]
    unit = ''
    for k, v in suf:
        if f.endswith(k): f = f[:-len(k)]; unit = v; break
    r = f.replace('_', ' ')
    r = re.sub(r'^Surf ', 'Surface: ', r)
    r = r.replace('Adherend1', 'Adherend 1').replace('Adherend2', 'Adherend 2').replace('Filler1', 'Filler 1').replace('Filler2', 'Filler 2').replace('PostCure', 'Post-cure')
    r = r.replace('Adhesive Thickness', 'Adhesive thickness').replace('Overlap Length', 'Overlap length').replace('Joint Width', 'Joint width').replace('Test Temp', 'Test temperature').replace('Cure Temp', 'Cure temperature').replace('Cure Duration', 'Cure duration').replace('Adhesive Modulus', 'Adhesive modulus').replace('Adhesive Strength', 'Adhesive strength').replace('Adhesive Elongation', 'Adhesive elongation').replace('Aging Days', 'Aging (days)').replace('Fiber Orientation Class', 'Fibre orientation')
    return r + unit


# vivid sequential (score) and diverging (residual) maps
SEQ = LinearSegmentedColormap.from_list('vivid_seq', ['#FFF4C2', '#FFB000', '#FF5A1F', '#D6006E', '#5B00C7', '#0A1F8F'])
SCORE = LinearSegmentedColormap.from_list('score', ['#B00020', '#FF6B00', '#FFD60A', '#7BC62D', '#00A676', '#006D77'])
DIV = LinearSegmentedColormap.from_list('vivid_div', ['#0047E0', '#2FA8FF', '#F5F5F5', '#FF9A3D', '#E0002B'])
CONF = LinearSegmentedColormap.from_list('conf', ['#FFFFFF', '#D6E6F5', '#8DB8E2', '#3F7FBF', '#17457F'])
IMP = LinearSegmentedColormap.from_list('imp', ['#FFFFFF', '#CFE8FF', '#3D9BFF', '#7B2CBF', '#E0006E'])


def setup():
    mpl.rcParams.update({
        'axes.unicode_minus': False, 'font.family': 'serif', 'font.serif': [SERIF], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'font.size': 14, 'axes.titlesize': 16, 'axes.titleweight': 'bold', 'axes.labelsize': 15, 'axes.labelweight': 'bold',
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5, 'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 1.3, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 1.2, 'ytick.major.width': 1.2, 'xtick.major.size': 4.5, 'ytick.major.size': 4.5,
        'xtick.minor.visible': False, 'ytick.minor.visible': False, 'axes.grid': False, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.7,
        'grid.linestyle': '--', 'axes.grid.which': 'major', 'legend.frameon': True, 'legend.edgecolor': INK, 'legend.framealpha': 0.95, 'legend.fancybox': False,
        'axes.spines.top': True, 'axes.spines.right': True, 'lines.linewidth': 2.0, 'patch.linewidth': 1.0, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def panel(ax, s, dx=-0.0, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=20, fontweight='bold', va='bottom', ha='left')
