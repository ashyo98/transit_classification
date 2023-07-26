from utils import collect_curves_tofiles

if __name__ == "__main__":
    n_curves = 200
    pct_transit = 50
    savepath = 'Data'
    collect_curves_tofiles(n_curves=n_curves, pct_transit=pct_transit, savepath=savepath, phase_fold=True, smooth=True)