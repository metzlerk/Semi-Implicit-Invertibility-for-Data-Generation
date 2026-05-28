import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
import argparse

# %% Functions
def axes():
    plt.figure(figsize=(8,2.5),dpi=600)
    plt.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0.2)
    return {0: plt.subplot2grid((1,2),(0,0),rowspan=1,colspan=1),
            1: plt.subplot2grid((1,2),(0,1),rowspan=1,colspan=1)}

def slugify(label):
    cleaned = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(label))
    return '_'.join(filter(None, cleaned.split('_')))


def classifier_tag(name):
    base_map = {
        "RandomForest": "rf",
        "MLP": "mlp",
    }
    return base_map.get(name, slugify(name))


def getdata(filename, classifiers=None):
    df = pd.read_csv(filename)
    if classifiers is None:
        classifiers = sorted(df["classifier"].dropna().unique())
    data_by_classifier = {}
    for cl in classifiers:
        class_df = df[df["classifier"] == cl]
        if class_df.empty:
            continue
        labels = sorted(class_df["std_label"].dropna().unique())
        data_by_classifier[cl] = {}
        for lb in labels:
            data_by_classifier[cl][lb] = class_df[class_df["std_label"] == lb]
    return data_by_classifier

def plotdata(ax,data,ycol_name,skip=None):
    pews = pe.withStroke(foreground="white",linewidth=3)
    colors,lines = ["b","g","r"],["-.","--","-"]
    for i,k in enumerate(data):
        ax.plot(data[k].iloc[skip:]["num_points_per_class"],
                data[k].iloc[skip:][ycol_name],
                path_effects=[pews],
                linestyle=lines[i],
                color=colors[i],
                label=f"{k}")
        
def plotshared(ax,ylabel=None,legend_loc=None):
    for ax in (ax[0],ax[1]):
        ax.axhline(0,color='k',linewidth=0.75,linestyle="--",zorder=1)
        ax.set_xlabel("points per class")
        ax.set_ylabel(ylabel)
        ax.grid(color="gray",linewidth=0.5,alpha=0.2)
        ax.legend(loc=legend_loc)
        
# %% Figures
def Figure1():
    data = getdata("plot_data_gain_over_real_only.csv")
    ycol_name,skip,ax = "gain_over_real_only",2,axes()
    
    plotdata(ax[0],data.get("RandomForest", {}),ycol_name,skip)
    plotdata(ax[1],data.get("MLP", {}),ycol_name,skip)
    ax[0].set_title("Random Forest: Accuracy Gain vs. Real-Only",fontsize=10)
    ax[1].set_title("MLP: Accuracy Gain vs. Real-Only",fontsize=10)
    plotshared(ax,"accuracy gain")

def Figure2():
    data = getdata("plot_data_optimal_real_ratio.csv")
    ycol_name,skip,ax = "optimal_real_ratio",None,axes()
    
    plotdata(ax[0],data.get("RandomForest", {}),ycol_name,skip)
    plotdata(ax[1],data.get("MLP", {}),ycol_name,skip)
    ax[0].set_title("Random Forest: Optimal Real-Data Fraction",fontsize=10)
    ax[1].set_title("MLP: Optimal Real-Data Fraction",fontsize=10)
    plotshared(ax,"ratio","center right")
    
# %% Heatmap functions
def iter_frames(data):
    if isinstance(data, dict):
        for value in data.values():
            yield from iter_frames(value)
    elif isinstance(data, (tuple, list)):
        for value in data:
            yield from iter_frames(value)
    else:
        yield data


def minmax(data):
    stats = pd.DataFrame(columns=["min","max"])
    for frame in iter_frames(data):
        row = len(stats)
        stats.loc[row,"min"] = frame.min().min()
        stats.loc[row,"max"] = frame.max().max()
    return stats["min"].min(),stats["max"].max()

def getacc(dd,ratio=None):
    new_dd = {}
    for classifier_name, std_map in dd.items():
        new_dd[classifier_name] = {}
        for k in std_map:
            df = std_map[k].copy()
            df["real_ratio"] = df["real_ratio"].round(1)
            df = df.pivot(index="num_points_per_class", columns="real_ratio", values="accuracy")
            if ratio is not None:
                df = df.sub(df[ratio],axis="rows")
            new_dd[classifier_name][k] = df
    return new_dd

def compare(dd,ratio,mode):
    new_dd = {}
    for classifier_name, std_map in dd.items():
        new_dd[classifier_name] = {}
        for k in std_map:
            df = std_map[k].copy()
            if mode == "gt":
                new_dd[classifier_name][k] = df.gt(df[ratio],axis="rows")
            if mode == "lt":
                new_dd[classifier_name][k] = df.lt(df[ratio],axis="rows")
    return new_dd

def FigureH(data,*,title,cnorm=None):
    if type(cnorm) is tuple: vmin,vmax = cnorm
    if cnorm is None: vmin,vmax = minmax(data)
    if cnorm == "off": vmin,vmax = None,None
    
    n_panels = len(data)
    max_x = max((len(df.index) for df in data.values()), default=1)
    fig_width = max(10, 0.42 * max_x * n_panels)
    ax,mesh,fig = {},{},plt.figure(figsize=(fig_width,2.6),dpi=600)
    plt.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0.03)
    kws1 = {"weight":"bold","size":9}
    kws2 = {"labelsize":9,"pad":2.5}
    for i,k in enumerate(data):
        df = data[k]
        X,Y = np.meshgrid(df.index,df.columns)
        ax[i] = plt.subplot2grid((1,n_panels),(0,i),rowspan=1,colspan=1)
        mesh[i] = ax[i].pcolormesh(X,Y,df.T,vmin=vmin,vmax=vmax,cmap="magma")
        ax[i].set_title(f"{title} | {k}",size=9,weight="bold")
        x_values = np.asarray(df.index, dtype=float)
        y_values = np.asarray(df.columns, dtype=float)
        y_labels = [f"{value:.1f}" for value in y_values]
        xtick_values = x_values
        x_labels = [f"{int(value):d}" for value in xtick_values]
        ax[i].set_xlabel("n (training points / class)",labelpad=1.5,**kws1)
        ax[i].set_ylabel("r (real-data ratio)",**kws1)
        ax[i].set_xticks(xtick_values)
        ax[i].set_xticklabels(x_labels, rotation=90)
        ax[i].set_yticks(y_values)
        ax[i].set_yticklabels(y_labels)
        ax[i].tick_params(**kws2)
        if i != 0: ax[i].get_yaxis().set_visible(False)    
    if cnorm != "off":
        cbar = fig.colorbar(mesh[0],pad=0.0075,ax=[ax for ax in ax.values()])
        cbar.ax.tick_params(**kws2)
        cbar.set_label("accuracy (aa)",**kws1)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary and heatmap plots from evaluation CSV.")
    parser.add_argument(
        "--csv-path",
        default="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/eval_synthetic_metrics.csv",
        help="Path to evaluation CSV from 4f-kjm-evaluatesynthetic-data.py",
    )
    parser.add_argument(
        "--classifiers",
        default="",
        help="Comma-separated classifier names to plot (default: all in CSV).",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results",
        help="Directory to save generated heatmap PNG files.",
    )
    return parser.parse_args()
        
# %% MAIN
if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    classifiers = [c.strip() for c in args.classifiers.split(',') if c.strip()] or None
    ## gets raw data
    dd = getdata(args.csv_path, classifiers=classifiers)
    
    ## converts to pivot table
    aa = getacc(dd)
    
    ## calculates difference relative to rows for ratio=r=1.0 baseline
    aad = getacc(dd,ratio=1.0)
    
    ## gets normalization values for shared colormap
    global_minmax = minmax(aa)
    
    ## Figures
    
    # Figure1()
    # Figure2()
    
    ## accuracy, global colormap
    for classifier_name, classifier_data in aa.items():
        FigureH(classifier_data,title=f"{classifier_name} | Acc",cnorm=global_minmax)
        plt.savefig(
            out_dir / f"heatmap_{classifier_tag(classifier_name)}_accuracy.png",
            bbox_inches="tight",
        )
        plt.close()
    
    # ## accuracy difference, independent colormap
    for classifier_name, classifier_data in aad.items():
        FigureH(classifier_data,title=f"{classifier_name} | \u0394Acc",cnorm="off")
        plt.savefig(
            out_dir / f"heatmap_{classifier_tag(classifier_name)}_delta_accuracy.png",
            bbox_inches="tight",
        )
        plt.close()
    
    ## examples
    
    ## ratio=None, ratio=0.0, ratio=0.5
    # FigureH(getacc(dd_MLP,ratio=None),
    #         title="MLP | ratio=None | Acc(r)")
    # FigureH(getacc(dd_MLP,ratio=0.0),
    #         title="MLP | ratio=0.0 | Acc(r)-Acc(r=0.0)")
    # FigureH(getacc(dd_MLP,ratio=0.5),
    #         title="MLP | ratio=0.5 | Acc(r)-Acc(r=0.5)")
    
    ## cnorm=None vs. cnorm="off"
    # FigureH(getacc(dd_MLP,ratio=0.0),cnorm=None,
    #         title="MLP | ratio=0.0, cnorm=None")
    # FigureH(getacc(dd_MLP,ratio=0.0),cnorm="off",
    #         title="MLP | ratio=0.0, cnorm='off'")
    
    ## booleans
    # FigureH(compare(aa_RF,ratio=1.0,mode="gt"),
    #         title="RF | mode='gt' | Acc(r) > Acc(r=1.0)")
    # FigureH(compare(aa_MLP,ratio=1.0,mode="gt"),
    #         title="MLP | mode='gt' | Acc(r) > Acc(r=1.0)")
