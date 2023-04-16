from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt


def calc_flow_dfs(g, precip, retention=None, psi=None):
    """
    Calculates runoff from each subcatchment in the graph, taking into consideration precipitation, runon and retention volume.
    Args:
        g (nx.DiGraph): Graph of drainage network
        precip (float): Precipitation in mm
        retention (np.array): Retention volumes for [n for n in g.nodes]
        psi (float): Discharge coefficient, not implemented

    Returns:
        np.array: array with runoff from each catchment
    """
    # dfs search for root
    order = nx.dfs_successors(g.reverse(), "alois-hamtod-weg")
    # initialisieren der konstanten werte
    nodes = np.array([n for n in g.nodes])
    if retention is None:
        retention = np.zeros(len(nodes))

    precip = calc_VQR(g, precip)
    precip = np.array([precip[n] for n in nodes])
    vq = np.maximum(precip - retention, 0)
    retention = np.maximum(retention - precip, 0)
    for element in reversed(order):
        i = np.where(nodes==element)[0]
        mask = np.isin(nodes, order[element])
        vq[i] += max(vq[mask].sum() - retention[i], 0)
    return vq


def calc_VQR(g, precip):
    """
    Returns VQR in m³, takes precip in mm and area in ha
    Args:
        g (nx.DiGraph or nx.Graph): graph with subcatchments and assigned area in ha
        precip (float): precipitation in mm

    Returns:
        dict: dictionary with node as key and VQR as value
    """
    def vr(node):
        return g.nodes[node]["area_ha"] * precip * 10
    return {n: vr(n) for n in g.nodes}


def set_retention(node, volume, graph, retention=None):
    """
    Creates a new or overwrites an existing array at the correct position to insert a retention volume for a given subcatchment.
    Args:
        node (str): Name of subcatchment ot add retention volume to
        volume (float): Volume of retention
        graph: (nx.DiGraph): Graph of subcatchments
        retention (np.array): Existing retention array

    Returns:
        np.array: Array with retention volume for each node in [n for n in g.nodes]
    """
    if retention is None:
        retention = np.zeros(len(g.nodes))
    retention[np.where(np.array(graph.nodes) == node)] = volume
    return retention


def plot_vq(g, subcat, df_storms, retention=None, odir=None):
    """
    plots runoff from specified subcatchment for various design storms
    Args:
        g (nx.DiGraph): Graph of hydrological model
        subcat (str): Name of subcatchment for which to plot runoff
        df_storms (pd.DataFrame): Dataframe containing the precipitation volumes for return periods and durations
        retention (np.array): Numpy-Array containing retention volumes for each subcatchment
        odir (Path): Path for plot to write to, not implemented

    Returns:
        None
    """
    nodes = np.array(g.nodes)
    index_of_interest = np.where(nodes==subcat)[0][0]
    storms_of_interest = df_storms.columns.values
    durations = df_storms.index.values

    fig, ax1 = plt.subplots(constrained_layout=True, figsize=[8.5, 6])

    fig.suptitle(f"Abflussvolumina nach Jährlichkeit und Dauerstufe - {subcat}")

    sm = plt.cm.ScalarMappable(cmap=mpl.colormaps["cool"], norm=mpl.colors.LogNorm(vmin=1, vmax=100))

    for return_period in storms_of_interest:
        vq = list(map(lambda p: calc_flow_dfs(g, p, retention=retention)[index_of_interest], df_storms[return_period].values))
        ax1.plot(durations, vq, color=mpl.colormaps["cool"](mpl.colors.LogNorm(vmin=1, vmax=100)(return_period)))

    ax1.set(ylabel="Niederschlagsmenge [m³]",
            xlabel="Dauer [h]", xlim=[0, 1440], xticks=np.arange(0, 1620, 180))
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_formatter(lambda x, pos: f"{x/60:.0f}")
    formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))

    fig.colorbar(sm, ax=ax1, ticks=[1, 3, 5, 10, 30, 100], format=formatter, label="Jährlichkeit")
    custom_lines = [mpl.lines.Line2D([0], [0], color="black", lw=1),
                    mpl.lines.Line2D([0], [0], color="black", lw=1, linestyle="dotted")]
    fig.legend(custom_lines, ["Niederschlagsvolumen", "Niederschlagsintensität"], loc="upper left", bbox_to_anchor=(0.1,0.95))
    ax1.grid(zorder=0)
    odir = Path(r"Y:\PROJECTS\PeriSponge\07_Fallstudie\Feldbach\Detailanalyse\Modell-Check\02_subcats\plots")
    odir.mkdir(parents=True, exist_ok=True)
    #fig.savefig(odir/"niederschlagsvolumina_oedter.png")
    return
