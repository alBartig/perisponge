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
        retention = np.zeros(len(graph.nodes))
    retention[np.where(np.array(graph.nodes) == node)] = volume
    return retention


def plot_vq(g, subcat, df_storms, retention=None, odir=None, save=False):
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
    index_of_interest = np.where(nodes == subcat)[0][0]
    storms_of_interest = df_storms.columns.values
    durations = df_storms.index.values

    fig, ax1 = plt.subplots(constrained_layout=True, figsize=[8.5, 6])

    fig.suptitle(f"Abflussvolumina nach Jährlichkeit und Dauerstufe von: {subcat}")

    sm = plt.cm.ScalarMappable(cmap=mpl.colormaps["cool"], norm=mpl.colors.LogNorm(vmin=1, vmax=100))

    for return_period in storms_of_interest:
        vq = list(map(lambda p: calc_flow_dfs(g, p, retention=retention)[index_of_interest], df_storms[return_period].values))
        ax1.plot(durations, vq, color=mpl.colormaps["cool"](mpl.colors.LogNorm(vmin=1, vmax=100)(return_period)))

    ax1.set(ylabel="Abflussvolumen [m³]",
            xlabel="Dauer des Regenereignisses [h]", xlim=[0, 1440], xticks=np.arange(0, 1620, 180))
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_formatter(lambda x, pos: f"{x/60:.0f}")
    formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))

    fig.colorbar(sm, ax=ax1, ticks=[1, 3, 5, 10, 30, 100], format=formatter, label="Jährlichkeit")
    custom_lines = [mpl.lines.Line2D([0], [0], color="black", lw=1),
                    mpl.lines.Line2D([0], [0], color="black", lw=1, linestyle="dotted")]
    # fig.legend(custom_lines, ["Oberflächenabfluss Gesamt", "Oberflächenabfluss Intensität"],
    #            loc="upper left", bbox_to_anchor=(0.1,0.9))
    ax1.grid(zorder=0)
    if save:
        fig.savefig(odir/f"oberflaechenabfluss_{subcat}.png", dpi=600, transparent=True)
    return


def plot_total_runoff(df_storms, df_subcats, odir=None, save=False):
    storms_of_interest = df_storms.columns.values
    durations = df_storms.index.values
    area = df_subcats["area_ha"].sum()

    fig, ax1 = plt.subplots(constrained_layout=True, figsize=[8.5, 6])
    ax2 = ax1.twinx()

    fig.suptitle("Anfallende Niederschlagsvolumina nach Jährlichkeit und Dauerstufe")

    sm = plt.cm.ScalarMappable(cmap=mpl.colormaps["cool"], norm=mpl.colors.LogNorm(vmin=1, vmax=100))

    for i, return_period in enumerate(storms_of_interest):
        vr = (df_storms[return_period] * area * 10).values
        ax2.plot(durations, vr, color=mpl.colormaps["cool"](mpl.colors.LogNorm(vmin=1, vmax=100)(return_period)))
        q = vr / durations * 1000 / 60
        ax1.plot(durations, q, color=mpl.colormaps["cool"](mpl.colors.LogNorm(vmin=1, vmax=100)(return_period)),
                 linestyle="dotted")

    ax2.set(ylabel="Niederschlagsmenge [m³]", ylim=[0, 12000],
            xlabel="Dauer [h]", xlim=[0, 1440], xticks=np.arange(0, 1620,
                                                                 180))  # , xticks=[0, 60, 120, 360, 720, 1440, 2880, 7200], xticklabels=[0, "1h", "2h", "6h", "12h", "24h", "2d", "5d"])
    ax2.xaxis.set_major_formatter(lambda x, pos: f"{x / 60:.0f}")
    ax1.set(ylabel="Niederschlagsintensität [L/s]", ylim=[0, 80], yticks=[0.25 * t for t in ax2.get_yticks()],
            xlabel="Dauer [h]")
    formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))

    fig.colorbar(sm, ax=ax2, ticks=[1, 3, 5, 10, 30, 100], format=formatter, label="Jährlichkeit")
    custom_lines = [mpl.lines.Line2D([0], [0], color="black", lw=1),
                    mpl.lines.Line2D([0], [0], color="black", lw=1, linestyle="dotted")]
    fig.legend(custom_lines, ["Niederschlagsvolumen", "Niederschlagsintensität"], loc="upper left",
               bbox_to_anchor=(0.1, 0.9))
    ax1.grid(zorder=0)
    if save:
        fig.savefig(odir / "niederschlagsvolumina.png")
    return None