"""
精緻台灣地圖視覺化模組
===================
以 Cartopy + matplotlib 繪製台灣鄉鎮層級太陽能潛力熱力圖，
無需外部 shapefile，直接以 centroid_lat/lon 定位。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path


# ---------------------------------------------------------------------------
# Font / style setup
# ---------------------------------------------------------------------------
def _setup_font():
    """Try common CJK fonts."""
    for font_name in ['Noto Sans CJK TC', 'Microsoft JhengHei', 'PingFang TC',
                      'Heiti TC', 'Arial Unicode MS']:
        try:
            matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
def _make_priority_cmap():
    """Custom RdYlGn_r-like cmap but more print-friendly."""
    colors = [
        (0.90, 0.95, 0.85),   # very pale green
        (0.55, 0.80, 0.45),   # green
        (1.00, 0.90, 0.40),   # yellow
        (0.95, 0.60, 0.25),   # orange
        (0.85, 0.25, 0.25),   # red
    ]
    return LinearSegmentedColormap.from_list('priority', colors)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_merged_data(root='.'):
    root = Path(root)
    pred = pd.read_csv(root / 'outputs/results/transfer/taiwan_priority_ranking.csv')
    geo  = pd.read_csv(root / 'data/taiwan/氣候資料/taiwan_climate_annual.csv')
    geo = geo[['TOWNCODE', 'centroid_lat', 'centroid_lon']].copy()
    geo['TOWNCODE'] = geo['TOWNCODE'].astype(int)
    pred['TOWNCODE'] = pred['TOWNCODE'].astype(int)
    df = pred.merge(geo, on='TOWNCODE', how='left')
    return df


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def _idw_interpolate(lon, lat, values, grid_lon, grid_lat, power=2):
    """Inverse Distance Weighting interpolation."""
    points = np.column_stack([lon, lat])
    grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    dist = np.linalg.norm(grid_points[:, None, :] - points[None, :, :], axis=2)
    dist = np.where(dist < 1e-6, 1e-6, dist)
    weights = 1.0 / (dist ** power)
    weights_sum = weights.sum(axis=1)
    result = (weights @ values) / weights_sum
    return result.reshape(grid_lon.shape)


def _make_grid(lon, lat, n=300, margin=0.3):
    lon_min, lon_max = lon.min() - margin, lon.max() + margin
    lat_min, lat_max = lat.min() - margin, lat.max() + margin
    glon = np.linspace(lon_min, lon_max, n)
    glat = np.linspace(lat_min, lat_max, n)
    return np.meshgrid(glon, glat)


# ---------------------------------------------------------------------------
# Map drawing core
# ---------------------------------------------------------------------------
def _draw_taiwan_base(ax, extent_main=(119.2, 122.3, 21.7, 25.6),
                       draw_inset_islands=True):
    """Draw Taiwan base map with coastline, borders, and grid."""
    ax.set_extent(extent_main, crs=ccrs.PlateCarree())

    # Ocean / land
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dceefb', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f4f4f0', zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='#4a4a4a',
                   linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='#4a4a4a',
                   linewidth=0.6, linestyle=':', zorder=3)

    # Grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                      alpha=0.5, linestyle='--', zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'color': '#555555'}
    gl.ylabel_style = {'size': 8, 'color': '#555555'}

    # Inset for outlying islands (Penghu, Kinmen, Lienchiang)
    if draw_inset_islands:
        inset_ax = ax.inset_axes([0.02, 0.02, 0.28, 0.28],
                                  projection=ccrs.PlateCarree())
        inset_ax.set_extent([117.8, 120.6, 23.3, 26.5], crs=ccrs.PlateCarree())
        inset_ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#dceefb', zorder=0)
        inset_ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f4f4f0', zorder=1)
        inset_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#4a4a4a',
                             linewidth=0.6, zorder=3)
        inset_ax.set_title('離島', fontsize=8, pad=4)
        return inset_ax
    return None


# ---------------------------------------------------------------------------
# Main public figure functions
# ---------------------------------------------------------------------------
def fig_topsis_heatmap(df, out_path=None, dpi=300):
    """
    Figure 1: TOPSIS priority score heatmap over Taiwan.
    Uses IDW interpolation for smooth color field.
    """
    _setup_font()

    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    inset_ax = _draw_taiwan_base(ax)

    # Separate mainland + Penghu from outlying islands
    mainland_mask = ~df['COUNTYNAME'].isin(['連江縣', '金門縣'])
    df_main = df[mainland_mask].copy()
    df_isl  = df[~mainland_mask].copy()

    # Interpolate mainland scores
    lon = df_main['centroid_lon'].values
    lat = df_main['centroid_lat'].values
    val = df_main['topsis_score'].values

    glon, glat = _make_grid(lon, lat, n=400, margin=0.15)
    grid_val = _idw_interpolate(lon, lat, val, glon, glat, power=2.5)

    # Mask ocean (using elevation as proxy, or distance-based mask)
    # Simple distance-based mask: keep points within ~30 km of any township centroid
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([lon, lat]))
    pts = np.column_stack([glon.ravel(), glat.ravel()])
    dist, _ = tree.query(pts, k=1)
    mask = dist.reshape(glon.shape) > 0.22   # degrees (~24 km at 24°N)
    grid_val = np.ma.masked_where(mask, grid_val)

    # Plot heatmap
    cmap = plt.cm.RdYlGn_r
    im = ax.pcolormesh(glon, glat, grid_val, cmap=cmap, shading='gouraud',
                       vmin=0.45, vmax=0.85, transform=ccrs.PlateCarree(),
                       alpha=0.85, zorder=2)

    # Overlay township centroids (small circles)
    scatter = ax.scatter(lon, lat, c=val, s=35, cmap=cmap, vmin=0.45, vmax=0.85,
                         edgecolors='white', linewidths=0.4, zorder=5,
                         transform=ccrs.PlateCarree())

    # Outlying islands (just dots, no interpolation)
    if not df_isl.empty and inset_ax is not None:
        inset_ax.scatter(df_isl['centroid_lon'], df_isl['centroid_lat'],
                         c=df_isl['topsis_score'], cmap=cmap, vmin=0.45, vmax=0.85,
                         s=80, edgecolors='white', linewidths=0.5, zorder=5,
                         transform=ccrs.PlateCarree())

    # Annotate top 5 townships only (avoid clutter)
    top5 = df_main.nsmallest(5, 'priority_rank')
    for _, row in top5.iterrows():
        ax.annotate(
            f"{row['priority_rank']}. {row['COUNTYNAME']}{row['TOWNNAME']}",
            xy=(row['centroid_lon'], row['centroid_lat']),
            xytext=(8, 5), textcoords='offset points',
            fontsize=8, color='#1a1a1a', fontweight='bold', zorder=6,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
            arrowprops=dict(arrowstyle='-', color='#666666', lw=0.5)
        )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, aspect=30)
    cbar.set_label('TOPSIS Priority Score', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    # Title
    ax.set_title('太陽能推廣優先度 — TOPSIS 綜合排序\n'
                 '(368 鄉鎮市區：模型潛力 × 日照 × FIT × 收入)',
                 fontsize=16, fontweight='bold', pad=16)

    # Annotation box
    ax.text(0.98, 0.98,
            'Top 5:\n'
            '1. 高雄市鳥松區\n'
            '2. 屏東縣屏東市\n'
            '3. 屏東縣麟洛鄉\n'
            '4. 屏東縣潮州鎮\n'
            '5. 高雄市林園區',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#cccccc', alpha=0.92),
            zorder=7)

    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {out_path}')
    plt.close(fig)
    return fig


def fig_multi_panel(df, out_path=None, dpi=300):
    """
    Figure 2: 4-panel comparison map.
    (a) TOPSIS score  (b) Model combined score
    (c) Solar radiation  (d) Median household income
    """
    _setup_font()

    panels = [
        ('topsis_score', 'TOPSIS 綜合優先度', 'RdYlGn_r', 0.45, 0.85),
        ('combined_score', '模型部署傾向分數', 'viridis', 2.2, 2.95),
        ('daily_solar_radiation', '年均日照輻射量 (kWh/m²/day)', 'YlOrRd', 3.7, 4.8),
        ('avg_fit_rate', '平均躉購費率 (元/度)', 'Blues', 5.5, 6.6),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 18),
                              subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    mainland_mask = ~df['COUNTYNAME'].isin(['連江縣', '金門縣'])
    df_main = df[mainland_mask].copy()

    for idx, (col, title, cmap_name, vmin, vmax) in enumerate(panels):
        ax = axes[idx]
        ax.set_extent((119.2, 122.3, 21.7, 25.6), crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dceefb', zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f4f4f0', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='#4a4a4a',
                       linewidth=0.7, zorder=3)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                          alpha=0.4, linestyle='--', zorder=4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 7, 'color': '#555555'}
        gl.ylabel_style = {'size': 7, 'color': '#555555'}

        lon = df_main['centroid_lon'].values
        lat = df_main['centroid_lat'].values
        val = df_main[col].values

        # IDW interpolation
        glon, glat = _make_grid(lon, lat, n=350, margin=0.12)
        grid_val = _idw_interpolate(lon, lat, val, glon, glat, power=2.5)

        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([lon, lat]))
        dist, _ = tree.query(np.column_stack([glon.ravel(), glat.ravel()]), k=1)
        mask = dist.reshape(glon.shape) > 0.35
        grid_val = np.ma.masked_where(mask, grid_val)

        cmap = plt.cm.get_cmap(cmap_name)
        im = ax.pcolormesh(glon, glat, grid_val, cmap=cmap, shading='gouraud',
                           vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                           alpha=0.88, zorder=2)

        ax.scatter(lon, lat, c=val, s=25, cmap=cmap, vmin=vmin, vmax=vmax,
                   edgecolors='white', linewidths=0.3, zorder=5,
                   transform=ccrs.PlateCarree())

        cbar = plt.colorbar(im, ax=ax, shrink=0.65, pad=0.02, aspect=25)
        cbar.set_label(title.split('(')[0].strip(), fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {out_path}')
    plt.close(fig)
    return fig


def fig_county_summary_bars(df, out_path=None, dpi=300):
    """
    Figure 3: County-level horizontal bar chart with map-like color coding.
    """
    _setup_font()

    county = (
        df.groupby('COUNTYNAME')
        .agg(mean_topsis=('topsis_score', 'mean'),
             mean_combined=('combined_score', 'mean'),
             mean_solar=('daily_solar_radiation', 'mean'),
             mean_income=('median_household_income', 'mean'),
             best_rank=('priority_rank', 'min'))
        .sort_values('mean_topsis', ascending=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # Left: mean TOPSIS score
    ax = axes[0]
    colors = plt.cm.RdYlGn_r(
        (county['mean_topsis'] - county['mean_topsis'].min()) /
        (county['mean_topsis'].max() - county['mean_topsis'].min())
    )
    ax.barh(county.index, county['mean_topsis'], color=colors, edgecolor='white', height=0.7)
    ax.set_xlabel('Mean TOPSIS Score', fontsize=11, fontweight='bold')
    ax.set_title('縣市平均 TOPSIS 優先度', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 0.85)
    ax.grid(axis='x', alpha=0.3)
    for i, (idx, row) in enumerate(county.iterrows()):
        ax.text(row['mean_topsis'] + 0.01, i, f"{row['mean_topsis']:.3f}",
                va='center', fontsize=9)

    # Right: mean solar radiation
    ax = axes[1]
    county_s = county.sort_values('mean_solar', ascending=True)
    colors = plt.cm.YlOrRd(
        (county_s['mean_solar'] - county_s['mean_solar'].min()) /
        (county_s['mean_solar'].max() - county_s['mean_solar'].min())
    )
    ax.barh(county_s.index, county_s['mean_solar'], color=colors, edgecolor='white', height=0.7)
    ax.set_xlabel('Mean Daily Solar Radiation (kWh/m²/day)', fontsize=11, fontweight='bold')
    ax.set_title('縣市平均日照輻射量', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (idx, row) in enumerate(county_s.iterrows()):
        ax.text(row['mean_solar'] + 0.02, i, f"{row['mean_solar']:.2f}",
                va='center', fontsize=9)

    plt.suptitle('台灣各縣市太陽能潛力指標總覽', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {out_path}')
    plt.close(fig)
    return fig


def fig_top20_detail(df, out_path=None, dpi=300):
    """
    Figure 4: Top-20 priority townships detail card with map inset.
    """
    _setup_font()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1])

    # --- Top-left: map with top 20 highlighted ---
    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax_map.set_extent((119.2, 122.3, 21.7, 25.6), crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dceefb', zorder=0)
    ax_map.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f4f4f0', zorder=1)
    ax_map.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='#4a4a4a',
                       linewidth=0.8, zorder=3)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                          alpha=0.5, linestyle='--', zorder=4)
    gl.top_labels = False
    gl.right_labels = False

    mainland_mask = ~df['COUNTYNAME'].isin(['連江縣', '金門縣'])
    df_main = df[mainland_mask].copy()

    # Background: all townships in pale gray
    ax_map.scatter(df_main['centroid_lon'], df_main['centroid_lat'],
                   c='#cccccc', s=20, zorder=2, transform=ccrs.PlateCarree())

    # Top 20 highlighted
    top20 = df_main.nsmallest(20, 'priority_rank')
    cmap = plt.cm.RdYlGn_r
    sc = ax_map.scatter(top20['centroid_lon'], top20['centroid_lat'],
                        c=top20['topsis_score'], s=180, cmap=cmap,
                        vmin=0.45, vmax=0.85, edgecolors='black',
                        linewidths=0.8, zorder=5, transform=ccrs.PlateCarree())

    # Number labels
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        ax_map.annotate(
            str(rank),
            xy=(row['centroid_lon'], row['centroid_lat']),
            fontsize=7, fontweight='bold', color='white', ha='center', va='center',
            zorder=6, transform=ccrs.PlateCarree()
        )
        ax_map.annotate(
            f"{row['COUNTYNAME']}{row['TOWNNAME']}",
            xy=(row['centroid_lon'], row['centroid_lat']),
            xytext=(8, 0), textcoords='offset points',
            fontsize=7.5, color='#222222', zorder=7,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
        )

    plt.colorbar(sc, ax=ax_map, shrink=0.5, pad=0.02, label='TOPSIS Score')
    ax_map.set_title('Top 20 優先鄉鎮市區分布', fontsize=13, fontweight='bold', pad=10)

    # --- Top-right: radar / spider chart placeholder (4 metrics) ---
    ax_radar = fig.add_subplot(gs[0, 1])
    metrics = ['combined_score', 'daily_solar_radiation', 'avg_fit_rate', 'median_household_income']
    metric_labels = ['部署傾向', '日照輻射', 'FIT費率', '家戶收入']
    # Normalize to 0-1 for display
    norms = {}
    for m in metrics:
        vals = df_main[m].values
        norms[m] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

    # Show average of top 20 vs all
    all_avg = [norms[m].mean() for m in metrics]
    top20_idx = top20.index
    top20_avg = [norms[m][top20_idx].mean() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    ax_radar.barh(x + width/2, all_avg, width, label='全體平均', color='#aaaaaa')
    ax_radar.barh(x - width/2, top20_avg, width, label='Top 20 平均', color='#e74c3c')
    ax_radar.set_yticks(x)
    ax_radar.set_yticklabels(metric_labels)
    ax_radar.set_xlim(0, 1)
    ax_radar.set_xlabel('正規化分數 (0–1)', fontsize=10)
    ax_radar.set_title('Top 20 vs 全體 — 屬性比較', fontsize=13, fontweight='bold')
    ax_radar.legend(loc='lower right')
    ax_radar.grid(axis='x', alpha=0.3)

    # --- Bottom-right: top 20 table ---
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis('off')

    table_data = []
    for _, row in top20.iterrows():
        table_data.append([
            f"{row['priority_rank']}",
            f"{row['COUNTYNAME']}{row['TOWNNAME']}",
            f"{row['topsis_score']:.3f}",
            f"{row['combined_score']:.3f}",
            f"{row['daily_solar_radiation']:.2f}",
            f"{row['avg_fit_rate']:.2f}",
            f"{row['median_household_income']:.0f}",
        ])

    col_labels = ['排名', '鄉鎮市區', 'TOPSIS', '部署傾向', '日照', 'FIT', '收入']
    table = ax_table.table(
        cellText=table_data, colLabels=col_labels,
        loc='center', cellLoc='center',
        colColours=['#4472C4'] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)
    for i in range(len(col_labels)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_facecolor('#4472C4')
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f4fa')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    # 不用 ax_table.set_title，避免與 suptitle 重疊
    plt.suptitle('太陽能推廣優先序 — Top 20 深度分析', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {out_path}')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# CLI / direct execution
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    df = load_merged_data()
    root = Path('.') / 'outputs/figures/transfer'
    fig_topsis_heatmap(df, out_path=root / 'map_01_topsis_heatmap.png', dpi=300)
    fig_multi_panel(df, out_path=root / 'map_02_four_panel.png', dpi=300)
    fig_county_summary_bars(df, out_path=root / 'map_03_county_bars.png', dpi=300)
    fig_top20_detail(df, out_path=root / 'map_04_top20_detail.png', dpi=300)
