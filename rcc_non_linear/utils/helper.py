import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def interpolate_z(df, x, y, method="linear", fallback="nearest"):
    '''Interpolate z for two independent variables x & y'''
    x_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values
    z_data = df.iloc[:, 2].values

    points = np.column_stack((x_data, y_data))

    xq = np.atleast_1d(x)
    yq = np.atleast_1d(y)
    query_points = np.column_stack((xq, yq))

    # Primary interpolation
    z_interp = griddata(points, z_data, query_points, method=method)

    # Fallback for points outside convex hull
    if fallback is not None:
        mask = np.isnan(z_interp)
        if np.any(mask):
            z_interp[mask] = griddata(
                points, z_data, query_points[mask], method=fallback
            )

    return z_interp[0].item() if z_interp.size == 1 else z_interp.item()


def caltrans_bilinear(df, yield_index):
    """
    Generate Caltrans-style bilinear idealization.
    - First branch: elastic slope up to yield, extended linearly until idealized yield
    - Second branch: horizontal (constant strength)
    - Area equality enforced beyond yield point.
    """
    cols = df.columns
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()

    # Extract key points
    Dy, Vy = x[yield_index], y[yield_index]
    Du, Vu = x[-1], np.max(y)
    Ke = Vy / Dy  # elastic slope

    # Actual area beyond yield
    area_actual = np.trapz(y[yield_index:], x[yield_index:])

    # Function to compute area under bilinear idealized curve (beyond yield)
    def bilinear_area(Dp):
        # beyond yield: from Dy → Dp (still increasing linearly), then horizontal at Vp
        Vp = Vy + Ke * (Dp - Dy)
        area1 = 0.5 * (Vp + Vy) * (Dp - Dy)  # trapezoid under rising branch
        area2 = Vp * (Du - Dp)               # rectangular plateau
        return area1 + area2, Vp

    # Find Dp (plastic hinge start) such that areas match
    Dp_low, Dp_high = Dy, Du
    for _ in range(1000):
        Dp_mid = 0.5 * (Dp_low + Dp_high)
        area_mid, _ = bilinear_area(Dp_mid)
        if abs(area_mid - area_actual) < 1e-3:
            break
        if area_mid > area_actual:
            Dp_high = Dp_mid
        else:
            Dp_low = Dp_mid

    Dp = Dp_mid
    _, Vp = bilinear_area(Dp)

    # Construct bilinear coordinates
    x_bi = [0, Dy, Dp, Du]
    y_bi = [0, Vy, Vp, Vp]
    bilinear_df = pd.DataFrame({cols[0]: x_bi, cols[1]: y_bi})

    return bilinear_df

def get_disp_mPhi(model):
    curvs = model.df_m_phi_idealized["curvatures"]
    curv_yi, curv_u = curvs[2].item(), curvs[3].item()
    disp_yi = curv_yi * model.L**2 /3
    disp_p = (curv_u-curv_yi)*model.lp*(model.L-model.lp/2)
    disp_u = disp_yi + disp_p
    return disp_yi, disp_u

def plot_response(
    df,
    x_label=None,
    y_label=None,
    title="Response Curve",
    grid=True,
    show=True,
    figsize=(8, 5),
):
    """
    Plots a 2D response curve using matplotlib.
    """
    import matplotlib.pyplot as plt

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    x_label = x_label or str(df.columns[0])
    y_label = y_label or str(df.columns[1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linewidth=2, color="#1f77b4")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_response_multi(
    dfs,
    names=None,
    colors=None,
    x_label=None,
    y_label=None,
    title="Response Curve",
    grid=True,
    show=True,
    figsize=(8, 5),
):
    """
    Plots multiple response curves on the same matplotlib axes for comparison.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for i, df in enumerate(dfs):
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        label = names[i] if names and i < len(names) else f"Trace {i+1}"
        color = colors[i] if colors and i < len(colors) else None

        ax.plot(x, y, label=label, linewidth=2, color=color)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x_label or str(dfs[0].columns[0]), fontsize=11)
    ax.set_ylabel(y_label or str(dfs[0].columns[1]), fontsize=11)
    ax.legend(frameon=True)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def extract_backbone_curve(df, disp_col=None, force_col=None, envelope="both"):
    """
    Extracts the cyclic backbone (envelope) curve from hysteretic force-displacement data.
    Finds peak response points for each cycle excursion and forms a monotonic backbone curve.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing displacement and force columns.
    disp_col : str, optional
        Name of the displacement column. Defaults to 'displacements' or column 0.
    force_col : str, optional
        Name of the force column. Defaults to 'forces' or column 1.
    envelope : {'both', 'positive', 'negative', 'average'}, default='both'
        - 'both': returns combined positive and negative backbone [ -u_max ... 0 ... +u_max ]
        - 'positive': returns only positive displacement backbone [0 ... +u_max]
        - 'negative': returns only negative displacement backbone [0 ... -u_max]
        - 'average': returns average of positive and mirrored negative backbones

    Returns
    -------
    backbone_df : pandas.DataFrame
        DataFrame with extracted backbone curve points.
    """
    if disp_col is None:
        disp_col = 'displacements' if 'displacements' in df.columns else df.columns[0]
    if force_col is None:
        force_col = 'forces' if 'forces' in df.columns else df.columns[1]

    d = df[disp_col].to_numpy()
    f = df[force_col].to_numpy()

    # 1. Identify local extrema (peaks and valleys) in displacement
    diffs = np.diff(d)
    direction = np.sign(diffs)
    # Filter out zero increments
    non_zero_idx = np.where(direction != 0)[0]
    if len(non_zero_idx) < 2:
        return pd.DataFrame({disp_col: [0.0], force_col: [0.0]})

    filtered_dir = direction[non_zero_idx]
    dir_changes = np.where(filtered_dir[:-1] != filtered_dir[1:])[0]
    extrema_idx = non_zero_idx[dir_changes + 1]

    # Include first and last points if relevant
    candidate_indices = np.unique(np.concatenate(([0], extrema_idx, [len(d) - 1])))

    # Separate into positive and negative peaks
    pos_points = [(0.0, 0.0)]
    neg_points = [(0.0, 0.0)]

    for idx in candidate_indices:
        disp_val = d[idx]
        force_val = f[idx]

        if disp_val > 1e-5:
            # Positive excursion peak
            pos_points.append((disp_val, force_val))
        elif disp_val < -1e-5:
            # Negative excursion valley
            neg_points.append((disp_val, force_val))

    # Sort and filter for sequential upper envelope (monotonic displacement increase)
    pos_points = sorted(pos_points, key=lambda p: p[0])
    filtered_pos = [(0.0, 0.0)]
    max_disp_so_far = 0.0
    for p_d, p_f in pos_points:
        if p_d > max_disp_so_far + 1e-4:
            filtered_pos.append((p_d, p_f))
            max_disp_so_far = p_d

    # Negative points (sorted from most negative to 0)
    neg_points = sorted(neg_points, key=lambda p: p[0])
    filtered_neg = []
    min_disp_so_far = 0.0
    # Process from 0 downwards
    for p_d, p_f in reversed(neg_points):
        if p_d < min_disp_so_far - 1e-4:
            filtered_neg.insert(0, (p_d, p_f))
            min_disp_so_far = p_d
    filtered_neg.append((0.0, 0.0))

    if envelope == 'positive':
        res_d, res_f = zip(*filtered_pos)
    elif envelope == 'negative':
        res_d, res_f = zip(*filtered_neg)
    elif envelope == 'average':
        # Interpolate positive and absolute negative to average curve
        pos_arr = np.array(filtered_pos)
        neg_arr = np.array([(abs(x), abs(y)) for x, y in filtered_neg])
        neg_arr = neg_arr[np.argsort(neg_arr[:, 0])]
        
        max_d = min(pos_arr[-1, 0], neg_arr[-1, 0]) if (len(pos_arr) > 1 and len(neg_arr) > 1) else pos_arr[-1, 0]
        eval_d = np.linspace(0, max_d, max(len(filtered_pos), len(filtered_neg)))
        f_pos_interp = np.interp(eval_d, pos_arr[:, 0], pos_arr[:, 1])
        f_neg_interp = np.interp(eval_d, neg_arr[:, 0], neg_arr[:, 1])
        avg_f = 0.5 * (f_pos_interp + f_neg_interp)
        res_d, res_f = eval_d, avg_f
    else:  # 'both'
        # Combine negative (ascending) and positive (ascending)
        combined = filtered_neg[:-1] + filtered_pos
        res_d, res_f = zip(*combined)

    backbone_df = pd.DataFrame({
        disp_col: list(res_d),
        force_col: list(res_f)
    })
    if 'drift %' in df.columns or ('displacements' in df.columns and hasattr(df, 'L')):
        if 'drift %' in df.columns:
            # maintain drift %
            backbone_df['drift %'] = backbone_df[disp_col] * (df['drift %'].iloc[-1] / df[disp_col].iloc[-1]) if df[disp_col].iloc[-1] != 0 else 0.0

    return backbone_df


