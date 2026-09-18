from scipy.interpolate import interp1d
import numpy as np

def find_curve_intersection(
    x,
    y1, 
    y2,
    atol=1e-3,
    n_points=1000,
):
    """
    Find the inlet-temperature value where two curves intersect.

    Parameters
    ----------
    x : array-like
        x-axis values.
    y1 : array-like
        First curve values.
    y2 : array-like
        Second curve values.
    atol : float
        Tolerance for zero crossing.
    n_points : int
        Number of interpolation points to sample.

    Returns
    -------
    float or None
        Intersection x-value, or None if no clear crossing is found.
    """

    # Scale to [0, 1] so comparison is dimensionless
    y1_min, y1_max = y1.min(), y1.max()
    y2_min, y2_max = y2.min(), y2.max()

    y1_scaled = (y1 - y1_min) / (y1_max - y1_min) if (y1_max - y1_min) != 0 else np.zeros_like(y1)
    y2_scaled = (y2 - y2_min) / (y2_max - y2_min) if (y2_max - y2_min) != 0 else np.zeros_like(y2)

    f1 = interp1d(x, y1_scaled, kind='linear', fill_value='extrapolate')
    f2 = interp1d(x, y2_scaled, kind='linear', fill_value='extrapolate')

    x_vals = np.linspace(x.min(), x.max(), n_points)
    y_diff = f1(x_vals) - f2(x_vals)

    intersections = x_vals[np.isclose(y_diff, 0, atol=atol)]

    return intersections[0] if len(intersections) > 0 else None