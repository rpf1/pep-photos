"""Frozen geometry functions extracted verbatim from leaflet-v2 snapshot. See DECISIONS.md."""
import math
import numpy as np

def weighted_percentile(values, weights, percentile):
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    return float(np.interp((percentile / 100.0) * cdf[-1], cdf, values))


def circular_center(z, box_length):
    angle = 2.0 * np.pi * (z % box_length) / box_length
    s, c = np.mean(np.sin(angle)), np.mean(np.cos(angle))
    mean_angle = math.atan2(s, c)
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return box_length * mean_angle / (2.0 * np.pi)


def periodic_relative(z, center, box_length):
    return (z - center + 0.5 * box_length) % box_length - 0.5 * box_length


def make_modes(lx, ly, min_wavelength):
    max_mx = max(1, int(np.floor(lx / min_wavelength)))
    max_my = max(1, int(np.floor(ly / min_wavelength)))
    modes = []
    for mx in range(0, max_mx + 1):
        for my in range(-max_my, max_my + 1):
            if mx == 0 and my <= 0:
                continue
            spatial_frequency = np.sqrt((mx / lx) ** 2 + (my / ly) ** 2)
            if spatial_frequency == 0:
                continue
            wavelength = 1.0 / spatial_frequency
            if wavelength + 1e-12 < min_wavelength:
                continue
            modes.append((mx, my, 2.0*np.pi*mx/lx, 2.0*np.pi*my/ly, wavelength))
    return modes


def design_matrix(x, y, modes):
    cols = [np.ones_like(x, dtype=float)]
    for _, _, kx, ky, _ in modes:
        theta = kx*x + ky*y
        cols.extend([np.cos(theta), np.sin(theta)])
    return np.column_stack(cols)


def fit_surface(x, y, z, lx, ly, min_wavelength):
    modes = make_modes(lx, ly, min_wavelength)
    X = design_matrix(x, y, modes)
    coeff, _, rank, _ = np.linalg.lstsq(X, z, rcond=None)
    return {'coeff': coeff, 'modes': modes, 'rank': int(rank),
            'n_parameters': int(X.shape[1]), 'n_markers': int(len(z)),
            'rmse_nm': float(np.sqrt(np.mean((z-X@coeff)**2)))}


def evaluate_surface(surface, x, y):
    """Return h, hx, hy, hxx, hyy, hxy at matching array coordinates."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    coeff, modes = surface['coeff'], surface['modes']
    h = np.full_like(x, coeff[0], dtype=float)
    hx, hy, hxx, hyy, hxy = [np.zeros_like(x) for _ in range(5)]
    j = 1
    for _, _, kx, ky, _ in modes:
        a, b = coeff[j], coeff[j+1]
        j += 2
        theta = kx*x + ky*y
        ct, st = np.cos(theta), np.sin(theta)
        term = a*ct + b*st
        h += term
        hx += -a*kx*st + b*kx*ct
        hy += -a*ky*st + b*ky*ct
        hxx += -(kx**2)*term
        hyy += -(ky**2)*term
        hxy += -(kx*ky)*term
    return h, hx, hy, hxx, hyy, hxy


def validate_box(dimensions_A):
    d = np.asarray(dimensions_A, dtype=float)
    if d.size < 3 or not np.isfinite(d[:3]).all() or np.any(d[:3] <= 0):
        raise ValueError('Invalid/nonpositive box lengths')
    if d.size >= 6 and not np.allclose(d[3:6], 90.0, atol=1e-3, rtol=0):
        raise ValueError(f'Orthorhombic cell required; angles={d[3:6]}')
    return d[:3] / 10.0


def fit_leaflets(po4_positions_A, dimensions_A, min_wavelength_nm):
    """Original equal PO4 split, with explicit membership and added diagnostics."""
    if not np.isfinite(min_wavelength_nm) or min_wavelength_nm <= 0:
        raise ValueError('Minimum wavelength must be positive')
    xyz = np.asarray(po4_positions_A, dtype=float) / 10.0
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('PO4 coordinates must be finite N x 3')
    lx, ly, lz = validate_box(dimensions_A)
    x, y, z = xyz[:, 0] % lx, xyz[:, 1] % ly, xyz[:, 2] % lz
    if len(z) < 4 or len(z) % 2:
        raise ValueError(f'Even, nonempty PO4 set required; got {len(z)}')
    z_center = circular_center(z, lz)
    z_rel = periodic_relative(z, z_center, lz)
    order = np.argsort(z_rel)
    half = len(order)//2
    lower_idx, upper_idx = order[:half], order[half:]
    lower = fit_surface(x[lower_idx], y[lower_idx], z_rel[lower_idx], lx, ly, min_wavelength_nm)
    upper = fit_surface(x[upper_idx], y[upper_idx], z_rel[upper_idx], lx, ly, min_wavelength_nm)
    angle = 2*np.pi*z/lz
    return upper, lower, {
        'lx_nm': float(lx), 'ly_nm': float(ly), 'lz_nm': float(lz),
        'po4_count': len(z), 'leaflet_po4_count': half,
        'PO4_upper_count': half, 'PO4_lower_count': half,
        'z_center_nm': float(z_center),
        'z_center_resultant': float(np.hypot(np.mean(np.sin(angle)), np.mean(np.cos(angle)))),
        'median_leaflet_gap_nm': float(np.median(z_rel[upper_idx])-np.median(z_rel[lower_idx])),
        'po4_z_branch_margin_nm': float(lz/2-np.max(np.abs(z_rel))),
        'upper_indices': upper_idx, 'lower_indices': lower_idx,
    }


def normal_thickness(upper, lower, xg, yg, hm, hmx, hmy, hu, hl,
                     max_iter=10, tolerance_nm=0.01):
    """Original midplane-normal Newton intersections; unchanged numerical method."""
    norm = np.sqrt(1.0+hmx**2+hmy**2)
    nx, ny, nz = -hmx/norm, -hmy/norm, 1.0/norm
    vertical = hu-hl
    safe_nz = np.where(nz > 1e-8, nz, np.nan)
    su, sl = 0.5*vertical/safe_nz, -0.5*vertical/safe_nz
    for _ in range(max_iter):
        xu, yu = xg+su*nx, yg+su*ny
        h_u, hx_u, hy_u, _, _, _ = evaluate_surface(upper, xu, yu)
        fu, dfu = hm+su*nz-h_u, nz-hx_u*nx-hy_u*ny
        good = np.abs(dfu)>1e-8
        step = np.zeros_like(su)
        step[good] = fu[good]/dfu[good]
        su = su-step
        xl, yl = xg+sl*nx, yg+sl*ny
        h_l, hx_l, hy_l, _, _, _ = evaluate_surface(lower, xl, yl)
        fl, dfl = hm+sl*nz-h_l, nz-hx_l*nx-hy_l*ny
        good = np.abs(dfl)>1e-8
        step = np.zeros_like(sl)
        step[good] = fl[good]/dfl[good]
        sl = sl-step
    h_u, _, _, _, _, _ = evaluate_surface(upper, xg+su*nx, yg+su*ny)
    h_l, _, _, _, _, _ = evaluate_surface(lower, xg+sl*nx, yg+sl*ny)
    residual_u, residual_l = np.abs(hm+su*nz-h_u), np.abs(hm+sl*nz-h_l)
    thickness = su-sl
    valid = (np.isfinite(thickness) & np.isfinite(residual_u) & np.isfinite(residual_l)
             & (su>0) & (sl<0) & (thickness>0.5) & (thickness<8.0)
             & (residual_u<=tolerance_nm) & (residual_l<=tolerance_nm))
    return np.where(valid, thickness, np.nan), valid, residual_u, residual_l


def surface_differential(fields):
    h, hx, hy, hxx, hyy, hxy = fields
    J = np.sqrt(1+hx**2+hy**2)
    H = ((1+hy**2)*hxx-2*hx*hy*hxy+(1+hx**2)*hyy)/(2*(1+hx**2+hy**2)**1.5)
    return J, H


def build_surface_geometry(po4_positions_A, dimensions_A, min_wavelength_nm=4.,
                           grid_n=32, grid_offset=0.0):
    """Sole geometry builder. offset=0 retains morphology; .5 matches lipid bins."""
    if grid_n < 4 or int(grid_n) != grid_n or grid_offset not in (0., .5):
        raise ValueError('Grid must be integer >=4; offset must be 0 or 0.5')
    upper, lower, qc = fit_leaflets(po4_positions_A, dimensions_A, min_wavelength_nm)
    if qc['median_leaflet_gap_nm'] <= 1.0:
        raise RuntimeError(f"Ambiguous PO4 leaflet gap: {qc['median_leaflet_gap_nm']:.3f} nm")
    for name, surface in [('upper', upper), ('lower', lower)]:
        if surface['rank'] < surface['n_parameters']:
            raise RuntimeError(f'{name} surface fit is rank deficient')
        if any(abs(mode[0])*2 >= grid_n or abs(mode[1])*2 >= grid_n for mode in surface['modes']):
            raise RuntimeError('Grid cannot resolve fitted Fourier modes')
    lx, ly = qc['lx_nm'], qc['ly_nm']
    x = (np.arange(grid_n)+grid_offset)*lx/grid_n
    y = (np.arange(grid_n)+grid_offset)*ly/grid_n
    xg, yg = np.meshgrid(x, y, indexing='xy')
    u, l = evaluate_surface(upper, xg, yg), evaluate_surface(lower, xg, yg)
    m = tuple(.5*(a+b) for a,b in zip(u,l))
    geom = {'upper': upper, 'lower': lower, 'qc': qc, 'x_nm': x, 'y_nm': y,
            'xg': xg, 'yg': yg, 'cell_projected_area_nm2': lx*ly/grid_n**2,
            'projected_area_nm2': lx*ly, 'grid_offset': grid_offset}
    for name, values in [('upper',u), ('lower',l), ('midplane',m)]:
        J,H = surface_differential(values)
        geom[name+'_height_nm'] = values[0]
        geom[name+'_centered_height_nm'] = values[0]-np.mean(values[0])
        geom[name+'_jacobian'] = J
        geom[name+'_mean_curvature_nm_inv'] = H
        geom[name+'_abs_mean_curvature_nm_inv'] = np.abs(H)
    thickness, valid, ru, rl = normal_thickness(upper, lower, xg, yg, m[0], m[1], m[2], u[0], l[0])
    geom.update(normal_thickness_nm=thickness, thickness_valid_mask=valid,
                thickness_residual_upper_nm=ru, thickness_residual_lower_nm=rl)
    qc['surface_crossing_fraction'] = float(np.mean(u[0] <= l[0]))
    qc['minimum_vertical_gap_nm'] = float(np.min(u[0]-l[0]))
    qc['surface_z_branch_margin_nm'] = float(qc['lz_nm']/2-max(np.max(np.abs(u[0])),np.max(np.abs(l[0]))))
    return geom


def geometry_metrics(g, total_membrane_molecules):
    if total_membrane_molecules <= 0 or total_membrane_molecules % 2:
        raise ValueError('Canonical denominator requires positive even membrane-molecule total')
    area = g['projected_area_nm2']
    cell = g['cell_projected_area_nm2']
    result = {'projected_area_nm2': area,
              'projected_APL_A2': area/(total_membrane_molecules/2)*100}
    for name in ('upper','lower','midplane'):
        J = g[name+'_jacobian']
        H = g[name+'_mean_curvature_nm_inv']
        z = g[name+'_centered_height_nm']
        result[name+'_surface_area_nm2'] = float(np.sum(J)*cell)
        result[name+'_excess_area_fraction'] = result[name+'_surface_area_nm2']/area-1
        result[name+'_height_rms_nm'] = float(np.sqrt(np.mean(z**2)))
        result[name+'_H_rms_nm_inv'] = float(np.sqrt(np.sum(H**2*J)/np.sum(J)))
    result['surface_APL_proxy_A2'] = (result['upper_surface_area_nm2']+result['lower_surface_area_nm2'])/total_membrane_molecules*100
    result['height_rms_nm'] = result['midplane_height_rms_nm']
    result['H_rms_nm_inv'] = result['midplane_H_rms_nm_inv']
    z = g['midplane_centered_height_nm']
    result['height_p95_p05_nm'] = float(np.percentile(z,95)-np.percentile(z,5))
    J = g['midplane_jacobian']
    result['H_abs_p95_nm_inv'] = weighted_percentile(g['midplane_abs_mean_curvature_nm_inv'],J,95)
    weights = np.where(g['thickness_valid_mask'],J,0.)
    for tag,p in [('p05',5),('p25',25),('median',50),('p75',75),('p95',95)]:
        result['thickness_'+tag+'_nm'] = weighted_percentile(g['normal_thickness_nm'],weights,p)
    result['thickness_iqr_nm'] = result['thickness_p75_nm']-result['thickness_p25_nm']
    result['thickness_valid_fraction'] = float(np.mean(g['thickness_valid_mask']))
    for leaflet in ('upper','lower'):
        result['fit_rmse_'+leaflet+'_nm'] = g[leaflet]['rmse_nm']
        result['surface_parameters_'+leaflet] = g[leaflet]['n_parameters']
    for key in ('median_leaflet_gap_nm','po4_count','PO4_upper_count','PO4_lower_count',
                'z_center_nm','z_center_resultant','po4_z_branch_margin_nm',
                'surface_z_branch_margin_nm','surface_crossing_fraction','minimum_vertical_gap_nm'):
        result[key] = g['qc'][key]
    for source,stem,unit in [('surface_area_nm2','surface_area','_nm2'),
                             ('excess_area_fraction','excess_area',''),
                             ('height_rms_nm','height_rms','_nm'),
                             ('H_rms_nm_inv','H_rms','_nm_inv')]:
        diff = result['upper_'+source]-result['lower_'+source]
        result[stem+'_upper_minus_lower'+unit] = diff
        result[stem+'_abs_upper_minus_lower'+unit] = abs(diff)
    return result

