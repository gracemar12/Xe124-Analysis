#This script takes the output file from xe124_chi_squared_fit.py and plots the 2D contour plot of the chi^2 surface for any two parameters of interest (valid ones are listed in the script)
# It should be run after xe124_chi_squared_fit.py has been run and the output file has been generated.
#This is useful for visualizing how each parameter actually affected the MINUIT fit, and can be run on any parameters that seemed less accurate and should be investigated
#These plots are also useful for the published paper, especially if there are any paramters that have a stronger pull on the IYR fit

#Author: Grace Martin :)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from iminuit import Minuit
import json
import os

_dir = os.path.dirname(os.path.abspath(__file__))

# run name: must match RUN_NAME used in xe124_nuisance_fit.py
RUN_NAME = 'xe124_run1'

# scan settings: change these two lines to pick any pair of parameters
# valid options: 'IYR', 'eps', 'Pgm', 'Pgg', 'Thm', 'Thg',
#                'sg_burn', 'sm_burn', 'sig124', 'phi', 'N124_0'
PARAM_X  = 'IYR'
PARAM_Y  = 'eps'
N_GRID   = 20    # grid points per axis (N_GRID x N_GRID minimizations)
N_SIGMA  = 2.5   # how many sigma to scan around best fit on each axis

# axis labels for plot
AXIS_LABELS = {
    'IYR':     'Isomeric Yield Ratio (IYR)',
    'eps':     r'Efficiency ratio $\varepsilon$',
    'Pgm':     r'$P_\gamma$(111.3 keV, Xe-125m)',
    'Pgg':     r'$P_\gamma$(188.4 keV, Xe-125g)',
    'Thm':     r'$T_{1/2}$(Xe-125m) [s]',
    'Thg':     r'$T_{1/2}$(Xe-125g) [s]',
    'sg_burn': r'$\sigma_\mathrm{burn}$(Xe-125g) [cm$^2$]',
    'sm_burn': r'$\sigma_\mathrm{burn}$(Xe-125m) [cm$^2$]',
    'sig124':  r'$\sigma_{124}$ (n,$\gamma$) [cm$^2$]',
    'phi':     r'$\phi$ [n/cm$^2$/s]',
    'N124_0':  r'$N_{124,0}$ (initial atoms)',
}

# parameters (must match xe124_nuisance_fit.py) # Needs updating once beam schedule is known
BEAM_SCHEDULE = [
    # (t_on, t_off),
]

KEYS  = ['eps', 'Pgm', 'Pgg', 'Thm', 'Thg', 'sg_burn', 'sm_burn', 'sig124', 'phi', 'N124_0']
FIXED = {'N124_0'}

NOM = {
    'eps':     1.0,
    'Pgm':     0.602,
    'Pgg':     0.538,
    'Thm':     56.9,
    'Thg':     60732.0,
    'sg_burn': 5.60e-22,
    'sm_burn': 3.79e-24,
    'sig124':  1.65e-22,
    'phi':     4.084e12,
    'N124_0':  1.0,
}

REL = {
    'eps':     0.05,
    'Pgm':     0.01,
    'Pgg':     0.01,
    'Thm':     0.9   / 56.9,
    'Thg':     288.0 / 60732.0,
    'sg_burn': 0.30,
    'sm_burn': 0.50,
    'sig124':  11.0  / 165.0,
    'phi':     0.0372,
    'N124_0':  0.05,
}

N124_0_val = NOM['N124_0']

# ode model
def _ode_irr(t, y, alpha, km, kg, Sm_r, Sg_r, lm, lg):
    N124, Nm, Ng = y
    return [
        -alpha * N124,
        Sm_r * N124 - km * Nm,
        Sg_r * N124 + lm * Nm - kg * Ng,
    ]

def _ode_decay(t, y, lm, lg):
    Nm, Ng = y
    return [-lm * Nm, lm * Nm - lg * Ng]

def integrate_schedule(beam_schedule, N124_0, lm, lg, alpha, km, kg, Sm_r, Sg_r):
    y3 = np.array([N124_0, 0.0, 0.0])
    t_prev = 0.0
    for t_on, t_off in beam_schedule:
        if t_on > t_prev:
            sol = solve_ivp(_ode_decay, [t_prev, t_on], y3[1:],
                            args=(lm, lg), method='RK45', rtol=1e-9, atol=1e-14)
            y3[1], y3[2] = sol.y[:, -1]
            t_prev = t_on
        sol = solve_ivp(_ode_irr, [t_on, t_off], y3,
                        args=(alpha, km, kg, Sm_r, Sg_r, lm, lg),
                        method='RK45', rtol=1e-9, atol=1e-12)
        y3 = sol.y[:, -1]
        t_prev = t_off
    return y3

def model(t_query, IYR, eps, Pgm, Pgg, Thm, Thg, sg_burn, sm_burn, sig124, phi, N124_0,
          beam_schedule=BEAM_SCHEDULE):
    lm    = np.log(2) / Thm
    lg    = np.log(2) / Thg
    alpha = phi * sig124
    km    = lm + phi * sm_burn
    kg    = lg + phi * sg_burn
    Sm_r  = phi * IYR       * sig124
    Sg_r  = phi * (1 - IYR) * sig124
    _, Nm_end, Ng_end = integrate_schedule(
        beam_schedule, N124_0, lm, lg, alpha, km, kg, Sm_r, Sg_r)
    t_final_off = beam_schedule[-1][1]
    t_q         = np.atleast_1d(np.asarray(t_query, dtype=float))
    tau_q       = t_q - t_final_off
    sol = solve_ivp(_ode_decay, [0.0, float(tau_q.max())], [Nm_end, Ng_end],
                    args=(lm, lg), method='RK45', rtol=1e-9, atol=1e-14,
                    dense_output=True)
    pop    = sol.sol(tau_q)
    Nm, Ng = pop[0], pop[1]
    return eps * (lm * Nm * Pgm) / (lg * Ng * Pgg)

# load data
CSV_PATH = os.path.join(_dir, 'xe124_data.csv') #change directory to whatever the name of your data file is
df    = pd.read_csv(CSV_PATH)
t_arr = df['time_s'].values.astype(float)
r_arr = df['ratio'].values.astype(float)
u_arr = df['frac_unc'].values.astype(float)
mask  = (u_arr < 0.30) & np.isfinite(r_arr) & (r_arr > 0)
t_m   = t_arr[mask]
r_m   = r_arr[mask]
s_m   = r_m * u_arr[mask]

# load best-fit values from main fit
BF_PATH = os.path.join(_dir, f'{RUN_NAME}_bestfit.json')  #make sure RUN_NAME above matches the one used in xe124_nuisance_fit.py
with open(BF_PATH) as f:
    bf = json.load(f)
print(f'loaded best fit: IYR = {bf["IYR"]:.6f}, fval = {bf["fval"]:.2f}')

# chi squared (same as main script)
def chi2(IYR, **x_vals):
    p = {k: x_vals[f'x_{k}'] * NOM[k] for k in KEYS}
    prediction = model(t_m, IYR,
                       p['eps'], p['Pgm'], p['Pgg'], p['Thm'], p['Thg'],
                       p['sg_burn'], p['sm_burn'], p['sig124'], p['phi'], p['N124_0'])
    data_term = np.sum(((r_m - prediction) / s_m) ** 2)
    penalty   = sum(((x_vals[f'x_{k}'] - 1.0) / REL[k]) ** 2
                    for k in KEYS if k not in FIXED)
    return data_term + penalty

# scan range helper
def scan_range(param):
    if param == 'IYR':
        center = bf['IYR']
        sigma  = 0.5 * (bf['IYR_err_lo'] + bf['IYR_err_hi'])
    else:
        center = bf[f'x_{param}'] * NOM[param]
        sigma  = REL[param] * NOM[param]
    return np.linspace(center - N_SIGMA * sigma, center + N_SIGMA * sigma, N_GRID)

def to_scale(param, val):
    if param == 'IYR':
        return val
    return val / NOM[param]

# 2d grid scan
x_vals = scan_range(PARAM_X)
y_vals = scan_range(PARAM_Y)
chi2_grid = np.zeros((N_GRID, N_GRID))

print(f'scanning {N_GRID}x{N_GRID} grid for ({PARAM_X}, {PARAM_Y})...')

for i, vx in enumerate(x_vals):
    for j, vy in enumerate(y_vals):

        x_start = {f'x_{k}': bf[f'x_{k}'] for k in KEYS}
        iyr_start = bf['IYR']

        if PARAM_X == 'IYR':
            iyr_start = vx
        else:
            x_start[f'x_{PARAM_X}'] = to_scale(PARAM_X, vx)

        if PARAM_Y == 'IYR':
            iyr_start = vy
        else:
            x_start[f'x_{PARAM_Y}'] = to_scale(PARAM_Y, vy)

        ms = Minuit(chi2, IYR=iyr_start, **x_start)
        ms.errordef = Minuit.LEAST_SQUARES
        for k in KEYS:
            ms.errors[f'x_{k}'] = REL[k]
            if k in FIXED:
                ms.fixed[f'x_{k}'] = True
        ms.errors['IYR']       = 1e-3
        ms.limits['IYR']       = (0, 1)
        ms.limits['x_sg_burn'] = (0, None)
        ms.limits['x_sm_burn'] = (0, None)

        if PARAM_X == 'IYR':
            ms.fixed['IYR'] = True
        else:
            ms.fixed[f'x_{PARAM_X}'] = True

        if PARAM_Y == 'IYR':
            ms.fixed['IYR'] = True
        else:
            ms.fixed[f'x_{PARAM_Y}'] = True

        ms.migrad()
        chi2_grid[j, i] = ms.fval

    print(f'  column {i+1}/{N_GRID} done')

dchi2_grid = chi2_grid - bf['fval']

# best-fit point in physical units
bf_x = bf['IYR'] if PARAM_X == 'IYR' else bf[f'x_{PARAM_X}'] * NOM[PARAM_X]
bf_y = bf['IYR'] if PARAM_Y == 'IYR' else bf[f'x_{PARAM_Y}'] * NOM[PARAM_Y]

# plot
fig, ax = plt.subplots(figsize=(6, 5))

levels_fill    = [0, 1.0, 2.30, 4.0, 6.18, 9.0]
levels_contour = [1.0, 2.30, 4.0, 6.18]
level_labels   = {1.0: r'$1\sigma$', 4.0: r'$2\sigma$ ($\Delta\chi^2=4$)'}

cf = ax.contourf(x_vals, y_vals, dchi2_grid,
                 levels=levels_fill, cmap='Blues_r', extend='max')
cs = ax.contour(x_vals, y_vals, dchi2_grid,
                levels=levels_contour, colors='black', linewidths=0.8)
ax.clabel(cs, levels=[1.0, 4.0], fmt=level_labels, fontsize=8, inline=True)

ax.plot(bf_x, bf_y, '+', color='red', ms=12, mew=2, label='Best fit')

cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r'$\Delta\chi^2$', fontsize=10)
cbar.set_ticks([0, 1.0, 2.3, 4.0, 6.18, 9.0])

ax.set_xlabel(AXIS_LABELS.get(PARAM_X, PARAM_X), fontsize=10)
ax.set_ylabel(AXIS_LABELS.get(PARAM_Y, PARAM_Y), fontsize=10)
ax.set_title(
    rf'$\Delta\chi^2$ surface — {AXIS_LABELS.get(PARAM_X, PARAM_X)} vs {AXIS_LABELS.get(PARAM_Y, PARAM_Y)}'
    '\n(minimised over all other nuisance parameters)',
    fontsize=10
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25, lw=0.5)

outname = f'{RUN_NAME}_contour_{PARAM_X}_vs_{PARAM_Y}'
plt.savefig(os.path.join(_dir, outname + '.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(_dir, outname + '.png'), dpi=300, bbox_inches='tight')
print(f'saved {outname}.pdf / .png')
plt.show()

