#This script models the abundance of Xe-124, Xe-125m, and Xe-125g during and after neutron irradiation using a numerical model
#It then loads the data in the form of a cvs file (make sure timestamps are accurate) and fits the model to the data and performs a chi^2 analysis
#It should report the best fit IYR + uncertainty (which was calculted using nuisance parameters) + the chi^2 of the best fit
#It should also print a table that shows the deviation at best fit for every nuisance parameter, as well as plot the activity ratio fit to the model, and plot the chi^2 profile scan
#last thing it does is save an output file which can be plugged into another script to make a 2D contour plots

#author: Grace Martin :)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from iminuit import Minuit
import os
import json

_dir = os.path.dirname(os.path.abspath(__file__))

# run name: change this for each new run to avoid overwriting output files
RUN_NAME = 'xe124_run1'

# beam schedule: list of (t_on, t_off) in seconds from start of experiment
BEAM_SCHEDULE = [
    # (t_on, t_off),
    # (t_on, t_off),
]

# parameters
# FIXED: parameters held constant during the fit (not floated by minuit)
# to float a fixed parameter, remove it from FIXED
# to fix a free parameter, add its key to FIXED
KEYS  = ['eps', 'Pgm', 'Pgg', 'Thm', 'Thg', 'sg_burn', 'sm_burn', 'sig124', 'phi', 'N124_0']
FIXED = {'N124_0'}

NOM = {
    'eps':     1.079980584,        # efficiency ratio (metastable gamma / ground gamma)
    'Pgm':     0.199,      # gamma intensity at 141.4 keV (xe-125m)
    'Pgg':     0.538,      # gamma intensity at 188.4 keV (xe-125g)
    'Thm':     56.9,       # half life of xe-125m [s]
    'Thg':     60732.0,    # half life of xe-125g [s] (16.87 h)
    'sg_burn': 5.60e-22,   # neutron capture cross section of xe-125g [cm2] (560 barn)
    'sm_burn': 3.79e-24,   # neutron capture cross section of xe-125m [cm2] (3.79 barn)
    'sig124':  1.65e-22,   # total (n,g) cross section of xe-124 [cm2] (165 barn)
    'phi':     4.084e12,   # total neutron flux [n/cm2/s]
    'N124_0':  1.0,        # initial xe-124 atom count (placeholder — update from sample)
}

REL = {
    'eps':     0.0174267911,
    'Pgm':     0.0242,
    'Pgg':     0.01,
    'Thm':     0.9   / 56.9,
    'Thg':     288.0 / 60732.0,
    'sg_burn': 0.30,
    'sm_burn': 0.50,
    'sig124':  11.0  / 165.0,
    'phi':     0.0372,
    'N124_0':  0.05,       # 5% placeholder — update from sample mass uncertainty
}

# ode model with beam schedule support
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
            sol = solve_ivp(
                _ode_decay, [t_prev, t_on], y3[1:],
                args=(lm, lg),
                method='RK45', rtol=1e-9, atol=1e-14,
            )
            y3[1], y3[2] = sol.y[:, -1]
            t_prev = t_on
        sol = solve_ivp(
            _ode_irr, [t_on, t_off], y3,
            args=(alpha, km, kg, Sm_r, Sg_r, lm, lg),
            method='RK45', rtol=1e-9, atol=1e-12,
        )
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
        beam_schedule, N124_0, lm, lg, alpha, km, kg, Sm_r, Sg_r
    )

    t_final_off = beam_schedule[-1][1]
    t_q         = np.atleast_1d(np.asarray(t_query, dtype=float))
    tau_q       = t_q - t_final_off
    tau_max     = float(tau_q.max())

    sol = solve_ivp(
        _ode_decay, [0.0, tau_max], [Nm_end, Ng_end],
        args=(lm, lg),
        method='RK45', rtol=1e-9, atol=1e-14,
        dense_output=True,
    )
    pop    = sol.sol(tau_q)
    Nm, Ng = pop[0], pop[1]

    return eps * (lm * Nm * Pgm) / (lg * Ng * Pgg)

# load data from csv
# columns: time_s (seconds from experiment start), ratio (Am/Ag), frac_unc
CSV_PATH = os.path.join(_dir, 'xe124_data.csv')       #change directory to whatever the name of your data file is
df    = pd.read_csv(CSV_PATH)
t_arr = df['time_s'].values.astype(float)
r_arr = df['ratio'].values.astype(float)
u_arr = df['frac_unc'].values.astype(float)

mask = (u_arr < 0.30) & np.isfinite(r_arr) & (r_arr > 0)
t_m  = t_arr[mask]
r_m  = r_arr[mask]
s_m  = r_m * u_arr[mask]
print(f'data points: {len(t_m)}')

# chi squared
def chi2(IYR, **x_vals):
    p = {k: x_vals[f'x_{k}'] * NOM[k] for k in KEYS}
    prediction = model(t_m, IYR,
                       p['eps'], p['Pgm'], p['Pgg'], p['Thm'], p['Thg'],
                       p['sg_burn'], p['sm_burn'], p['sig124'], p['phi'], p['N124_0'])
    data_term = np.sum(((r_m - prediction) / s_m) ** 2)
    penalty   = sum(((x_vals[f'x_{k}'] - 1.0) / REL[k]) ** 2
                    for k in KEYS if k not in FIXED)
    return data_term + penalty

# fit
IYR_INIT = 0.05
mfit = Minuit(chi2, IYR=IYR_INIT, **{f'x_{k}': 1.0 for k in KEYS})
mfit.errordef = Minuit.LEAST_SQUARES

for k in KEYS:
    mfit.errors[f'x_{k}'] = REL[k]
    if k in FIXED:
        mfit.fixed[f'x_{k}'] = True

mfit.errors['IYR']       = 1e-3
mfit.limits['IYR']       = (0, 1)
mfit.limits['x_sg_burn'] = (0, None)
mfit.limits['x_sm_burn'] = (0, None)
mfit.limits['x_phi']     = (0, None)

print('running migrad...')
mfit.migrad()
print('running hesse...')
mfit.hesse()
print('running minos...')
mfit.minos()
print(mfit)

# pull table
LABELS = {
    'eps':     'efficiency ratio',
    'Pgm':     'Pg(141.4 keV, Xe-125m)',
    'Pgg':     'Pg(188.4 keV, Xe-125g)',
    'Thm':     'T1/2(Xe-125m) [s]',
    'Thg':     'T1/2(Xe-125g) [s]',
    'sg_burn': 'sigma_burn(Xe-125g) [cm2]',
    'sm_burn': 'sigma_burn(Xe-125m) [cm2]',
    'sig124':  'sigma_total(Xe-124) [cm2]',
    'phi':     'flux [n/cm2/s]',
    'N124_0':  'initial Xe-124 atoms (fixed)',
}

print('\n' + '='*72)
print(f'{"parameter":<34} {"nominal":>12} {"best fit":>12} {"pull (sigma)":>10}')
print('='*72)
for k in KEYS:
    xv   = mfit.values[f'x_{k}']
    if k in FIXED:
        pull_str = 'fixed'
    else:
        pull_str = f'{(xv - 1.0) / REL[k]:>10.3f}'
    print(f'{LABELS[k]:<34} {NOM[k]:>12.4g} {xv*NOM[k]:>12.4g} {pull_str:>10}')
print('='*72)
print(f'{"IYR":<34} {"—":>12} {mfit.values["IYR"]:>12.6f} {"—":>10}')
print('='*72)

ndof    = len(t_m) - 1
IYR_fit = mfit.values['IYR']
err_lo  = abs(mfit.merrors['IYR'].lower)
err_hi  = mfit.merrors['IYR'].upper
print(f'\nIYR = {IYR_fit:.6f}  +{err_hi:.6f} / -{err_lo:.6f}')
print(f'chi2/ndof = {mfit.fval:.2f} / {ndof} = {mfit.fval/ndof:.3f}')

# save best-fit results for 2d contour script
bestfit = {
    'IYR':        float(IYR_fit),
    'IYR_err_lo': float(err_lo),
    'IYR_err_hi': float(err_hi),
    'fval':       float(mfit.fval),
}
for k in KEYS:
    bestfit[f'x_{k}']     = float(mfit.values[f'x_{k}'])
    bestfit[f'x_{k}_err'] = float(mfit.errors[f'x_{k}'])
with open(os.path.join(_dir, f'{RUN_NAME}_bestfit.json'), 'w') as f:
    json.dump(bestfit, f, indent=2)
print(f'saved {RUN_NAME}_bestfit.json')            # this is the output file that can be plugged into the 2D contour plot script, make sure RUN_NAME matches in both scripts

# plot settings
plt.rcParams.update({
    'font.family':         'serif',
    'mathtext.fontset':    'dejavuserif',
    'font.size':           9,
    'axes.linewidth':      0.8,
    'axes.labelsize':      9.5,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.frameon':      False,
    'savefig.bbox':        'tight',
})

C_FIT  = '#0072B2'
C_BAND = '#0072B2'

BF = {k: mfit.values[f'x_{k}'] * NOM[k] for k in KEYS}

def pred(t_abs, IYR):
    return model(t_abs, IYR,
                 BF['eps'], BF['Pgm'], BF['Pgg'], BF['Thm'], BF['Thg'],
                 BF['sg_burn'], BF['sm_burn'], BF['sig124'], BF['phi'], BF['N124_0'])

IYR_lo = IYR_fit - err_lo
IYR_hi = IYR_fit + err_hi
t_fine = np.linspace(t_m.min(), t_m.max() * 1.02, 300)

# figure 1: data and model with pull subpanel
fig, (ax, axp) = plt.subplots(
    2, 1, figsize=(3.4, 3.6), sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06}
)

ax.errorbar(t_m, r_m, yerr=s_m, fmt='o', color='black',
            ms=2.2, elinewidth=0.6, mew=0.6, zorder=3, label='data')
ax.fill_between(t_fine, pred(t_fine, IYR_lo), pred(t_fine, IYR_hi),
                color=C_BAND, alpha=0.25, lw=0, zorder=2, label=r'$\pm1\sigma$')
ax.plot(t_fine, pred(t_fine, IYR_fit), color=C_FIT, lw=1.3, zorder=4,
        label='best fit, IYR = %.5f $\\pm$ %.5f' % (IYR_fit, 0.5*(err_lo+err_hi)))
ax.set_yscale('log')
ax.set_ylabel(r'$A_{111}/A_{188}$')
ax.legend(fontsize=7.5, loc='upper right', handlelength=1.8)
ax.tick_params(labelbottom=False)

pulls = (r_m - pred(t_m, IYR_fit)) / s_m
axp.axhspan(-1, 1, color='0.90', lw=0)
axp.axhline(0, color='0.4', lw=0.6)
axp.plot(t_m, pulls, 'o', color='black', ms=2.0)
axp.set_ylim(-3.4, 3.4)
axp.set_yticks([-2, 0, 2])
axp.set_xlabel('time since start of experiment (s)')
axp.set_ylabel(r'pull ($\sigma$)')

plt.savefig(os.path.join(_dir, f'{RUN_NAME}_nuisance_fit.pdf'))
plt.savefig(os.path.join(_dir, f'{RUN_NAME}_nuisance_fit.png'), dpi=600)
print(f'saved {RUN_NAME}_nuisance_fit.pdf / .png')

# figure 2: delta chi2 profile
print('running profile scan...')
n_scan   = 31
half     = 2.5 * max(err_lo, err_hi)
IYR_scan = np.linspace(max(1e-4, IYR_fit - half), IYR_fit + half, n_scan)
dchi2    = np.zeros(n_scan)

for i, iv in enumerate(IYR_scan):
    ms = Minuit(chi2, IYR=iv, **{f'x_{k}': mfit.values[f'x_{k}'] for k in KEYS})
    ms.errordef = Minuit.LEAST_SQUARES
    for k in KEYS:
        ms.errors[f'x_{k}'] = REL[k]
        if k in FIXED:
            ms.fixed[f'x_{k}'] = True
    ms.limits['x_sg_burn'] = (0, None)
    ms.limits['x_sm_burn'] = (0, None)
    ms.fixed['IYR']        = True
    ms.migrad()
    dchi2[i] = ms.fval - mfit.fval

fig2, ax2 = plt.subplots(figsize=(3.4, 2.8))
ax2.plot(IYR_scan, dchi2, 'o', color=C_FIT, ms=3, zorder=4, label='profile scan')
co = np.polyfit(IYR_scan, dchi2, 2)
xf = np.linspace(IYR_scan[0], IYR_scan[-1], 400)
ax2.plot(xf, np.polyval(co, xf), color=C_FIT, lw=1.2, zorder=3, label='parabolic fit')
for lev, lab in ((1.0, r'$68\%$ CL'), (3.84, r'$95\%$ CL')):
    ax2.axhline(lev, color='0.45', lw=0.7, ls=':')
    ax2.text(IYR_scan[-1], lev, ' '+lab, va='bottom', ha='right', fontsize=7.5, color='0.35')
ax2.axvline(IYR_fit, color='0.3', lw=0.7, ls='--')
for xv in (IYR_lo, IYR_hi):
    ax2.axvline(xv, color=C_FIT, lw=0.6, ls='--', alpha=0.6)
ax2.set_xlabel('IYR')
ax2.set_ylabel(r'$\Delta\chi^{2}$')
ax2.set_ylim(0, max(dchi2) * 1.05)
ax2.legend(fontsize=7.5, loc='upper center')

plt.savefig(os.path.join(_dir, f'{RUN_NAME}_iyr_profile.pdf'))
plt.savefig(os.path.join(_dir, f'{RUN_NAME}_iyr_profile.png'), dpi=600)
print(f'saved {RUN_NAME}_iyr_profile.pdf / .png')

plt.show()

