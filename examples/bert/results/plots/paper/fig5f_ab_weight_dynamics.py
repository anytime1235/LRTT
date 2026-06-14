"""Figure 5(f): Auxiliary tile B weight dynamics over training steps.

|w|_max  = max(|w_min|, |w_max|)  — exact from diagnostic
|w|_mean = mean_abs_A  — exact from diagnostic (multi_tiles)

Averaged across layers L0, L6, L11 for each sublayer Q/K/V/O.
NOTE: code A = paper B. Only paper B (code A) is shown.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_PATH = '/root/LRTT/examples/bert/results/plots/fig5f_data.json'
with open(DATA_PATH) as f:
    d = json.load(f)

sublayers = ['query', 'key', 'value', 'output']
sublabel  = {'query': 'Q', 'key': 'K', 'value': 'V', 'output': 'O'}
layers    = ['L0', 'L6', 'L11']

ref_key = f'{layers[0]}_{sublayers[0]}'
all_steps = np.array(d['multi_tile_data'][ref_key]['steps'])

# Collect data for code A (= paper B) only
data = {}
for sl in sublayers:
    abs_max_layers = []
    mean_abs_layers = []
    for layer in layers:
        key = f'{layer}_{sl}'
        td = d['multi_tile_data'][key]
        mn = np.array(td['A_eff_min'])
        mx = np.array(td['A_eff_max'])
        ma = np.array(td['mean_abs_A'])
        abs_max_layers.append(np.maximum(np.abs(mn), np.abs(mx)))
        mean_abs_layers.append(ma)
    data[sl] = {
        'abs_max':  np.mean(abs_max_layers, axis=0),
        'mean_abs': np.mean(mean_abs_layers, axis=0),
    }

# --- Plot ---
plt.rcParams.update({
    'font.size':        9,
    'axes.labelsize':   10,
    'legend.fontsize':  7.5,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'axes.linewidth':   0.8,
})

colors = {
    'query':  '#1f77b4',
    'key':    '#ff7f0e',
    'value':  '#2ca02c',
    'output': '#d62728',
}

fig, ax = plt.subplots(figsize=(5.5, 3.2))

for sl in sublayers:
    lbl = sublabel[sl]
    c = colors[sl]
    ax.plot(all_steps, data[sl]['mean_abs'], color=c, linestyle='-', linewidth=1.4,
            label=rf'$\overline{{|w|}}$ ({lbl})')
    ax.plot(all_steps, data[sl]['abs_max'], color=c, linestyle='--', linewidth=1.8,
            label=rf'$|w|_{{\mathrm{{max}}}}$ ({lbl})')

ax.set_ylabel('Tile B weight magnitude')
ax.set_xlabel('Training step')
ax.set_xlim(all_steps[0], all_steps[-1])
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

handles, labels = ax.get_legend_handles_labels()
mean_h = handles[0::2]
mean_l = labels[0::2]
max_h  = handles[1::2]
max_l  = labels[1::2]
ax.legend(mean_h + max_h, mean_l + max_l,
          ncol=4, loc='lower right',
          handlelength=1.8, columnspacing=0.6,
          fontsize=7.5, framealpha=0.95, edgecolor='0.7')

fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/fig5f_ab_weight_dynamics.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')
