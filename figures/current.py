"""Render the November A100 studies from their checked measurement summaries."""
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from render import ROOT, BASE, GRID, INK, MUTED, OPT, SECOND, export


def figure(title, subtitle, note, width=12.8):
    fig = plt.figure(figsize=(width, 6.6))
    fig.text(.05, .94, title, fontsize=24, weight='bold', va='top')
    fig.text(.05, .848, subtitle, fontsize=13, color=MUTED, va='top')
    fig.text(.05, .038, note, fontsize=10.2, color=MUTED, va='bottom')
    return fig


def tidy(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.tick_params(length=0, pad=6)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def toto():
    results = json.loads((ROOT / 'data/toto-compile-a100.json').read_text())['points']
    cases = [
        ('assetops-c2048-h96-b1-window-0-k1', '1 × 2048\n1 client'),
        ('assetops-c2048-h96-b1-window-0-k4', '1 × 2048\n4 clients'),
        ('assetops-c512-h96-b4-window-0-k1', '4 × 512\n1 client'),
        ('assetops-c512-h96-b4-window-0-k4', '4 × 512\n4 clients'),
    ]
    modes = [('eager', 'Eager', BASE), ('core', 'Guarded compile', SECOND),
             ('whole', 'Whole-forecast compile', OPT)]
    fig = figure('Toto whole-forecast compilation lifts warm throughput',
                 'A100   ·   MetroPT   ·   Horizon 96   ·   One server slot   ·   Eager FP32 and two compile scopes',
                 '2,400 timed requests. Whole compile first use: 21–32 s per shape. Chiller stress test: 69 eager and 110 whole-compile parity misses out of 4,320.')
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.085, right=.975, top=.73, bottom=.28, wspace=.32)
    for ax, key, unit in zip(axes, ('req_per_s', 'p95_ms'), ('Throughput (requests/s)', 'End-to-end p95 latency (ms)'), strict=True):
        tidy(ax)
        ax.set_title(unit, loc='left', fontsize=14, weight='bold', pad=12)
        for offset, (mode, _, color) in enumerate(modes):
            values = [results[k][mode][key] for k, _ in cases]
            assert all(results[k][mode]['requests'] == 200 for k, _ in cases)
            positions = [i + (offset-1)*.24 for i in range(len(cases))]
            ax.bar(positions, values, width=.22, color=color)
        ax.set_xticks(range(len(cases)), [label for _, label in cases], fontsize=11)
        ax.set_ylim(bottom=0)
    fig.legend([Patch(facecolor=c) for _, _, c in modes], [l for _, l, _ in modes], ncol=3,
               frameon=False, loc='center', bbox_to_anchor=(.5,.135), fontsize=12)
    export(fig, '07-toto-compile-a100')



def attention():
    data = json.loads((ROOT / 'data/moirai-attention-a100.json').read_text())
    modes = data['horizon_96']['modes']
    failures = data['horizon_1024']['fused_parity']['failures_by_64_step_block']
    assert sum(failures) == data['horizon_1024']['fused_parity']['outside']
    fig = figure('Moirai attention depends on forecast horizon',
                 'A100   ·   Eight tiled MetroPT histories, context 7,168   ·   One outstanding HTTP request',
                 'H96: 200 requests per mode, two orders. H1024: 450 of 73,728 values exceeded native tolerance; all belonged to one series.')
    left, right = fig.subplots(1, 2)
    fig.subplots_adjust(left=.09, right=.965, top=.715, bottom=.20, wspace=.34)
    tidy(left)
    labels = ['Upstream', 'Fused univariate']
    values = [modes[k]['p95_ms'] for k in ['upstream', 'univariate']]
    left.bar(labels, values, color=[BASE, OPT], width=.52)
    for i,val in enumerate(values):
        left.text(i, val + max(values)*.025, f'{val:.1f} ms', ha='center', fontsize=13)
    left.set_ylim(0, max(values)*1.18)
    left.set_ylabel('End-to-end p95 latency (ms)')
    left.set_title('96 steps  ·  Both modes pass parity', loc='left', fontsize=14,weight='bold',pad=12)
    tidy(right)
    right.bar(range(1,17), failures, color=SECOND)
    right.set_xticks([1,4,8,12,16])
    right.set_xlabel('Forecast block (64 steps each)')
    right.set_ylabel('Quantile values outside tolerance')
    right.set_title('1,024 steps  ·  Fused attention errors',loc='left',fontsize=14,weight='bold',pad=12)
    export(fig, '08-moirai-attention-a100')



def blocks():
    data = json.loads((ROOT / 'data/moirai-blocks-a100.json').read_text())
    profiles = {(p['mode'],p['scenario']):p for p in data['profiles']}
    assert all(p['paired_comparison_qualified'] and p['completed']==200 for p in profiles.values())
    fig = figure('Moirai block scheduling trades throughput for memory',
                 'A100   ·   MetroPT open-loop arrivals   ·   Four slots   ·   Forecasts interleave at 64-step boundaries',
                 '800 timed requests across two server orders. All 160 separate concurrent warmups passed native parity. Memory is peak Torch allocation.')
    axes = fig.subplots(1,3)
    fig.subplots_adjust(left=.076,right=.985,top=.705,bottom=.29,wspace=.30)
    specs=[('pooled_rps',1,'Completed requests/s','.1f'),('p95_offered_ms',1000,'Arrival-to-completion p95 (s)','.1f'),
           ('peak_torch_allocated_bytes',2**20,'Peak GPU allocation (MiB)','.0f')]
    for ax,(metric,divisor,title,fmt) in zip(axes,specs,strict=True):
        tidy(ax)
        ax.set_title(title,weight='bold',fontsize=12.5,loc='left',pad=12)
        peak=max(p[metric]/divisor for p in profiles.values())
        for offset,(mode, color) in enumerate([('whole',BASE),('continuous',OPT)]):
            positions=[i+(offset-.5)*.34 for i in range(2)]
            vals=[profiles[mode,scenario][metric]/divisor for scenario in ['saturated','bursty']]
            ax.bar(positions,vals,width=.31,color=color)
            for x,val in zip(positions,vals,strict=True):
                ax.text(x,val+peak*.02,format(val,fmt),ha='center',fontsize=10.5)
        ax.set_ylim(0,peak*1.17)
        ax.set_xticks([0,1],['H1024 burst','Mixed burst'],fontsize=10.7)
    fig.legend([Patch(facecolor=c) for c in [BASE,OPT]],['Whole forecast','64-step blocks, requests isolated'],
               frameon=False,loc='center',bbox_to_anchor=(.5,.14),fontsize=12,ncol=2)
    export(fig,'09-moirai-block-scheduling')



def replay():
    from render import canvas, horizontal, legend
    control=json.loads((ROOT/'data/toto-qualification-a100.json').read_text())['direct_replay']
    modes=['eager','whole']
    labels=['Eager','Whole-forecast\ncompile']
    fig,(a,b)=canvas('Toto CUDA graph replay at a fixed shape',
        'A100   ·   Direct tensor calls   ·   B1/C2048/H96   ·   Experimental device-side unit scaling',
        '100 calls per cell in four blocks. Both paths copy inputs, clone outputs and synchronize. Each graph held an additional 8.13 MiB on the GPU.',wide=True)
    for key in modes:
        cell=control[key]
        assert cell['status']=='passed' and cell['unmodified_capture_status']=='failed'
        assert all(cell[arm]['count']==100 for arm in ['ordinary','graph'])
    horizontal(a,labels,[[control[m][arm]['p95_ms'] for m in modes] for arm in ['ordinary','graph']],
               [BASE,OPT],'p95 latency (ms)','Steady-state tensor call',fmt='.1f')
    horizontal(b,labels,[[1000*control[m]['capture_s'] for m in modes]],[OPT],
               'One-time capture (ms)','CUDA graph setup',fmt='.1f')
    legend(fig,['Ordinary invocation','CUDA graph'],[BASE,OPT])
    export(fig,'13-toto-direct-graph')


if __name__ == '__main__':
    toto()
    attention()
    blocks()
    replay()
