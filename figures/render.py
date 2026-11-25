"""Render verified vTSFM measurements as vector PDFs and slide-resolution PNGs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'measurements.json').read_text())
INK, MUTED, GRID = '#152F3D', '#455B66', '#DAE2E5'
BASE, OPT, SECOND = '#70838C', '#006F70', '#31689B'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12.5, 'text.color': INK,
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK,
                     'pdf.fonttype': 42, 'axes.unicode_minus': False})


def canvas(title, subtitle, footer, *, wide=False):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6))
    fig.subplots_adjust(left=.17 if wide else .145, right=.962, top=.74, bottom=.255,
                        wspace=.76 if wide else .59)
    fig.text(.05,.921,title,fontsize=24,weight='bold',va='top')
    fig.text(.05,.835,subtitle,fontsize=13.5,color=MUTED,va='top')
    fig.text(.05,.044,footer,fontsize=10.4,color=MUTED,va='bottom')
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='x',color=GRID,linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=6)
        ax.xaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
    return fig, axes


def horizontal(ax, labels, groups, colors, xlabel, title, *, xmax=None, fmt='g', missing=None):
    """All bars start at zero. A missing qualifying rate gets a textual label."""
    maxval=max(value for group in groups for value in group if value is not None)
    limit=xmax or maxval*1.28
    height=.55 if len(groups)==1 else .28
    for j,(group,color) in enumerate(zip(groups,colors,strict=True)):
        for i,value in enumerate(group):
            y=i+(j-(len(groups)-1)/2)*(height+.06)
            if value is None:
                ax.text(.015*limit,y,missing or 'Unavailable',va='center',fontsize=10,color=MUTED)
                continue
            c=color[i] if isinstance(color,list) else color
            ax.barh(y,value,height=height,color=c)
            ax.text(value+.02*limit,y,format(value,fmt),va='center',fontsize=11.5,color=INK)
    ax.set_xlim(0,limit)
    ax.set_ylim(len(labels)-.50,-.50)
    ax.set_yticks(range(len(labels)),labels,fontsize=12)
    ax.set_xlabel(xlabel,fontsize=12,labelpad=9)
    ax.set_title(title,loc='left',fontsize=14,weight='bold',pad=12)


def legend(fig, labels, colors):
    fig.legend([Patch(facecolor=c) for c in colors],labels,loc='center',
               bbox_to_anchor=(.52,.127),ncol=len(labels),frameon=False,fontsize=11.5)


def export(fig, stem):
    for suffix in ['pdf','png']:
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {}
        fig.savefig(ROOT / f'{stem}.{suffix}',dpi=165,facecolor='white',metadata=metadata)
    plt.close(fig)


def serving():
    d=DATA['serving']; fig,(a,b)=canvas('Forecast serving across three models',
        'A100   ·   Chiller 6   ·   One series, 2,048 observations, 96 forecast steps   ·   Eager FP32',
        'Historical baseline 9cde2f1 · #51 · A rate qualified only if all requests completed and p99 ≤ 500 ms in all three repeats.')
    horizontal(a,d['models'],[d['fastapi_p95_ms'],d['vtsfm_p95_ms']],[BASE,OPT],
               'End-to-end p95 latency (ms)','One outstanding request',fmt='.1f')
    horizontal(b,d['models'],[d['fastapi_slo_offer'],d['vtsfm_slo_offer']],[BASE,OPT],
               'Highest qualifying offered rate (req/s)','Open-loop arrivals',missing='No tested rate qualified')
    legend(fig,['Native FastAPI HTTP','vTSFM whole-forecast fusion'],[BASE,OPT]); export(fig,'01-serving')


def graph():
    d=DATA['graph']; fig,(a,b)=canvas('TimesFM full-forecast CUDA graph replay',
        'A100   ·   Two series, context 512, horizon 1,024   ·   One outstanding HTTP request',
        'Fixed-shape serving experiment #70 · 300 timed requests per arm · Memory is sampled device residency, including graph pools.')
    for ax,key,label,title,fmt in [(a,'http_mean_ms','Mean latency (ms)','HTTP latency','.1f'),(b,'peak_nvml_mib','Peak sampled memory (MiB)','GPU memory','d')]:
        values=d[key]; horizontal(ax,d['layouts'],[values['Eager'],values['CUDA graph']],[BASE,OPT],label,title,fmt=fmt)
    legend(fig,['Eager','CUDA graph replay'],[BASE,OPT]); export(fig,'02-timesfm-cuda-graphs')


def attention():
    d=DATA['attention']; fig,(a,b)=canvas('Moirai univariate attention at long context',
        'RTX A6000   ·   Eight series, context 7,168, horizon 1,024   ·   Eager FP32',
        'Merged implementation bd1eca2 · #24 · Mean of two runs per mode in reversed order · Allocator peak is measured in the worker.')
    colors=[[BASE,OPT]]
    horizontal(a,d['modes'],[[v/1000 for v in d['http_latency_ms']]],colors,'HTTP latency (s)','Latency',fmt='.2f')
    horizontal(b,d['modes'],[d['peak_allocated_gib']],colors,'Peak GPU allocation (GiB)','Worker memory',fmt='.2f')
    export(fig,'03-moirai-attention')


def compile_scope():
    d=DATA['compile']; fig,(a,b)=canvas('Toto compilation across two experiments',
        'One series, context 2,048, horizon 96   ·   HTTP p50 latency   ·   Compare within each panel',
        'Issue #17 · Different hardware, revisions and compile scopes · The A6000 pilot predates the eager unit-scaling fast path.')
    horizontal(a,['Order 1','Order 2'],[d['a100_eager_p50'],d['a100_guarded_p50']],[BASE,SECOND],
               'HTTP p50 (ms)','A100  ·  shipped guarded compile',xmax=118,fmt='.1f')
    horizontal(b,['Eager','Whole-forecast\ncompile'],[[d['a6000_eager_p50'],d['a6000_whole_p50']]],[[BASE,OPT]],
               'HTTP p50 (ms)','RTX A6000  ·  earlier pilot',xmax=118,fmt='.1f')
    legend(fig,['Eager','Guarded compile','Whole-forecast compile'],[BASE,SECOND,OPT]); export(fig,'04-toto-compilation')


def blocks():
    d=DATA['blocks']; fig,(a,b)=canvas('TimesFM block scheduling under a long-horizon burst',
        'A100   ·   100 two-series MetroPT requests, 1 ms apart   ·   Context 512, horizon 1,024',
        'Issue #51 and PR #83 · Two separate campaigns · Final comparison used six fresh-server orders and passed all nine native quantiles.',wide=True)
    for ax,key,title,colors in [(a,'earlier','Decode and prefill ablation',[BASE,SECOND,OPT,BASE]),(b,'later','Final six-order comparison',[BASE,SECOND,OPT])]:
        values=d[key]; horizontal(ax,list(values),[list(values.values())],[colors], 'Completed requests/s',title,xmax=14.4,fmt='.3f')
    export(fig,'05-timesfm-blocks')


def toto_runtime():
    d=DATA['toto_runtime']; fig,(a,b)=canvas('Toto inference-only unit scaling',
        'A100   ·   Four series, context 128, horizon 96   ·   One server slot, repeated HTTP requests',
        'PRs #88 and #94 · Combined change versus 8110e34 · Four concurrent clients queue at the one-slot server.')
    labels=[f'{n} client' if n==1 else f'{n} clients' for n in d['clients']]
    horizontal(a,labels,[d['control']['rps'],d['candidate']['rps']],[BASE,OPT],
               'Completed requests/s','Throughput',fmt='.2f')
    horizontal(b,labels,[d['control']['p95_ms'],d['candidate']['p95_ms']],[BASE,OPT],
               'HTTP p95 latency (ms)','Tail latency',fmt='.1f')
    legend(fig,['Control','Inference fast path'],[BASE,OPT]); export(fig,'06-toto-runtime')


if __name__=='__main__':
    serving(); graph(); attention(); compile_scope(); blocks(); toto_runtime()
