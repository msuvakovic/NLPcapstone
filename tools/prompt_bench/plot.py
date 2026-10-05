"""Static, exportable figures from completed benchmark measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='output/prompt_benchmark_2026-09-29')
    root=Path(p.parse_args().output)
    rows=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    names=['Base instruction','Human-written','Existing OPRO (corrected evaluation)',
           'Official GEPA','Official GEPA + stable selection','ESPO-inspired + stable selection']
    selected=[next(r for r in rows if r['regime']=='single' and r['method']==name) for name in names]
    labels=['Base','Human','OPRO','GEPA','GEPA + stable','ESPO-inspired']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1.3,1]})
    colors={'sst2':'#546e8f','amazon':'#198c83','tweets':'#d39031'}
    x=np.arange(len(selected)); width=.25
    for offset,(domain,color) in enumerate(colors.items()):
        axes[0].bar(x+(offset-1)*width,[100*r['domain_accuracy'][domain] for r in selected],width,
                    color=color,label={'sst2':'Source: SST-2','amazon':'OOD: Amazon','tweets':'OOD: tweets'}[domain])
    axes[0].set_xticks(x,labels,rotation=30,ha='right')
    axes[0].set_ylim(0,100); axes[0].set_ylabel('Final accuracy (%)'); axes[0].legend(loc='lower left',fontsize=9)
    axes[0].set_title('Frozen prompts on 300 examples per domain',loc='left',fontweight='bold')
    for i,r in enumerate(selected[1:]):
        delta=100*r['ood_delta_vs_base']; lo,hi=np.array(r['ood_delta_ci95'])*100
        axes[1].plot([lo,hi],[i,i],color='#355b82',linewidth=2)
        axes[1].scatter([delta],[i],color='#198c83',s=45,zorder=3)
    axes[1].axvline(0,color='#9aa3ad',linestyle='--',linewidth=1)
    axes[1].set_yticks(range(len(selected)-1),labels[1:]); axes[1].invert_yaxis()
    axes[1].set_xlabel('OOD change versus base (percentage points)')
    axes[1].set_title('Paired 95% intervals',loc='left',fontweight='bold')
    fig.suptitle('Prompt optimization: controlled sentiment pilot',x=.07,ha='left',fontsize=17,fontweight='bold')
    fig.text(.07,.015,'Gemma 4 31B | Three search seeds, one data split | Intervals condition on observed seeds; exploratory comparisons',fontsize=9,color='#56606a')
    fig.tight_layout(rect=[0,.06,1,.93])
    fig.savefig(root/'performance.png',dpi=180)
    fig.savefig(root/'performance.svg')
    print('Wrote',root/'performance.png')
    plt.close(fig)
    diagnostic=[r for r in selected if r.get('overfitting')]
    fig,ax=plt.subplots(figsize=(10,5.2))
    x=np.arange(len(diagnostic))
    for offset,(metric,label,color) in enumerate([
        ('validation_gain','Validation gain','#546e8f'),
        ('source_gain','Untouched source gain','#198c83'),
        ('target_gain','Untouched OOD gain','#d39031'),
    ]):
        ax.bar(x+(offset-1)*width,[100*r['overfitting']['mean'][metric] for r in diagnostic],
               width,label=label,color=color)
    ax.axhline(0,color='#7c8791',linewidth=1)
    ax.set_xticks(x,['OPRO','GEPA','GEPA + stable','ESPO-inspired'])
    ax.set_ylabel('Change versus base instruction (percentage points)')
    ax.legend(loc='best',frameon=False)
    ax.set_title('Does the validation gain survive on untouched examples?',loc='left',fontsize=15,fontweight='bold',pad=18)
    fig.text(.08,.025,'Single-source optimization | Means across three search seeds on one split\nDescriptive gaps; 64 validation examples give only 1.56 pp resolution per seed.',fontsize=9,color='#56606a')
    fig.tight_layout(rect=[0,.09,1,1])
    fig.savefig(root/'overfitting.png',dpi=180)
    fig.savefig(root/'overfitting.svg')
    print('Wrote',root/'overfitting.png')


if __name__=='__main__': main()
