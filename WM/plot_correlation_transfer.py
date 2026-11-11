"""All pool transfers, with uncertainty and an explicitly idealised scale control."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'correlation_transfer.csv').open()))
groups=['baseline','long_period','weak_waves']; labels=['Baseline','Long period','Weak waves']
methods=['record_scaled','independent_target_scale']
selected=[r for r in rows if r['headings']=='12' and r['gain_reference']=='baseline' and r['coupling']=='joint']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for metric,title,filename in [('actual_ratio','Variance transfer improves; nominal coverage remains inadequate','correlation_transfer_variance.png'),
                              ('coverage95','Coverage of nominal 95% intervals on unchanged wave estimates','correlation_transfer_coverage.png')]:
    fig,axes=plt.subplots(1,2,figsize=(11.4,6.4))
    fig.subplots_adjust(left=.13,right=.88,bottom=.25,top=.81,wspace=.35)
    for ax,method,heading in zip(axes,methods,['Current-record variances (candidate)','Independent target variances (diagnostic)']):
        rs=[[next(r for r in selected if r['group']==g and r['correlation_pool']==p and r['method']==method)
             for g in groups] for p in groups]
        values=np.array([[float(r[metric]) for r in rr] for rr in rs])
        if metric=='actual_ratio':
            im=ax.imshow(values,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=.5,vcenter=1,vmax=1.5))
        else:im=ax.imshow(100*values,cmap='Blues',vmin=0,vmax=100)
        for i in range(3):
            for j in range(3):
                r=rs[i][j]
                if metric=='actual_ratio':
                    text=f'{values[i,j]:.2f}\n[{float(r["actual_ratio_ci_low"]):.2f}, {float(r["actual_ratio_ci_high"]):.2f}]'
                else:text=f'{100*values[i,j]:.1f}%\n[{100*float(r["coverage95_ci_low"]):.1f}, {100*float(r["coverage95_ci_high"]):.1f}]'
                ax.text(j,i,text,ha='center',va='center',fontsize=10,color='black',
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=3))
        ax.set_xticks(range(3),labels);ax.set_yticks(range(3),labels)
        ax.set(title=heading,xlabel='Test condition',ylabel='Independent correlation pool')
    fig.colorbar(im,cax=fig.add_axes([.91,.30,.02,.43]),label='Predicted / actual variance' if metric=='actual_ratio' else 'Coverage (%)')
    fig.suptitle(title,fontsize=14)
    fig.text(.025,.045,'Source: correlation_transfer.csv; 200 previously examined recordings per condition; 12 headings; fixed baseline gain, M = 8.785 and tau = 5.\n'
             'Brackets: pointwise conditional 95% intervals (whole-record bootstrap for variance; exact binomial for coverage). No pool selection or bias correction.\n'
             'The right-hand control assumes representative independent target calibration. It is not the deployable candidate or a new held-out test.',fontsize=9)
    fig.savefig(root/filename,dpi=180)
    plt.close(fig)
