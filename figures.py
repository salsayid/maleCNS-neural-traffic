"""Scientific figures rendered directly from MaleCNS SWC skeletons and weights."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

BG='#f5f3ec'
COLORS=['#555555','#8d6d49','#385a6c','#7b6956','#61765b','#6f5b78','#8c4e48','#416c65','#7f733b','#667785']
TYPES=['L1','L3','Mi1','Mi4','Mi9','Tm3','T4a','T4b','T4c','T4d']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':BG,'axes.facecolor':BG,
 'savefig.facecolor':BG,'text.color':'#222222','axes.labelcolor':'#222222'})

def skeletons():
    data={}
    for typ in TYPES:
        p=next((ROOT/'data').glob(typ+'-*.swc'));a=np.loadtxt(p);index={int(r[0]):i for i,r in enumerate(a)}
        xyz=a[:,2:5]*.008 # SWC coordinates in original 8nm voxel units -> micrometers
        lines=np.array([[xyz[index[int(row[6])]],xyz[i]] for i,row in enumerate(a) if int(row[6]) in index])
        data[typ]=dict(body=int(p.stem.split('-')[1]),xyz=xyz,lines=lines,path=p)
    return data

def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True);data=skeletons()
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,dims,title in zip(axes,[(0,2),(1,2)],['EM x–z projection','EM y–z projection']):
        for typ,color in zip(TYPES,COLORS):
            lines=data[typ]['lines'][:,:,dims]
            ax.add_collection(LineCollection(lines,colors=color,linewidths=.55,alpha=.8,label=typ))
        ax.autoscale();ax.set_aspect('equal');ax.set(title=title,xlabel=f'EM {"xyz"[dims[0]]} (µm)',ylabel='EM z (µm)')
    axes[1].legend(ncol=2,fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
    fig.savefig(out/'anatomy.png',dpi=190);fig.savefig(out/'anatomy.svg');plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(11,4.5),layout='constrained')
    for ax,typ,color in zip(axes.flat,TYPES,COLORS):
        lines=data[typ]['lines'][:,:,(0,2)].copy();center=(lines.reshape(-1,2).min(0)+lines.reshape(-1,2).max(0))/2;lines-=center
        ax.add_collection(LineCollection(lines,colors=color,linewidths=.7));ax.autoscale();ax.set_aspect('equal')
        ax.set_title(f'{typ}\nbody {data[typ]["body"]}',fontsize=11);ax.axis('off')
        ax.plot([-70,-50],[-55,-55],color='#333',lw=1.5);ax.text(-60,-59,'20 µm',ha='center',va='top',fontsize=8)
        ax.set_xlim(-85,85);ax.set_ylim(-70,60)
    fig.savefig(out/'samples.png',dpi=190);fig.savefig(out/'samples.svg');plt.close(fig)
    n=pd.read_csv(ROOT/'data/neurons.csv');e=pd.read_csv(ROOT/'data/edges.csv')
    matrix=np.zeros((len(n),len(n)))
    matrix[e.source,e.target]=e.weight
    order=np.argsort(n.type_index.to_numpy(),kind='stable');matrix=matrix[order][:,order]
    fig,ax=plt.subplots(figsize=(6,5.3),layout='constrained')
    im=ax.imshow(np.log10(matrix+1),cmap='Greys',interpolation='nearest')
    boundaries=[0];centers=[]
    for typ in TYPES:
        count=int((n.type==typ).sum());centers.append(boundaries[-1]+(count-1)/2);boundaries.append(boundaries[-1]+count)
    ax.set(xticks=centers,xticklabels=TYPES,yticks=centers,yticklabels=TYPES,xlabel='Postsynaptic neuron',ylabel='Presynaptic neuron')
    ax.tick_params(axis='x',rotation=45)
    for b in boundaries[1:-1]:ax.axhline(b-.5,color='#aaa',lw=.3);ax.axvline(b-.5,color='#aaa',lw=.3)
    fig.colorbar(im,ax=ax,label='log10(synaptic contacts + 1)',shrink=.7)
    fig.savefig(out/'connectivity.png',dpi=180);plt.close(fig)
    manifest=[]
    for typ,item in data.items():
        p=item['path'];manifest.append(dict(type=typ,body_id=item['body'],file=p.name,
            source=f'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{item["body"]}.swc',
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),coordinate_conversion='8 nm per coordinate unit; plotted in µm'))
    geo=ROOT/'data/downtown-signals.geojson'
    provenance=dict(neurons=manifest,city_data=dict(source='https://gis.lincoln.ne.gov/public/rest/services/LTUTraffic/TrafficSignals/MapServer/3',
        file=geo.name,sha256=hashlib.sha256(geo.read_bytes()).hexdigest(),retrieved='2026-09-26',used_fields=['IntersectionName','IntersectionNumber','geometry']),
        attribution='MaleCNS: FlyEM/HHMI Janelia, Cambridge, MRC LMB, Google Research (CC-BY). Signal locations: City of Lincoln/Lancaster County GIS.',
        note='Ten representative actual neurons from the 196-neuron model; morphology is shown for reference and is not used in the rate equations.')
    (ROOT/'data/provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Saved anatomy, neuron samples, connectivity, and provenance')

if __name__=='__main__':main()
