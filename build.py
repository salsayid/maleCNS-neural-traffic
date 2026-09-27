"""Build a plain scientific report from measured experiment outputs."""
from pathlib import Path
import base64,json,csv
import numpy as np
from figures import plt,BG
from traffic import SCENARIOS
ROOT=Path(__file__).resolve().parent
LABELS={'fixed':'Fixed schedule','heuristic':'Queue heuristic (teacher)','connectome':'MaleCNS circuit','rewired':'Rewired circuit','mlp':'Conventional network'}

def rows_for(m,kind,scenario):
    if kind in m['baselines']:return m['baselines'][kind][scenario]
    return [x for r in m['runs'] if r['kind']==kind for x in r['scenarios'][scenario]]

def main():
    m=json.loads((ROOT/'results/metrics.json').read_text());traces=json.loads((ROOT/'results/traces.json').read_text())
    summary={}
    for kind in LABELS:
        summary[kind]={}
        for scenario in SCENARIOS:
            rows=rows_for(m,kind,scenario)
            values=np.array([r['queue_seconds_per_arrival'] for r in rows])
            summary[kind][scenario]=dict(mean=float(values.mean()),max_residual=max(r['residual_queue'] for r in rows))
    # Paired bootstrap over traffic seeds, after averaging the three model seeds.
    fixed=np.array([r['queue_seconds_per_arrival'] for r in m['baselines']['fixed']['standard']])
    circuit=np.array([[x['queue_seconds_per_arrival'] for x in r['scenarios']['standard']] for r in m['runs'] if r['kind']=='connectome']).mean(0)
    rng=np.random.default_rng(6020);boot=[]
    for _ in range(5000):
        idx=rng.integers(0,len(fixed),len(fixed));boot.append(100*(1-circuit[idx].mean()/fixed[idx].mean()))
    effect=dict(reduction_percent=100*(1-circuit.mean()/fixed.mean()),paired_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),replicates=5000,
                note='Resample 20 traffic seeds; first average the three training seeds. Conditional on this simulator and demand distribution.')
    (ROOT/'results/summary.json').write_text(json.dumps(dict(summary=summary,effect=effect),indent=2))
    with open(ROOT/'results/summary.csv','w') as f:
        w=csv.writer(f);w.writerow(['model','scenario','queue_seconds_per_arrival','max_residual_queue'])
        for kind in summary:
            for scenario,v in summary[kind].items():w.writerow([kind,scenario,v['mean'],v['max_residual']])
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),layout='constrained')
    for kind,color,style in [('connectome','#7c4038','-'),('rewired','#526975','--'),('mlp','#454545',':')]:
        runs=[r for r in m['runs'] if r['kind']==kind]
        for ax,key in zip(axes,['train_loss','validation_agreement']):
            values=np.array([[h[key] for h in r['history']] for r in runs])
            scale=100 if key=='validation_agreement' else 1
            ax.plot(range(1,31),values.mean(0)*scale,label=LABELS[kind],color=color,ls=style,lw=1.4)
            ax.fill_between(range(1,31),values.min(0)*scale,values.max(0)*scale,color=color,alpha=.08)
    axes[0].set(xlabel='Epoch',ylabel='Training binary cross-entropy');axes[0].legend(frameon=False,fontsize=9)
    axes[1].set(xlabel='Epoch',ylabel='Validation phase agreement (%)',ylim=(75,100.5))
    fig.savefig(ROOT/'figures/training.png',dpi=180);fig.savefig(ROOT/'figures/training.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
    kinds=['fixed','heuristic','connectome','rewired','mlp'];vals=[summary[k]['standard']['mean'] for k in kinds]
    ax.barh([LABELS[k] for k in kinds],vals,color=['#aaa69c','#817b6e','#7c4038','#526975','#454545'])
    ax.invert_yaxis();ax.set_xlabel('Accumulated queue seconds per arrival (lower is better)')
    for i,v in enumerate(vals):ax.text(v+1,i,f'{v:.1f}',va='center',fontsize=10)
    ax.set_xlim(0,max(vals)*1.13);fig.savefig(ROOT/'figures/comparison.png',dpi=180);plt.close(fig)
    geodata=json.loads((ROOT/'data/downtown-signals.geojson').read_text())
    selected=[next(f for f in geodata['features'] if f['properties']['IntersectionName']==name) for name in ['9th & Q','9th & P','9th & O']]
    payload=dict(traces=traces,summary=summary,labels=LABELS,scenarios=SCENARIOS,signals=selected,effect=effect)
    html=(ROOT/'template.html').read_text().replace('/*__DATA__*/','const DATA='+json.dumps(payload,separators=(',',':'))+';')
    for name in ['anatomy','samples','connectivity','training','comparison']:
        html=html.replace('__IMAGE_'+name.upper()+'__','data:image/png;base64,'+base64.b64encode((ROOT/f'figures/{name}.png').read_bytes()).decode())
    table=''
    for kind in LABELS:
        table+='<tr><th scope="row">'+LABELS[kind]+'</th>'+''.join(f'<td>{summary[kind][sc]["mean"]:.1f}</td>' for sc in SCENARIOS)+'</tr>'
    html=html.replace('__RESULT_ROWS__',table)
    html=html.replace('__EFFECT__',f'{effect["reduction_percent"]:.1f}% (paired bootstrap 95% interval: {effect["paired_bootstrap_95"][0]:.1f}–{effect["paired_bootstrap_95"][1]:.1f}%)')
    seed11=next(r for r in m['runs'] if r['kind']=='connectome' and r['seed']==11)
    ablation_rows='<tr><th scope="row">None</th><td>'+f'{np.mean([r["queue_seconds_per_arrival"] for r in seed11["scenarios"]["standard"]]):.1f}'+'</td></tr>'
    for typ,rows in m['ablations_seed11'].items():
        ablation_rows+=f'<tr><th scope="row">Silence {typ}</th><td>{np.mean([r["queue_seconds_per_arrival"] for r in rows]):.1f}</td></tr>'
    html=html.replace('__ABLATION_ROWS__',ablation_rows)
    (ROOT/'index.html').write_text(html)
    report='''# Lincoln traffic control experiment

The MaleCNS circuit was trained by supervised imitation to choose the signal phases recommended by an explicit downstream-aware queue heuristic. It was evaluated on synthetic traffic at three actual Lincoln signal locations: 9th & Q, 9th & P, and 9th & O.

## Evaluation

Accumulated queue seconds per arrival, including every arriving vehicle. Lower is better. Mean over 20 held-out traffic seeds; learned policies also average three training seeds. All runs drained fully within the 60-minute horizon.

| Model | Standard pulse | Higher event demand | Shuttle assumption | P Street restriction |
|---|---:|---:|---:|---:|
'''
    for kind in LABELS:report+='| '+LABELS[kind]+' | '+' | '.join(f'{summary[kind][sc]["mean"]:.2f}' for sc in SCENARIOS)+' |\n'
    report+=f'\nThe circuit reduced simulated queue time relative to the fixed schedule by **{effect["reduction_percent"]:.1f}%**, with a paired bootstrap 95% interval of **{effect["paired_bootstrap_95"][0]:.1f}–{effect["paired_bootstrap_95"][1]:.1f}%**. This interval only describes the modeled traffic distribution. It does not quantify uncertainty about Lincoln traffic.\n'
    report+='''
The queue heuristic, rewired graph, and conventional network perform similarly. The comparison supplies no evidence for a special advantage of fly wiring. The MLP has 5,379 parameters versus 3,819 for each circuit; it is a practical reference, not a parameter-matched ablation. The rewired network is parameter-matched.

## Neural ablations

Mean standard-scenario queue seconds per arrival for the seed-11 circuit, over the same 20 traffic seeds. These are model interventions, not claims about living neurons.

| Intervention | Queue seconds per arrival |
|---|---:|
'''
    seed11=next(r for r in m['runs'] if r['kind']=='connectome' and r['seed']==11)
    report+=f'| None | {np.mean([r["queue_seconds_per_arrival"] for r in seed11["scenarios"]["standard"]]):.2f} |\n'
    for typ,rows in m['ablations_seed11'].items():report+=f'| Silence {typ} | {np.mean([r["queue_seconds_per_arrival"] for r in rows]):.2f} |\n'
    report+='''
## Scope

Signal locations are city GIS data. Arrival rates, queue storage, service capacities, phase plans, and the 25% shuttle-related demand change are assumptions. Current game-day restrictions, turning geometry, pedestrian flows, bus operations, and actual signal timing are not modeled. This is an uncalibrated proof of concept, not a forecast or signal-setting recommendation.

The ten anatomical samples are real SWC neuron skeletons from the 196-node circuit. Their geometries are rendered for interpretation; the network dynamics use connectivity, not the shapes. No existing motion-classifier weights were transferred: the traffic network was trained from scratch on the same graph.

The webpage replays saved seed-5100 evaluation traces, with model seed 11. Scenario controls select already-computed runs. Training is not performed in the browser. Full traces, all checkpoints, source hashes, and scripts accompany the report.
'''
    (ROOT/'results/REPORT.md').write_text(report)
    print(json.dumps(effect,indent=2));print('Built index.html and results/REPORT.md')

if __name__=='__main__':main()
