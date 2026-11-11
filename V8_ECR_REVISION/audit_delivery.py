"""Independent recalculation and DOCX projection audit (no human-review claim)."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import hashlib,json,math,re,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student
from docx import Document
from docx.oxml.ns import qn
B=Path(__file__).resolve().parent;A=B/'ANALYSIS';ROOT=B.parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def recalc(x):
    x=np.asarray(x,float);n=len(x);mean=x.sum()/n;variance=((x-mean)**2).sum()/(n-1);se=math.sqrt(variance/n)
    radius=student.ppf(.975,n-1)*se
    p=2*student.sf(abs(mean/se),n-1) if se>0 else (1. if mean==0 else 0.)
    return mean,mean-radius,mean+radius,p,math.sqrt(variance)

def scientific():
    data=pd.read_csv(B/'V8_RESULTS.csv');budget=pd.read_csv(B/'BUDGET_RESULTS.csv')
    assert len(data)==33840 and len(budget)==5400
    assert not data.duplicated(['condition','environment','scenario','policy','train_seed','eval_seed']).any()
    units=data.groupby(['condition','environment','scenario','policy','train_seed'],as_index=False).mean(numeric_only=True)
    contrasts=pd.read_csv(A/'paired_policy_contrasts.csv');inter=pd.read_csv(A/'condition_interactions.csv')
    maximum=0.;checked=0
    def means(frame,policy,metric):
        x=frame[frame.policy==policy].sort_values('train_seed')[metric].to_numpy()
        if len(x)==1:return np.repeat(x,30)
        assert len(x)==30;return x
    for table,is_interaction in [(contrasts,False),(inter,True)]:
        for _,row in table.iterrows():
            f=units[(units.condition==row.condition)&(units.environment==row.environment)&(units.scenario==row.scenario)]
            if is_interaction:
                control=units[(units.condition=='A_corrected')&(units.environment==row.environment)&(units.scenario==row.scenario)]
                x=means(f,'q_learning',row.metric)-means(f,'contextual_bandit',row.metric)-means(control,'q_learning',row.metric)+means(control,'contextual_bandit',row.metric)
            else:
                policy,baseline=row.comparison.split('_minus_');x=means(f,policy,row.metric)-means(f,baseline,row.metric)
            calculated=recalc(x);observed=[row['mean'],row.ci95_low,row.ci95_high,row.p_raw,row.sd_training]
            maximum=max(maximum,float(np.max(np.abs(np.asarray(calculated)-observed))));checked+=1
        for family,g in table[table.family.notna()].groupby('family'):
            ordered=g.sort_values('p_raw');adjusted=[];last=0
            for k,(_,r) in enumerate(ordered.iterrows()):
                last=max(last,min(1,(len(g)-k)*r.p_raw));adjusted.append(last)
            assert np.allclose(ordered.p_holm,adjusted,rtol=0,atol=1e-12)
            rad=student.ppf(1-.05/(2*len(g)),29)*g.sd_training/np.sqrt(30)
            assert np.allclose(g.simultaneous_low,g['mean']-rad) and np.allclose(g.simultaneous_high,g['mean']+rad)
    assert maximum<1e-10,maximum
    for condition in ['A_corrected','A_v7_exact']:
        g=contrasts[(contrasts.condition==condition)&(contrasts.metric=='mean_reward')&(contrasts.comparison=='q_learning_minus_contextual_bandit')]
        assert len(g)==12 and (g['mean']<0).all() and (g.p_holm<.05).all()
    mechanism=inter[inter.metric=='mean_reward'];assert len(mechanism)==12 and (mechanism.p_holm>=.05).all()
    required={'mean_reward','legitimate_contributions_accepted','security_incidents','rejected_legitimate_innovation','human_review_workload','governance_expenditure','safety_guard_interventions'}
    assert required.issubset(set(contrasts.metric))
    bunit=budget.groupby(['scenario','policy','budget','train_seed']).mean(numeric_only=True).reset_index()
    bg=pd.read_csv(A/'budget_policy_gaps.csv');bc=pd.read_csv(A/'budget_gap_changes.csv')
    for table,change in [(bg,False),(bc,True)]:
        for _,r in table.iterrows():
            pol=r.comparison.split('_minus_')[0]
            def gap(k):
                g=bunit[(bunit.scenario==r.scenario)&(bunit.budget==k)]
                p=g.pivot(index='train_seed',columns='policy',values='mean_reward')
                return p[pol]-p.contextual_bandit
            x=gap(r.budget)-(gap(180) if change else 0);values=recalc(x)
            assert np.allclose(values[:4],[r['mean'],r.ci95_low,r.ci95_high,r.p_raw],rtol=0,atol=1e-10)
    # Verify every nested 180-episode prefix using raw common numerical outcomes.
    primary=data[(data.condition=='A_corrected')&(data.environment=='sequential')]
    prefix=budget[budget.budget==180];keys=['scenario','policy','train_seed','eval_seed']
    joined=prefix.merge(primary,on=keys,suffixes=('_budget','_primary'),validate='one_to_one')
    assert len(joined)==1800
    prefix_max=0
    for col in set(primary.select_dtypes('number').columns)&set(prefix.select_dtypes('number').columns)-set(keys):
        prefix_max=max(prefix_max,float((joined[col+'_budget']-joined[col+'_primary']).abs().max()))
    assert prefix_max<1e-10
    assert data[data.implementation=='v8_corrected'].residual_pending_items.max()==0 and budget.residual_pending_items.max()==0
    comp=['current_action_reward_total','delayed_past_reward_total','collective_state_reward_total','terminal_liquidation_reward_total','guard_penalty_total']
    assert np.abs(data.total_reward-data[comp].sum(axis=1)).max()<1e-9
    assert np.abs(budget.total_reward-budget[comp].sum(axis=1)).max()<1e-9
    figures={2:pd.read_csv(B/'FIGURES'/'Fig2_source_data.csv'),3:pd.read_csv(B/'FIGURES'/'Fig3_source_data.csv'),4:pd.read_csv(B/'FIGURES'/'Fig4_source_data.csv')}
    for n,rows in figures.items():
        ref=contrasts if n==2 else inter if n==3 else bg
        for _,r in rows.iterrows():
            selectors=['condition','environment','scenario','comparison','metric'] if n<4 else ['scenario','budget','comparison']
            g=ref
            for k in selectors:g=g[g[k]==r[k]]
            assert len(g)==1
            for c in ['mean','ci95_low','ci95_high']:assert abs(float(g.iloc[0][c])-r[c])<1e-12
    return dict(primary_rows=len(data),budget_snapshot_rows=len(budget),independently_recalculated_contrasts=checked,maximum_numeric_error=maximum,prefix_rows_verified=len(joined),prefix_maximum_error=prefix_max)

def accepted_runs(element):
    out=[]
    for run in element.findall(qn('w:r')):
        style=run.find('./'+qn('w:rPr')+'/'+qn('w:rStyle'))
        if style is not None and style.get(qn('w:val'))=='ECRDeleted':continue
        for child in run:
            if child.tag==qn('w:t'):out.append(child.text or '')
            elif child.tag==qn('w:br'):out.append('\n')
            elif child.tag==qn('w:tab'):out.append('\t')
    return ''.join(out)

def projection(doc):
    out=[]
    for child in doc._element.body:
        if child.tag==qn('w:p'):
            st=child.find('./'+qn('w:pPr')+'/'+qn('w:pStyle'))
            if st is not None and st.get(qn('w:val'))=='ECRDeletedBlock':continue
            images=child.xpath('.//a:blip')
            if images:
                out.append(('figure',sha_bytes(doc.part.related_parts[images[0].get(qn('r:embed'))].blob)))
            else:
                text=accepted_runs(child)
                if text:out.append(('p',text))
        elif child.tag==qn('w:tbl'):
            desc=child.find('./'+qn('w:tblPr')+'/'+qn('w:tblDescription'))
            if desc is not None and desc.get(qn('w:val'))=='ECRDeletedTable':continue
            rows=[]
            for row in child.findall(qn('w:tr')):
                rows.append(tuple('\n'.join(accepted_runs(p) for p in cell.findall(qn('w:p'))) for cell in row.findall(qn('w:tc'))))
            out.append(('table',tuple(rows)))
    return out
def sha_bytes(data):return hashlib.sha256(data).hexdigest()

def documents():
    clean=Document(B/'ECR_MANUSCRIPT_FINAL.docx');tracked=Document(B/'ECR_MANUSCRIPT_TRACKED.docx')
    cp=projection(clean);tp=projection(tracked)
    assert cp==tp,'Clean and accepted redline content differ'
    for d in [clean,tracked]:
        for shape in d.inline_shapes:
            assert shape.width<=d.sections[0].page_width-d.sections[0].left_margin-d.sections[0].right_margin+10
    text='\n'.join(p.text for p in clean.paragraphs)
    for forbidden in [r'under explicit synth\b',r'The intercoder theoretical implication',r'Information Systems Frontiers’ focus',r'\{\{']:
        assert not re.search(forbidden,text),forbidden
    before_refs=text.split('\nReferences\n')[0]
    leftover=re.findall(r'(?:et al\.|\w+\s+(?:and|&)\s+\w+)\s*\(\d{4}\)|(?:[A-Z][a-z]+),\s*(?:19|20)\d{2}',before_refs)
    assert not leftover,('Unconverted author-year citations',leftover)
    numbers={int(k) for group in re.findall(r'\[([\d, ]+)\]',before_refs) for k in re.findall(r'\d+',group)}
    use=json.loads((B/'REFERENCE_USAGE.json').read_text(encoding='utf-8'));assert numbers==set(range(1,len(use['order'])+1))
    abstract=next(p.text.split('\n',1)[1] for p in clean.paragraphs if p.text.startswith('Abstract\n'))
    assert 100<=len(abstract.split())<=150
    assert any(p.text=='6 Conclusion' for p in clean.paragraphs)
    colors={};deleted=0;added=0
    for node in tracked._element.body.xpath('.//w:r'):
        st=node.find('./'+qn('w:rPr')+'/'+qn('w:rStyle'))
        if st is None:continue
        name=st.get(qn('w:val'))
        if name in ['ECRAdded','ECRDeleted','ECRUnchanged']:
            color=node.find('./'+qn('w:rPr')+'/'+qn('w:color'));assert color is not None
            expected={'ECRAdded':'0000FF','ECRDeleted':'FF0000','ECRUnchanged':'000000'}[name]
            assert color.get(qn('w:val')).upper()==expected
            if name=='ECRDeleted':
                assert node.find('./'+qn('w:rPr')+'/'+qn('w:strike')) is not None;deleted+=1
            if name=='ECRAdded':added+=1
    # Tables 2/3 must exactly match independent reconstruction from analysis CSVs.
    c=pd.read_csv(A/'paired_policy_contrasts.csv');inter=pd.read_csv(A/'condition_interactions.csv')
    levels=pd.read_csv(A/'policy_levels.csv')
    sourceblocks=json.loads((B/'MANUSCRIPT_BLOCKS.json').read_text(encoding='utf-8'))
    tables=[b for b in sourceblocks if b['kind']=='table'];assert len(tables)==3
    scenarios=['balanced_reference','high_security_pressure','high_review_capacity','constrained_review_capacity','high_false_restriction_cost','participation_sensitive_ecosystem']
    for row,s in zip(tables[1]['rows'][1:],scenarios):
        g=c[(c.condition=='A_corrected')&(c.environment=='sequential')&(c.scenario==s)&(c.comparison=='q_learning_minus_contextual_bandit')&(c.metric=='mean_reward')].iloc[0]
        assert row[3]==f'{g["mean"]:.4f} [{g.simultaneous_low:.4f}, {g.simultaneous_high:.4f}]'
        for col,policy in [(1,'contextual_bandit'),(2,'q_learning')]:
            r=levels[(levels.condition=='A_corrected')&(levels.environment=='sequential')&(levels.scenario==s)&(levels.policy==policy)&(levels.metric=='mean_reward')].iloc[0]
            assert row[col]==f'{r["mean"]:.4f} ({r.sd_training:.4f})'
    k=1
    for condition in ['B_loss25','B_loss50','C_reduced','C_enriched']:
        for s in [scenarios[0],scenarios[1],scenarios[3]]:
            r=inter[(inter.condition==condition)&(inter.scenario==s)&(inter.metric=='mean_reward')].iloc[0]
            assert tables[2]['rows'][k][2]==f'{r["mean"]:.4f} [{r.simultaneous_low:.4f}, {r.simultaneous_high:.4f}]'
            assert tables[2]['rows'][k][3]==f'{r.p_holm:.4f}';k+=1
    return dict(clean_and_redline_accepted_projection_identical=True,projection_blocks=len(cp),abstract_words=len(abstract.split()),cited_references=len(use['order']),added_runs=added,deleted_runs=deleted,tables_verified=3,figures=4,word_pagination='Not verified with a native Word/PDF renderer; geometry and content checked')

def preservation():
    checks={}
    for name,base in [('V7_PRESERVATION_SHA256.json',ROOT),('V8_ARTIFACT_SHA256.json',ROOT/'V8_minimal_extension')]:
        ledger=json.loads((B/'ORIGINAL_V8_PROTOCOL'/name).read_text());changed=[]
        for path,h in ledger.items():
            p=base/Path(path)
            if not p.exists() or sha(p)!=h:changed.append(path)
        assert not changed,(name,changed);checks[name]=len(ledger)
    provenance=json.loads((B/'SOURCE_PROVENANCE.json').read_text())
    path=Path(provenance.get('reference_source',provenance.get('path',provenance.get('source_path',''))))
    assert path.is_file(),provenance
    assert sha(path)==provenance['sha256']
    return checks

def main():
    if '--scientific-only' in sys.argv:
        print(json.dumps(dict(status='passed_independent_recalculation',scientific=scientific()),indent=2));return
    result=dict(status='passed_computational_checks',scientific=scientific(),documents=documents(),preservation=preservation(),independence='Separate recalculation code; not a second human or external scientific review',remaining=['Author declaration confirmation and ORCID','Author source verification and final approval','Native Word pagination review','Deposit nested-budget supplement with submission materials'])
    (B/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
