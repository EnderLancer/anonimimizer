# Будує data.js для візуалізації лінкування "особа -> її дані".
# Використовує gold-анотації test-набору ЛИШЕ для демонстрації (не для навчання).
import json, re, sys
sys.stdout.reconfigure(encoding='utf8')
SRC='../data/test/test/court_cases.jsonl'
rows=[json.loads(l) for l in open(SRC,encoding='utf8')]
def kind(p): return p.rsplit('_',1)[0]
def link(anns, text):
    """евристика: дані -> найближча попередня особа в межах 150 симв. без розриву абзацу,
    інакше найближча наступна в межах 60 симв., інакше без власника"""
    persons=[a for a in anns if kind(a['pseudonym'])=='ОСОБА']
    out=[]
    for a in anns:
        k=kind(a['pseudonym']); owner=None; dist=None
        if k!='ОСОБА' and persons:
            prev=[p for p in persons if p['end']<=a['start'] and a['start']-p['end']<=150 and '\n' not in text[p['end']:a['start']]]
            nxt=[p for p in persons if p['start']>=a['end'] and p['start']-a['end']<=60 and '\n' not in text[a['end']:p['start']]]
            if prev: p=max(prev,key=lambda p:p['end']); owner=p['pseudonym']; dist=a['start']-p['end']
            elif nxt: p=min(nxt,key=lambda p:p['start']); owner=p['pseudonym']; dist=p['start']-a['end']
        out.append(dict(start=a['start'],end=a['end'],pseudonym=a['pseudonym'],type=k,owner=owner if k!='ОСОБА' else a['pseudonym'],dist=dist))
    return out
cands=[]
for r in rows:
    an=r['annotations']; ks={kind(a['pseudonym']) for a in an}
    n_p=len({a['pseudonym'] for a in an if kind(a['pseudonym'])=='ОСОБА'})
    if 2<=n_p<=6 and len(ks)>=3 and 2500<len(r['original'])<6500:
        cands.append((len(ks),sum(kind(a['pseudonym'])!='ОСОБА' for a in an),r))
cands.sort(key=lambda t:(-t[0],-t[1]))
docs=[]
for _,_,r in cands[:5]:
    docs.append(dict(id=r['id'],text=r['original'],ents=link(sorted(r['annotations'],key=lambda a:a['start']),r['original'])))
open('data.js','w',encoding='utf8').write('const DOCS='+json.dumps(docs,ensure_ascii=False)+';')  # після цього вставити data.js у index.html
for d in docs: print(d['id'],len(d['ents']),sum(e['owner'] is None for e in d['ents']))
