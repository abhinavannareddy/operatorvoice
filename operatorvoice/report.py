"""Builds a single self-contained HTML report (no server, no network) for exploring the ranking.

The page re-ranks in the browser when the factor weights change, and each cluster can be opened to
see the posts and the matched words behind its pain and feasibility scores."""

import json
from collections import Counter
from pathlib import Path

POSTS_PER_CLUSTER = 40  # keeps the file small; the CSV export has every post


def build_payload(result: dict, uncertainty) -> dict:
    posts, opp = result["posts"], result["opportunities"].merge(uncertainty, on=["rank", "cluster"])
    clusters = []
    for row in opp.itertuples():
        members = posts[posts.cluster == row.cluster]
        top = members.sort_values("pain", ascending=False).head(POSTS_PER_CLUSTER)
        counts = Counter()
        for terms in members.sw_terms:
            counts.update(("software", t) for t in terms)
        for terms in members.hw_terms:
            counts.update(("hardware", t) for t in terms)
        evidence = [{"kind": k, "term": t, "posts": n} for (k, t), n in counts.most_common(8)]
        clusters.append(
            {
                "id": int(row.cluster),
                "terms": row.top_terms,
                "posts": int(row.posts),
                "share": float(row.share_pct),
                "demand": float(row.demand),
                "pain": float(row.pain),
                "feasibility": float(row.feasibility),
                "priority": float(row.priority),
                "lo": float(row.priority_lo),
                "hi": float(row.priority_hi),
                "p_top": float(row.p_top_n),
                "evidence": evidence,
                "samples": [
                    {
                        "text": r.text,
                        "rating": None if r.rating != r.rating else float(r.rating),
                        "pain": round(float(r.pain), 2),
                        "neg": r.neg_terms,
                        "sw": r.sw_terms,
                        "hw": r.hw_terms,
                    }
                    for r in top.itertuples()
                ],
            }
        )
    return {"n_posts": int(len(posts)), "k": int(result["k"]), "clusters": clusters}


def render(payload: dict) -> str:
    # "<" is escaped so post text containing "</script>" cannot break out of the data block.
    data = json.dumps(payload).replace("<", "\\u003c")
    return TEMPLATE.replace("__DATA__", data)


def write_report(result: dict, uncertainty, path) -> Path:
    path = Path(path)
    path.write_text(render(build_payload(result, uncertainty)), encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OperatorVoice report</title>
<style>
:root{--bg:#fafaf8;--fg:#1d2024;--mut:#5b6470;--card:#fff;--line:#d9dde3;--acc:#1f6feb;--sw:#1a7f4b;--hw:#b4540a;--neg:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#14171b;--fg:#e6e8eb;--mut:#9aa4af;--card:#1c2026;--line:#323943;--acc:#6ea8ff;--sw:#4cc38a;--hw:#f0a05a;--neg:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.5rem;margin:0}h2{font-size:1.1rem;margin:0 0 8px}
p.lead{color:var(--mut);margin:4px 0 20px;max-width:70ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px}
.sliders{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
label{display:block;font-weight:600}input[type=range]{width:100%}output{color:var(--mut);font-weight:400}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:middle}
th{font-size:.8rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}td.n{font-variant-numeric:tabular-nums}
.rowbtn{all:unset;cursor:pointer;color:var(--acc);text-decoration:underline;display:block}
.rowbtn:focus-visible,input:focus-visible,button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
tr[aria-current=true]{background:color-mix(in srgb,var(--acc) 10%,transparent)}
.bar{position:relative;height:14px;background:var(--line);border-radius:3px;min-width:120px}
.bar i{position:absolute;top:0;bottom:0;background:var(--acc);border-radius:3px}
.bar b{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--fg)}
.chip{display:inline-block;border:1px solid currentColor;border-radius:999px;padding:0 8px;margin:2px 4px 2px 0;font-size:.8rem}
.chip.software{color:var(--sw)}.chip.hardware{color:var(--hw)}.chip.neg{color:var(--neg)}
.post{border-top:1px solid var(--line);padding:8px 0}.mut{color:var(--mut);font-size:.85rem}
.scroll{overflow-x:auto}
</style></head><body><main>
<h1>OperatorVoice report</h1>
<p class="lead" id="summary"></p>
<section class="card" aria-labelledby="w"><h2 id="w">Priority = demand<sup>a</sup> &times; pain<sup>b</sup> &times; feasibility<sup>c</sup></h2>
<p class="mut">Exponent 1 is the default formula, 0 ignores a factor, 2 makes it count more. Rankings re-sort as you move a slider.
At the default weights the bar shows the 5-95% bootstrap range with the point estimate marked; at other weights it shows the
score relative to the top cluster.</p>
<div class="sliders" id="sliders"></div><p><button id="reset" type="button">Reset to defaults</button></p></section>
<section class="card scroll" aria-label="Ranked opportunities"><table><thead><tr><th>#</th><th>Cluster terms</th><th>Posts</th><th>Pain</th><th>Feasibility</th>
<th>Priority</th><th>Top-3 odds</th></tr></thead><tbody id="rows"></tbody></table></section>
<section class="card" id="detail" aria-live="polite" tabindex="-1"><p class="mut">Select a cluster to see the evidence behind its scores.</p></section>
<p class="mut">This file embeds the text of the posts shown above. Share it only if you may share the source feedback.
Top-3 odds is the share of bootstrap resamples (posts resampled inside fixed clusters) in which the cluster is in the top three at the
default weights; it does not capture uncertainty from the clustering itself.</p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const LABELS={a:'Demand exponent (a)',b:'Pain exponent (b)',c:'Feasibility exponent (c)'};
const w={a:1,b:1,c:1};let selected=null;
const $=(t,p={},...k)=>{const e=document.createElement(t);Object.assign(e,p);k.forEach(x=>e.append(x));return e};
const score=c=>Math.pow(c.demand,w.a)*Math.pow(c.pain,w.b)*Math.pow(c.feasibility,w.c);
document.getElementById('summary').textContent=`${D.n_posts} posts grouped into ${D.k} clusters. Scores come from TF-IDF clustering and keyword rubrics; see the limits in the README.`;
const sl=document.getElementById('sliders');
for(const key of Object.keys(LABELS)){
 const out=$('output',{htmlFor:'s'+key,textContent:' 1.00'});
 const inp=$('input',{type:'range',id:'s'+key,min:0,max:2,step:.25,value:1});
 inp.oninput=()=>{w[key]=+inp.value;out.textContent=' '+(+inp.value).toFixed(2);render()};
 sl.append($('div',{},$('label',{htmlFor:'s'+key},LABELS[key],out),inp))}
document.getElementById('reset').onclick=()=>{for(const k in w){w[k]=1;document.getElementById('s'+k).value=1}
 sl.querySelectorAll('output').forEach(o=>o.textContent=' 1.00');render()};
function render(){
 const order=[...D.clusters].sort((x,y)=>score(y)-score(x));
 const top=Math.max(...order.map(score),1e-9),maxHi=Math.max(...D.clusters.map(c=>c.hi),1e-9);
 const def=w.a===1&&w.b===1&&w.c===1;
 const tb=document.getElementById('rows');tb.replaceChildren();
 order.forEach((c,i)=>{
  const btn=$('button',{className:'rowbtn',type:'button',textContent:c.terms,onclick:()=>{selected=c.id;show(c);render()}});
  const fill=$('i'),mark=$('b');
  const bar=$('div',{className:'bar',role:'img',ariaLabel:`priority ${score(c).toFixed(3)}`},fill);
  if(def){fill.style.left=100*c.lo/maxHi+'%';fill.style.width=Math.max(1,100*(c.hi-c.lo)/maxHi)+'%';mark.style.left=100*c.priority/maxHi+'%';bar.append(mark)}
  else{fill.style.left='0';fill.style.width=100*score(c)/top+'%'}
  const tr=$('tr',{},$('td',{className:'n',textContent:i+1}),$('td',{},btn),$('td',{className:'n',textContent:`${c.posts} (${c.share}%)`}),
   $('td',{className:'n',textContent:c.pain.toFixed(2)}),$('td',{className:'n',textContent:c.feasibility.toFixed(2)}),$('td',{},bar),
   $('td',{className:'n',textContent:def?Math.round(100*c.p_top)+'%':'n/a'}));
  if(c.id===selected)tr.setAttribute('aria-current','true');
  tb.append(tr)})}
function chips(items,cls){return items.map(t=>$('span',{className:'chip '+cls,textContent:t}))}
function show(c){
 const d=document.getElementById('detail');
 d.replaceChildren($('h2',{textContent:c.terms}),
  $('p',{className:'mut',textContent:`${c.posts} posts. demand ${c.demand.toFixed(2)}, pain ${c.pain.toFixed(2)}, feasibility ${c.feasibility.toFixed(2)}. Priority 5-95% range ${c.lo} to ${c.hi}.`}),
  $('p',{className:'mut',textContent:'Feasibility = share of posts naming software things minus 0.8 x share naming hardware things. Terms driving it in this cluster:'}),
  $('div',{},...(c.evidence.length?c.evidence.map(e=>$('span',{className:'chip '+e.kind,textContent:`${e.term} (${e.kind}, ${e.posts} posts)`})):[$('span',{className:'mut',textContent:'No configured terms matched.'})])),
  $('h2',{textContent:`Highest-pain posts (${c.samples.length} of ${c.posts})`,style:'margin-top:16px'}),
  ...c.samples.map(p=>$('div',{className:'post'},$('div',{textContent:p.text}),
   $('div',{className:'mut',textContent:`pain ${p.pain}`+(p.rating===null?'':`, rating ${p.rating}`)}),
   $('div',{},...chips(p.neg,'neg'),...chips(p.sw,'software'),...chips(p.hw,'hardware')))));
 d.focus()}
render();
</script></body></html>
"""
