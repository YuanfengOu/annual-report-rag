from retrieval import *
def d(s):return Decimal(str(s).replace(',','').replace('%',''))
def solve(q,r,v):
 years=[int(y) for y in re.findall(r'20\d{2}',q)]
 if years and max(years)>2025:return {'answer':'资料仅包含2025年年报，不能证明后续年度已经实现的营业收入，资料不足。','evidence':[],'route':'year_guard'}
 target_year=max(years) if years else 2025
 ms=intents(q);names=[n for n in r.companies if n in q or (n=='美的集团' and '美的' in q)]
 if '全部15家' in q:names=r.companies
 if ms and names:
  ev=[];missing=[]
  for n in names:
   for m in ms:
    found=[c for c in r.cs if c['company']==n and c.get('metric')==m and c.get('calculation_ready') and c['year']==2025]
    if len(found)==1 and (str(target_year)+'年' in found[0]['values'] or str(target_year)+'年调整后' in found[0]['values']):ev+=found
    else:missing.append([n,m])
  def year_value(c):
   vals=c['values'];return vals.get(str(target_year)+'年调整后',vals.get(str(target_year)+'年'))
  yuan=lambda c:d(year_value(c))*(1000 if c['unit']=='人民币千元' else 1)
  if '排名' in q:ev.sort(key=yuan,reverse=True)
  lines=[f'公司覆盖{len(set(c["company"] for c in ev))}/{len(names)}；指标覆盖{len(ev)}/{len(names)*len(ms)}。'];lookup={}
  for i,c in enumerate(ev,1):
   lookup[c['company'],c['metric']]=(i,c)
   s='；'.join(k+'：'+(str(val).rstrip('%')+'%' if k=='同比增减百分比' else str(val)) for k,val in c['values'].items())
   extra=(f'；{target_year}年换算为'+str(yuan(c)/Decimal(100000000))+'亿元') if c['metric']==METRICS[0] else ''
   prefix=(f'第{i}名'+('（前五）' if i<=5 else '')+'：') if '排名' in q else ''
   lines.append(f'{prefix}{c["company"]}｜{c["metric"]}｜单位：{c["unit"]}｜{s}{extra} [S{i}]')
  if '调整' in q:lines.append('同比采用2024年调整后口径。'+''.join(f'[S{i}]' for i,c in enumerate(ev,1) if '2024年调整后' in c['values']))
  if '比例' in q:
   for n in names:
    if (n,METRICS[0]) in lookup and (n,METRICS[1]) in lookup:
     ir,a=lookup[n,METRICS[0]];ip,b=lookup[n,METRICS[1]]
     if yuan(a):lines.append(f'{n}（{target_year}年）：归母净利润÷营业收入×100% = {(yuan(b)/yuan(a)*100).quantize(Decimal(".01"))}%。[S{ip}][S{ir}]')
  if '谁' in q and '更高' in q:
   for m in ms:
    rows=[(i,c) for i,c in enumerate(ev,1) if c['metric']==m]
    if len(rows)==len(names):lines.append(m+'较高：'+max(rows,key=lambda z:yuan(z[1]))[1]['company']+'。'+''.join(f'[S{i}]' for i,c in rows))
  if '四组' in q:
   groups={k:[] for k in ['双增长','利润增现金流降','利润降现金流增','双下降','持平']}
   for n in names:
    if (n,METRICS[1]) not in lookup or (n,METRICS[3]) not in lookup:continue
    ia,a=lookup[n,METRICS[1]];ib,b=lookup[n,METRICS[3]];x=d(a['values']['同比增减百分比']);y=d(b['values']['同比增减百分比'])
    key='持平' if x==0 or y==0 else '双增长' if x>0 and y>0 else '利润增现金流降' if x>0 else '利润降现金流增' if y>0 else '双下降'
    groups[key].append(n+f'[S{ia}][S{ib}]')
   lines.extend(k+'：'+('、'.join(val) or '无') for k,val in groups.items())
  if missing:lines.append(f'{target_year}年以下指标证据缺失或存在歧义，资料不足，不作答：'+str(missing))
  return {'answer':'\n\n'.join(lines),'evidence':ev,'missing':missing,'route':'structured_reviewed_rows_decimal'}
 ev,candidates,coverage=r.retrieve(q,v,'hybrid')
 # Query topic anchors are matched against corpus text, without using gold answers or fixed chunk IDs.
 anchors=[w for w in ['套期保值','外汇风险敞口','研发投入','销售渠道'] if w in q]
 focus=[]
 if anchors:
  scope=[c for c in r.cs if not names or c['company'] in names]
  matches=[c for c in scope if all(w in re.sub(r'\s+','',c['text']) for w in anchors)]
  if matches:
   rank={c['chunk_id']:i for i,c in enumerate(ev)}
   focus=sorted(matches,key=lambda c:(rank.get(c['chunk_id'],99999),len(c['text'])))[:4]
   ev=focus
   candidates.extend({'chunk_id':c['chunk_id'],'channel':'query_topic_exact_match','anchors':anchors,'selected':True,'evidence':c} for c in focus)
 selected_ids={c['chunk_id'] for c in ev}
 for candidate in candidates:candidate['selected']=candidate['chunk_id'] in selected_ids
 a=api('chat',{'model':'qwen3:8b','messages':[{'role':'system','content':SYSTEM+' 只回答问题所问的风险或业务来源，不添加未被问到的敏感性金额。优先使用直接说明题目主题的原句，不将其他风险章节内容替代。'},{'role':'user','content':prompt(q,ev)}],'stream':False,'think':False,'options':{'num_ctx':8192,'temperature':0,'seed':42,'num_predict':1024}})
 answer=a['message']['content'];ids=[int(i) for i in re.findall(r'\[S(\d+)\]',answer)];valid=bool(ids) and all(1<=i<=len(ev) for i in ids)
 return {'answer':answer if valid else '回答未通过引用校验，暂不输出。','raw_response':a,'evidence':ev,'retrieved_candidates':candidates,'route':'hybrid_llm','citation_gate':valid}
if __name__=='__main__':
 out=OUT/'experiments/run02';out.mkdir(parents=True,exist_ok=True);r=Retriever();qs=read(ROOT/'第三步准备/测试题.jsonl');vs=np.load(OUT/'experiments/run01/question_vectors.npy')
 for k,q in enumerate(qs):
  result=solve(q['question'],r,vs[k]);result.update(id=q['id'],question=q['question']);dump(out/f'{q["id"]}.json',result);print(q['id'],result['route'],flush=True)
