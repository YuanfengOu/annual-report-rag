from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).parent/'scripts'))
import streamlit as st
import hashlib,importlib.util
pipeline_path=Path(__file__).parent/'scripts/repaired_pipeline.py'
pipeline_sha256=hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
@st.cache_resource
def load_pipeline(digest):
 spec=importlib.util.spec_from_file_location('qa_pipeline_'+digest,pipeline_path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
pipeline=load_pipeline(pipeline_sha256)
Retriever,solve,intents,embed,api,np,OUT,ROOT=(getattr(pipeline,k) for k in ['Retriever','solve','intents','embed','api','np','OUT','ROOT'])
st.set_page_config(page_title='家电行业年报问答',page_icon='📊',layout='wide')
st.title('家电行业年报问答')
st.caption('15家公司 · 2025年年度报告 · 回答附原文证据 · 页面版本 20261022.3')
st.info('财务指标已核验范围：营业收入、归母净利润、扣非归母净利润、经营现金流量净额。其他文字问答请结合原文核对。')
@st.cache_resource
def retriever(digest):return Retriever()
r=retriever(pipeline_sha256)
with st.sidebar:
 st.header('提问示例')
 examples=['美的集团2025年营业收入是多少？给出原单位和亿元数值及引用。','格力电器2024年归母净利润占营业收入的比例是多少？','覆盖全部15家公司，按2025年营业收入从高到低排名。','美的2025年年报披露的套期保值主要针对哪两类风险？外汇风险敞口来自哪些业务？','海尔智家2028年全年已经实现的营业收入是多少？']
 chosen=st.selectbox('选择示例',examples)
 if st.button('填入示例'):st.session_state['question']=chosen
 st.caption('报告年度为2025年；部分核心表提供2023—2024年比较数据。')
question=st.text_area('请输入问题',key='question',height=110,placeholder='例如：美的集团2025年营业收入是多少？')
if st.button('查询',type='primary'):
 q=question.strip()
 st.session_state.pop('result',None)
 if not q:st.warning('请先输入问题。')
 else:
  with st.spinner('正在查找证据并生成回答…'):
   try:
    years={int(y) for y in __import__('re').findall(r'20\d{2}',q)}
    if len(years)>1 and max(years)<=2025 and not (years=={2024,2025} and any(w in q for w in ['同比','调整前','调整后']) and not any(w in q for w in ['比例','占比','比值'])):st.session_state['result']={'answer':'当前页面的结构化计算每次支持一个目标年份，请分别提问后比较。','evidence':[],'route':'multi_year_guard','question':q}
    else:
     names=[n for n in r.companies if n in q or (n=='美的集团' and '美的' in q)]
     structured=bool(intents(q) and (names or '全部15家' in q))
     v=np.zeros(1024,dtype='float32')
     if not structured and not(years and max(years)>2025):
      v=embed(['Instruct: Given a question about Chinese listed company annual reports, retrieve relevant passages that answer the question.\nQuery: '+q+' '+' '.join(intents(q))])[0]
      api('embed',{'model':'qwen3-embedding:0.6b','input':[],'keep_alive':0})
     start=time.time();result=solve(q,r,v);result.update(question=q,elapsed_seconds=round(time.time()-start,2),pipeline_sha256=pipeline_sha256,page_version="20261022.3");st.session_state['result']=result
   except Exception as exc:st.error('本次查询未完成。请确认本地模型服务已启动。');st.caption(str(exc))
if 'result' in st.session_state:
 result=st.session_state['result'];st.subheader('回答');st.caption(result['question']);st.markdown(result['answer'])
 st.download_button('下载本次回答与证据记录',json.dumps(result,ensure_ascii=False,indent=2),file_name='年报问答记录.json',mime='application/json')
 st.subheader('引用与原文')
 if not result['evidence']:st.caption('本次未提供事实性证据引用。')
 for i,c in enumerate(result['evidence'],1):
  with st.expander(f'S{i} · {c["company"]} · PDF第{c["pdf_page"]}页'):
   st.caption(f'文件：{c["source_file"]}｜印刷页码：{c.get("printed_page") or "未识别"}｜{c["chunk_id"]}')
   st.text(c['text'])
   if c.get('source_url'):st.link_button('查看公告来源',c['source_url'])
   base=(ROOT/'年报原文').resolve();pdf=(base/c['source_file']).resolve()
   if pdf.parent==base and pdf.is_file():st.download_button('下载原PDF（按上方PDF页码定位）',pdf.read_bytes(),file_name=pdf.name,key=f'pdf_{i}')
 with st.expander('本次回答方式'):
  st.write({'structured_reviewed_rows_decimal':'已核验财务行查询，程序计算并绑定引用。','hybrid_llm':'向量与BM25混合检索，由本地模型依据召回证据回答。','year_guard':'年份证据不足，拒答。','multi_year_guard':'多年份计算暂不支持。'}.get(result['route'],result['route']))
