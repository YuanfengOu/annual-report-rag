from common import *
import jieba
idx=OUT/'index';idx.mkdir(exist_ok=True);parts=idx/'batches';parts.mkdir(exist_ok=True)
cs=read(CORPUS);sha=hashlib.sha256(CORPUS.read_bytes()).hexdigest();tags=api('tags');digest=next(m['digest'] for m in tags['models'] if m['name']=='qwen3-embedding:0.6b')
fingerprint={'corpus_sha256':sha,'count':len(cs),'embedding_digest':digest,'dimension':1024,'text_field':'retrieval_text','truncate':False,'num_ctx':4096,'batch_size':8}
meta=idx/'build_config.json'
if meta.exists():assert json.loads(meta.read_text(encoding='utf8'))==fingerprint
else:dump(meta,fingerprint)
start=time.time();last=0
for i in range(0,len(cs),8):
 f=parts/f'{i:05d}.npy'
 if not f.exists():
  v=embed([c['retrieval_text'] for c in cs[i:i+8]]);np.save(f,v)
 if time.time()-last>25:print(f'{min(i+8,len(cs))}/{len(cs)} elapsed={time.time()-start:.0f}s',flush=True);last=time.time()
v=np.concatenate([np.load(parts/f'{i:05d}.npy') for i in range(0,len(cs),8)])
assert v.shape==(len(cs),1024);np.save(idx/'vectors.npy',v)
(idx/'chunks.jsonl').write_bytes(CORPUS.read_bytes())
jieba.dt.tmp_dir=str(OUT);tokens=[list(jieba.cut(c['retrieval_text'].lower())) for c in cs]
from rank_bm25 import BM25Okapi
with (idx/'bm25.pkl').open('wb') as f:pickle.dump(BM25Okapi(tokens),f)
dump(idx/'manifest.json',{'configuration':fingerprint,'models':tags,'elapsed_seconds':time.time()-start,'vector_shape':list(v.shape),'vector_norm_min':float(np.linalg.norm(v,axis=1).min()),'vector_norm_max':float(np.linalg.norm(v,axis=1).max()),'companies':sorted({c['company'] for c in cs}),'status':'complete'})
api('embed',{'model':'qwen3-embedding:0.6b','input':[],'keep_alive':0})
print('INDEX COMPLETE',flush=True)
