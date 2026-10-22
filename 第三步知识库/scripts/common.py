import json,hashlib,time,urllib.request,pickle
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'第三步知识库'
CORPUS=ROOT/'第二步材料处理/复核修订版/chunks.jsonl'
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def api(path,payload=None):
 req=urllib.request.Request('http://127.0.0.1:11434/api/'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=600) as r:return json.load(r)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def embed(texts):
 v=np.asarray(api('embed',{'model':'qwen3-embedding:0.6b','input':texts,'truncate':False,'keep_alive':'30m','options':{'num_ctx':4096}})['embeddings'],dtype='float32')
 assert v.shape==(len(texts),1024) and np.isfinite(v).all()
 norms=np.linalg.norm(v,axis=1,keepdims=True);assert (norms>0).all();return v/norms
