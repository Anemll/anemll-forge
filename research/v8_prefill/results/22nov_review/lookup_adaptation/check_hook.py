"""CPU-only contract check. The synthetic IDs never enter the serialized output."""
from pathlib import Path
from types import SimpleNamespace as NS
import json, os, tempfile
import research_hook

def setup():
    class TestTree:
        def __init__(self):
            self.mb=NS(pos=0)
            self.log=lambda *_:None
        def _gate(self,l,t): return True
        def request_end(self): pass
        def cycle(self,eng,anchor,samp,rng,seen,hist,dry,orig):
            self._lk=([11003,11004,11005,11006,11007,11008,11009],9) if self.mb.pos==0 else None
            outputs=[11003,11004] if self.mb.pos==0 else [11005,12008]
            self.mb.pos+=len(outputs); eng.tm["draft"]+=.001
            return outputs, [anchor,*outputs[:-1]]
    return NS(Tree=TestTree,LK_E={"sampled":{6:5,16:7}})
with tempfile.TemporaryDirectory() as d:
    os.environ["FORGE_GATE_COUNTS"]=d
    module=setup();research_hook.install(module,"observe")
    t=module.Tree(); e=NS(tm={"draft":0})
    t.cycle(e,11999,[1],None,None,None,None,None)
    t.cycle(e,11004,[1],None,None,None,None,None)
    t.request_end()
    s=(Path(d)/'request_001.json').read_text()
    assert all(str(i) not in s for i in [11003,11004,11005,12008,11999])
    rows=json.loads(s)['rounds'];assert rows[0][2]==3 and rows[1][2]==0
m=setup();research_hook.install(m,"feedback_1")
t=m.Tree();assert not t._gate(8,1.) and t._gate(16,1.)
t._yield_ema=.06;t._cyc_ema=110;t._draft_ema=10
assert not t._gate(8,1.)
print("PASS: contiguous copy, miss, empty lookup, no stored IDs, causal gate cold start")
# A rejected hypothetical copy updates only after its future is visible to the live sampler.
import sys
dummy=NS(lookup=lambda *a,**kw:([991,992,993,994,995,996,997],12))
sys.modules["ngram_draft"]=dummy
class Shadow:
    def __init__(self):
        self.log=lambda *_:None; self.mb=NS(pos=0); self._yield_ema=.09
        self._cyc_ema=110; self._draft_ema=10
    def _gate(self,L,temp,*pos):return True
    def request_end(self):pass
    def cycle(self,eng,anchor,*args):
        p=self.mb.pos;self.mb.pos+=1
        if p==0:
            hit=dummy.lookup(None)
            assert not self._gate(hit[1],1.,p)
        return [[991,992,1199][p]],[anchor]
module=NS(Tree=Shadow,LK_E={"sampled":{6:5,16:7}})
research_hook.install(module,"shadow_3")
t=module.Tree();eng=NS(ngram=NS(cycles=0,accepted=0))
for i,last in enumerate([1100,991,992]):
    t.cycle(eng,last,[1],None,None,None,None,None)
    assert t._review_errors["short"]==([] if i<2 else [-2])
t.request_end()
assert not t._review_errors["short"] and not t._shadow_wait
print("PASS: shadow feedback matures after the mismatch, resets between requests")
