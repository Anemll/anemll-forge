"""Count-only on-device observation, or request-local feedback. Activated only on the named bench arm."""
import json
import os
from pathlib import Path
import time

def install(module, arm):
    cls=module.Tree
    cycle, finish, gate, init=cls.cycle, cls.request_end, cls._gate, cls.__init__
    def initialize(self,*args,**kwargs):
        init(self,*args,**kwargs)
        self.log("review gate: "+arm+"; scalar diagnostics")
        self._review_num=0
    cls.__init__=initialize
    if arm=="observe":
        def observe(self,eng,anchor,samp,rng,seen,hist,dry,orig):
            if not hasattr(self,"_review_rows"):
                self._review_rows=[]; self._review_pending=[]; self._review_prev=None
            pos=int(self.mb.pos)
            if self._review_prev!=(pos,int(anchor)):
                self._review_pending.clear()
            start=time.perf_counter(); draft_before=eng.tm.get("draft",0.)
            out=cycle(self,eng,anchor,samp,rng,seen,hist,dry,orig)
            wall=1000*(time.perf_counter()-start); draft=1000*(eng.tm.get("draft",0.)-draft_before)
            emitted=out[0]
            hit=getattr(self,"_lk",None)
            length=int(hit[1]) if hit is not None else 0
            # [match length, tree yield, verified copy prefix or null, drafter ms, wall ms, start position]
            record=[length,len(emitted),None if length else 0,round(draft,3),round(wall,3),pos]
            self._review_rows.append(record)
            if length:
                self._review_pending.append([record, [int(v) for v in hit[0] if v>=0],0])
            for actual in emitted:
                waiting=[]
                for row,proposal,n in self._review_pending:
                    if int(actual)!=proposal[n]:
                        row[2]=n
                    elif n+1==len(proposal):
                        row[2]=n+1
                    else:
                        waiting.append([row,proposal,n+1])
                self._review_pending=waiting
            self._review_prev=(int(self.mb.pos),int(emitted[-1])) if emitted else None
            return out
        def end(self):
            try:
                rows=getattr(self,"_review_rows",[])
                self._review_num+=1
                target=Path(os.environ["FORGE_GATE_COUNTS"])/f"request_{self._review_num:03d}.json"
                target.write_text(json.dumps({"rounds":rows},separators=(",",":")))
            finally:
                self._review_rows=[];self._review_pending=[];self._review_prev=None
                finish(self)
        cls.cycle=observe; cls.request_end=end
    elif arm.startswith("feedback_") or arm in ("recent_floor","shadow_3"):
        if arm=="shadow_3":
            # Reuse the existing lookup call. The extra state is request-local and lives only in RAM.
            import ngram_draft as ng
            import threading
            local=threading.local()
            original_lookup=ng.lookup
            def observe_proposal(*args,**kw):
                result=original_lookup(*args,**kw)
                if getattr(local,"watch",False):
                    local.hit=result
                    local.watch=False
                return result
            ng.lookup=observe_proposal
        def add_error(self,key,error):
            row=self._review_errors[key]
            row.append(error);del row[:-3]
        def adjusted(self,L,temp,*position):
            if temp<=0: return gate(self,L,temp,*position)
            tab=module.LK_E["sampled"]
            prior=tab[max(b for b in tab if b<=max(L,6))]
            key="short" if L<16 else "long"
            if not hasattr(self,"_review_errors"): self._review_errors={"short":[],"long":[]}
            errors=self._review_errors[key]
            if arm=="recent_floor":
                take=gate(self,L,temp,*position) and not (len(errors)>=2 and sum(errors[-2:]) < -3 and L<32)
            elif getattr(self,"_yield_ema",None) is None or getattr(self,"_cyc_ema",None) is None:
                take=L>=16
            else:
                pred=max(1.,min(8.,prior+sum(errors)/(float(arm.split("_")[1])+len(errors))))
                take=pred/max(1.,self._cyc_ema-self._draft_ema)>1.25*self._yield_ema
            self._review_prediction=(key,prior) if take else None
            return take
        def adaptive(self,eng,anchor,samp,rng,seen,hist,dry,orig):
            self._review_prediction=None
            n_before=eng.ngram.cycles if eng.ngram else 0
            accepted_before=eng.ngram.accepted if eng.ngram else 0
            if arm=="shadow_3":
                if getattr(self,"_shadow_next",None)!=(int(self.mb.pos),int(anchor)):
                    self._shadow_wait=[]
                local.watch=True;local.hit=None
            try:
                out=cycle(self,eng,anchor,samp,rng,seen,hist,dry,orig)
            finally:
                if arm=="shadow_3":local.watch=False
            paid=self._review_prediction is not None and eng.ngram.cycles==n_before+1
            if arm=="shadow_3":
                hit=local.hit;local.hit=None
                if not paid and hit is not None and samp[0]>0:
                    proposal,L=hit; tab=module.LK_E["sampled"]
                    key="short" if L<16 else "long"
                    prior=tab[max(b for b in tab if b<=max(L,6))]
                    self._shadow_wait.append([proposal,key,prior,0])
                for token in out[0]:
                    keep=[]
                    for proposal,key,prior,k in self._shadow_wait:
                        if int(token)!=int(proposal[k]):add_error(self,key,1+k-prior)
                        elif k+1==len(proposal):add_error(self,key,2+k-prior)
                        else:keep.append([proposal,key,prior,k+1])
                    self._shadow_wait=keep
                self._shadow_next=(int(self.mb.pos),int(out[0][-1])) if out[0] else None
            if self._review_prediction is not None and eng.ngram.cycles==n_before+1:
                key,prior=self._review_prediction
                add_error(self,key,1+eng.ngram.accepted-accepted_before-prior)
            return out
        def end(self):
            self._review_errors={"short":[],"long":[]}
            self._shadow_next=None;self._shadow_wait=[]
            finish(self)
        cls._gate=adjusted; cls.cycle=adaptive; cls.request_end=end
    else: raise ValueError("unknown research arm")
