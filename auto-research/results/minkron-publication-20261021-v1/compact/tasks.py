"""Balanced four-register task. Protected key is never provided before query.

Writes=8*key+value; queries=32+key; distractors=36..43. Initialization writes
ALL keys in independently shuffled order. Overwrite cases present conflicting
old values, then replace ALL four keys in a new shuffled order. This avoids
revealing the future query key through a specially positioned target write.
Nominal age L gives true final-query age L-3..L due to that shuffled order.
No task mask or simulator state enters the model. Task labels are endpoint only.
"""
import numpy as np
MODES=('quiet','repeat','competing','overwrite')

def episode(seed,batch=32,age=32,mode='quiet'):
    assert age>=4 and mode in MODES
    r=np.random.default_rng(seed);rows=np.arange(batch)
    vals=r.integers(0,8,(batch,4));protected=r.integers(0,4,batch)
    xs=[];last=np.full((batch,4),-1);old=None
    def emit(tok):
        tok=np.asarray(tok,dtype=np.int64);wr=tok<32;rr=rows[wr]
        vals[rr,tok[wr]//8]=tok[wr]%8;last[rr,tok[wr]//8]=len(xs);xs.append(tok.copy())
    def all_writes(v):
        order=np.argsort(r.random((batch,4)),axis=-1)
        for i in range(4):
            k=order[:,i];emit(8*k+v[rows,k])
    all_writes(vals.copy())
    if mode=='overwrite':
        for _ in range(28):
            tok=36+r.integers(0,8,batch);mask=r.random(batch)<.5;tok[mask]=r.integers(0,32,mask.sum());emit(tok)
        old=vals[rows,protected].copy()
        new=(vals+r.integers(1,8,(batch,4)))%8;all_writes(new)
    target=vals[rows,protected].copy();repeat=36+r.integers(0,8,batch)
    for _ in range(age-4):
        tok=36+r.integers(0,8,batch)
        if mode=='repeat':tok=repeat.copy()
        if mode in ('competing','overwrite'):
            mask=r.random(batch)<.8;k=(protected+r.integers(1,4,batch))%4
            tok[mask]=8*k[mask]+r.integers(0,8,mask.sum())
        emit(tok)
    emit(32+protected)
    actual=len(xs)-1-last[rows,protected]
    assert np.all(vals[rows,protected]==target)
    assert np.all((actual>=age-3)&(actual<=age))
    return np.stack(xs,1),target,{'mode':mode,'nominal_age':age,'actual_age':actual,'query_key':protected,'stale':old}

def verify_oracle(x,y):
    for b in range(len(x)):
        values=[None]*4
        for token in x[b]:
            if token<32:values[token//8]=token%8
        assert y[b]==values[x[b,-1]-32]
