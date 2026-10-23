"""Recreate training data and a document-held-out confirmation set unused by the prior pilot."""
from pathlib import Path
import hashlib, json, re, urllib.request
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
URL='https://raw.githubusercontent.com/pytorch/examples/d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/word_language_model/data/wikitext-2/train.txt'

def main():
    d=ROOT/'data';d.mkdir(exist_ok=True);p=d/'wikitext2_train.txt'
    if not p.exists():
        with urllib.request.urlopen(URL,timeout=60) as r:p.write_bytes(r.read())
    parts=re.split(r'(?m)^ = ([^=\n]+) = \s*$',p.read_text())
    tok=Tokenizer.from_file(str(d/'tokenizer.json'))
    train=[];monitor=[];seen=set()
    for i in range(1,len(parts)-1,2):
        document=parts[i]+'\n'+parts[i+1];digest=hashlib.sha256(document.encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest);ids=tok.encode(document).ids
        if len(ids)<513:continue
        bucket=int(digest[:16],16)%100
        if bucket<90:
            (train if bucket<80 else monitor).append((digest,np.asarray(ids,dtype=np.uint16)))
    # Reconstruct only which monitor documents the earlier fixed evaluation touched.
    # Its entire previous confirmation split is excluded from this experiment.
    offsets=np.asarray([0]+list(np.cumsum([len(x[1]) for x in monitor])),dtype=np.int64)
    used=set()
    for index in range(8):
        rng=np.random.default_rng(np.random.SeedSequence([91823,index,8721]))
        for _ in range(4):
            doc=int(rng.integers(len(monitor)));lo,hi=offsets[doc:doc+2]
            rng.integers(lo,hi-512);used.add(doc)
    confirmation=[x for i,x in enumerate(monitor) if i not in used]
    records={};docs=[]
    for split,values in [('train',train),('confirmation',confirmation)]:
        offsets=np.asarray([0]+list(np.cumsum([len(x[1]) for x in values])),dtype=np.int64)
        path=d/(split+'.npz');np.savez_compressed(path,tokens=np.concatenate([x[1] for x in values]),offsets=offsets)
        records[split]={'documents':len(values),'tokens':int(offsets[-1]),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        docs.extend({'split':split,'sha256':h,'tokens':len(x)} for h,x in values)
    manifest={'source_url':URL,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
              'tokenizer_sha256':hashlib.sha256((d/'tokenizer.json').read_bytes()).hexdigest(),
              'split_description':'Training: original first16-SHA256 mod100 buckets0..79. Confirmation: original buckets80..89, excluding all27 documents previously evaluated by the fixed monitor window sampler. No previous confirmation documents are used.',
              'previously_evaluated_monitor_documents_excluded':[monitor[i][0] for i in sorted(used)],
              'splits':records,'documents':docs,'official_validation_reads':0,'official_test_reads':0}
    assert len(confirmation)==38 and len(train)==482
    assert not ({h for h,_ in train}&{h for h,_ in confirmation})
    (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
