"""Prepare a document-held-out pilot from WikiText-2 TRAIN only."""
import hashlib, json, re, urllib.request
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
URL="https://raw.githubusercontent.com/pytorch/examples/d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/word_language_model/data/wikitext-2/train.txt"
def main():
    d=ROOT/"data";d.mkdir(exist_ok=True)
    p=d/"wikitext2_train.txt"
    if not p.exists():
        with urllib.request.urlopen(URL,timeout=60) as r: p.write_bytes(r.read())
    text=p.read_text()
    parts=re.split(r"(?m)^ = ([^=\n]+) = \s*$",text)
    docs=[parts[i]+"\n"+parts[i+1] for i in range(1,len(parts)-1,2)]
    tok=Tokenizer.from_file(str(d/"tokenizer.json"))
    groups={"train":[],"monitor":[],"confirmation":[]};seen=set();meta=[]
    for document in docs:
        digest=hashlib.sha256(document.encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        ids=tok.encode(document).ids
        if len(ids)<513:continue
        bucket=int(digest[:16],16)%100
        split="train" if bucket<80 else "monitor" if bucket<90 else "confirmation"
        groups[split].append(np.asarray(ids,dtype=np.uint16))
        meta.append({"sha256":digest,"split":split,"tokens":len(ids)})
    output={}
    for split,values in groups.items():
        offsets=np.asarray([0]+list(np.cumsum([len(x) for x in values])),dtype=np.int64)
        dest=d/(split+".npz")
        np.savez_compressed(dest,tokens=np.concatenate(values),offsets=offsets)
        output[split]={"documents":len(values),"tokens":int(offsets[-1]),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}
    manifest={"source_url":URL,"source_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
      "tokenizer_source":"rohanarun/Super-1@492a87a5e263e806b0a2581c57259e81f1a000e6/auto-research/results/gru-extended-20261016-v1/data_metadata/tokenizer.json",
      "tokenizer_sha256":hashlib.sha256((d/"tokenizer.json").read_bytes()).hexdigest(),
      "split":"Exact-document dedup; SHA256(document) mod100: 0..79 train,80..89 monitor,90..99 confirmation; minimum513 tokens; all derived solely from WikiText2 train",
      "official_validation_reads":0,"official_test_reads":0,"splits":output,"documents":meta}
    (d/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(output),flush=True)
if __name__=="__main__":main()
