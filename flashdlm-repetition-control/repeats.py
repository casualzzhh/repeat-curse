"""Operational repetition heuristics; not a semantic correctness oracle."""
import difflib
import re

KINDS=('token','phrase','sentence')

def requested(prompt):
    return bool(re.search(r'重复|复述|\brepeat\w*\b|\breproduce\b|逐字',prompt,re.I))

def spans(ids,tok,protect=False):
    if protect or not ids:return []
    ids=list(map(int,ids));n=len(ids);result=[]
    text=tok.decode(ids,skip_special_tokens=True)
    ends=[len(tok.decode(ids[:i],skip_special_tokens=True)) for i in range(n+1)]
    def append(kind,start,end,source,detail=''):
        if end>start:result.append(dict(kind=kind,start=start,end=end,source=source,detail=detail))
    for i in range(1,n):
        if ids[i]!=ids[i-1]:continue
        lo,hi=ends[i-1],ends[i+1];piece=text[lo:hi]
        # 100, finite repeated decimal digits and word-boundary BPE collisions are not loop labels.
        if not piece.strip() or re.fullmatch(r'[\d\s.,+\-/%]+',piece):continue
        before=text[ends[i]:ends[i+1]]
        following=text[ends[i+1]:ends[i+1]+1]
        if before.strip().isascii() and before.strip().isalpha() and following.isalpha():continue
        append(0,i,i+1,i-1,piece)
    for width in range(2,13):
        copies=3 if width<=3 else 2
        for end in range(width*copies,n+1):
            block=ids[end-width:end]
            if len(set(block))<2:continue
            if not all(ids[end-width*j:end-width*(j-1)]==block for j in range(2,copies+1)):continue
            phrase=tok.decode(block,skip_special_tokens=True)
            letters=re.sub(r'[\W\d_]+','',phrase)
            if len(letters)<2:continue
            append(1,end-width,end,end-width*2,phrase)
    sentences=[]
    for m in re.finditer(r'[^。！？!?\n]+[。！？!?\n]|[^.!?\n]+\.(?=\s|$)',text):
        raw=m.group();normalized=re.sub(r'[\W_]+','',raw).lower()
        if len(normalized)<8:continue
        start=next((i for i in range(n) if ends[i+1]>m.start()),n)
        end=next((i for i in range(n+1) if ends[i]>=m.end()),n)
        for old,oldstart in sentences[-8:]:
            equal=normalized==old
            similar=(len(normalized)>=20 and len(old)>=20 and
                difflib.SequenceMatcher(None,normalized,old).ratio()>=.9 and normalized[:10]==old[:10])
            if equal or similar:
                append(2,start,end,oldstart,'exact' if equal else 'template');break
        sentences.append((normalized,start))
    # One maximal later-copy span per type/start, avoiding every suffix of a long cycle.
    merged={}
    for s in result:
        key=(s['kind'],s['start'])
        if key not in merged or s['end']>merged[key]['end']:merged[key]=s
    return sorted(merged.values(),key=lambda s:(s['end'],s['kind'],s['start']))

def metrics(ids,tok,protect=False):
    events=spans(ids,tok,protect)
    return dict(raw_adjacent=sum(a==b for a,b in zip(ids,ids[1:])),opportunities=max(0,len(ids)-1),
        spans={name:sum(s['kind']==k for s in events) for k,name in enumerate(KINDS)},
        repeated_positions={name:len({p for s in events if s['kind']==k for p in range(s['start'],s['end'])}) for k,name in enumerate(KINDS)},
        events=events)
