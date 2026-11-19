import json, csv, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from PIL import Image
import sys, os
R=(sys.argv[1] if len(sys.argv)>1 else "results/track_validation").rstrip("/")+"/"
os.makedirs("fig",exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Liberation Serif","DejaVu Serif"],"font.size":7,"axes.linewidth":0.6})
BL="#1b6ca8"; OR="#d95f02"; GR="#888888"; LB="#9ecae1"; W=3.4
clips=["video1","video2","video3","video4","video5","video6","video7","white1","blacktip"]
lab={"white1":"white1","blacktip":"blacktip"}
off={c:json.load(open(R+f"offline_gate/{c}_summary.json")) for c in clips}
pad={c:json.load(open(R+f"offline_gate_padded/{c}_summary.json")) for c in clips}
# ---------- table data
rows=[]
for c in clips:
    b=off[c]["baseline"]; m=off[c]["median_clip"]; o=off[c]["oracle"]; p=pad[c]["median_clip"]
    rows.append(dict(clip=c,extras=b["extra_tracks_retained"],extras_removed=b["extra_tracks_retained"]-m["extra_tracks_retained"],
        sharks=b["shark_tracks_retained"],sharks_deleted=b["shark_tracks_retained"]-m["shark_tracks_retained"],
        ub=b["union_dice"],um=m["union_dice"],uo=o["union_dice"],up=p["union_dice"],
        db=b["authors_dice"],dm=m["authors_dice"],dp=p["authors_dice"],rb=b["recall"],rm=m["recall"],
        extras_removed_pad=b["extra_tracks_retained"]-p["extra_tracks_retained"],sharks_deleted_pad=b["shark_tracks_retained"]-p["shark_tracks_retained"]))
e2e=[]
for e in ("blacktip","video2","video5","video6"):
    j=json.load(open(R+f"e2e_{e}/validation/evaluation.json")); M=j["methods"]; b=M["baseline (all tracks)"]; m=M["median_clip"]
    e2e.append(dict(clip=e,sharks=b["shark_tracks_retained"],extras=b["extra_tracks_retained"],db=b["authors_dice"],ub=b["union_dice"],rb=b["recall"],
        dm=m["authors_dice"],um=m["union_dice"],rm=m["recall"],extras_kept=m["extra_tracks_retained"],sharks_kept=m["shark_tracks_retained"]))
json.dump(dict(offline=rows,e2e=e2e),open("val_tables.json","w"),indent=1)
for r in rows: print(r["clip"],r["extras_removed"],"/",r["extras"],"sharks deleted",r["sharks_deleted"],round(r["ub"],3),round(r["um"],3),round(r["up"],3),"pad extras removed",r["extras_removed_pad"],"pad sharks deleted",r["sharks_deleted_pad"])
for r in e2e: print(r)

# ---------- Fig: union Dice before / after
fig,axs=plt.subplots(2,1,figsize=(W,3.7),gridspec_kw={"hspace":.75})
ax=axs[0]; x=np.arange(len(clips)); bw=.27
ax.bar(x-bw,[r["ub"] for r in rows],bw,color=GR,label="all tracks (FLAIR, unpruned)")
ax.bar(x,[r["um"] for r in rows],bw,color=BL,label="median CLIP rule")
ax.bar(x+bw,[r["uo"] for r in rows],bw,color="white",edgecolor="black",lw=.6,label="oracle (uses labels)")
ax.set_xticks(x); ax.set_xticklabels(["v1","v2","v3","v4","v5","v6","v7","white1","blacktip"],fontsize=6); ax.set_ylim(0,1.38); ax.set_ylabel("union Dice")
ax.set_title("(a) Offline, released masks (9 clips)",fontsize=7,loc="left"); ax.scatter([],[],marker="x",color=OR,s=22,lw=1.4,label="real shark deleted"); ax.legend(fontsize=5.6,frameon=False,ncol=2,loc="upper center",columnspacing=.8,handlelength=1.2)
for i,r in enumerate(rows):
    if r["sharks_deleted"]>0: ax.scatter([i],[max(r["ub"],r["um"],r["uo"])+.08],marker="x",color=OR,s=22,lw=1.4,zorder=4)
ax=axs[1]; x=np.arange(4); 
ax.bar(x-.2,[r["ub"] for r in e2e],.4,color=GR); ax.bar(x+.2,[r["um"] for r in e2e],.4,color=BL)
ax.set_xticks(x); ax.set_xticklabels(["blacktip","video 2","video 5","video 6"],fontsize=6.3); ax.set_ylim(0,1.05); ax.set_ylabel("union Dice")
for i,r in enumerate(e2e):
    ax.text(i-.2,r["ub"]+.02,f'{r["ub"]:.2f}',ha="center",fontsize=5.8); ax.text(i+.2,r["um"]+.02,f'{r["um"]:.2f}',ha="center",fontsize=5.8,color=OR if r["um"]==0 else "black")
ax.set_title("(b) End-to-end, notebook 04 (4 clips)",fontsize=7,loc="left")
for a in axs:
    for sp in ("top","right"): a.spines[sp].set_visible(False)
fig.savefig("fig/fig_val_union.png",dpi=300,bbox_inches="tight",pad_inches=0.03); plt.close()

# ---------- Fig: strip plot of offline track scores
fig,ax=plt.subplots(figsize=(W,2.5)); rng=np.random.RandomState(3)
S=[];E=[]
for c in clips:
    for r in csv.DictReader(open(R+f"offline_gate/{c}_tracks.csv")):
        (S if r["shark_associated"]=="True" else E).append((c,float(r["median"])))
for k,(grp,col,name) in enumerate(((E,OR,"extra tracks"),(S,BL,"shark tracks"))):
    ys=[v for _,v in grp]; xs=k+rng.uniform(-.18,.18,len(ys))
    ax.scatter(xs,ys,s=14,color=col,alpha=.8,edgecolor="white",lw=.3,zorder=3)
ax.axhline(.95,color="black",lw=.8,ls="--"); ax.text(-0.48,.953,"threshold 0.95",fontsize=6,va="bottom",ha="left")
ax.set_xticks([0,1]); ax.set_xticklabels([f"extra tracks (n={len(E)})",f"shark tracks (n={len(S)})"],fontsize=6.5)
ax.set_xlim(-.5,1.5); ax.set_ylim(.4,1.02); ax.set_ylabel("median CLIP P(shark) per track")
for c,v in S:
    if v<.95: ax.annotate({"video3":"v3","video5":"v5","white1":"white1"}.get(c,c),(1,v),xytext=(1.2,v),fontsize=5.8,arrowprops=dict(arrowstyle="-",lw=.4,color=GR),va="center")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.savefig("fig/fig_val_scores.png",dpi=300,bbox_inches="tight",pad_inches=0.03); plt.close()
print("shark below .95:",[(c,round(v,3)) for c,v in S if v<.95]); print("extras >=.95:",[(c,round(v,3)) for c,v in E if v>=.95]); print(len(S),len(E))

# ---------- Fig: crop montage (from repo diagnostics): video 5 and video 3, first three columns
d=R+"offline_gate/diagnostics/"
a=Image.open(d+"video5_crops.png").convert("RGB"); b=Image.open(d+"video3_crops.png").convert("RGB")
a=a.crop((0,30,990,a.height)); b=b.crop((0,30,990,b.height))
m=Image.new("RGB",(990,a.height+b.height),"white"); m.paste(a,(0,0)); m.paste(b,(0,a.height)); m.save("fig/fig_val_crops.png")
print(m.size)
