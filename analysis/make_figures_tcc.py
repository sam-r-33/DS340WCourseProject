import json, csv, math, os, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D=(sys.argv[1] if len(sys.argv)>1 else "results").rstrip("/")+"/"   # run from the repo root: python3 analysis/make_figures_tcc.py results
os.makedirs("fig",exist_ok=True)
S=json.load(open(D+"gate_replay/gate_replay_summary.json"))
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Liberation Serif","DejaVu Serif"],"font.size":7,"axes.linewidth":0.6})
BL="#1b6ca8"; OR="#d95f02"; GR="#555555"; LB="#9ecae1"
W=3.4

# ---- Fig: proposed pipeline with both modular changes
fig,ax=plt.subplots(figsize=(W,3.9)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
steps=[("SAM 2 automatic mask generation","every 30th frame; size filter",False),
("CLIP zero-shot filter","p(target) > 0.95 on single boxes",False),
("Change 2: TCC gate","skip candidates with no accepted\nneighbour on an adjacent sampled frame",True),
("SAM 2 propagation, ±3 s windows","only admitted boxes are prompted",False),
("Confirmation (IoU > 0.7)","unchanged from FLAIR",False),
("Change 1: track-level CLIP validator","replaces manual pruning (KEEP_KEYS)\nusing CLIP evidence over the whole track",True)]
n=len(steps); hs=[1.0,1.0,1.55,1.0,1.0,1.55]; gap=(10-sum(hs))/(n-1); y=10
for i,(a_,b_,hl) in enumerate(steps):
    h=hs[i]; y-=h
    ax.add_patch(FancyBboxPatch((0.4,y),9.2,h,boxstyle="round,pad=0.02,rounding_size=0.15",fc="#fdebd9" if hl else "#f2f2f2",ec=OR if hl else GR,lw=1.3 if hl else 0.7))
    ax.text(5,y+h*(0.72 if not hl else 0.78),a_,ha="center",va="center",fontsize=7.2,fontweight="bold",color=OR if hl else "black")
    ax.text(5,y+h*(0.28 if not hl else 0.33),b_,ha="center",va="center",fontsize=6.3,color="#222")
    if i<n-1:
        ax.annotate("",xy=(5,y-gap+0.02),xytext=(5,y-0.02),arrowprops=dict(arrowstyle="-|>",color=GR,lw=0.8))
        y-=gap
fig.savefig("fig/fig6_pipeline_tcc.png",dpi=300,bbox_inches="tight",pad_inches=0.03); plt.close()

# ---- Fig 7: candidate chain map
A=S["accepted_candidates"]; lam=1.0
def nd(a,b): return math.hypot(a["cx"]-b["cx"],a["cy"]-b["cy"])/((a["diag"]+b["diag"])/2)
for a in A:
    ds=[nd(a,b) for b in A if abs(b["f"]-a["f"])==1]
    a["adm"]=bool(ds) and min(ds)<=lam
fig,ax=plt.subplots(figsize=(W,2.7))
for a in A:
    if a["adm"]:
        for b in A:
            if b["adm"] and b["f"]==a["f"]+1 and nd(a,b)<=lam: ax.plot([a["cx"],b["cx"]],[a["cy"],b["cy"]],"-",color=BL,lw=1,alpha=.7)
for a in A:
    if a["adm"]: ax.scatter(a["cx"],a["cy"],s=26,color=BL,zorder=3)
    else: ax.scatter(a["cx"],a["cy"],s=34,marker="X",color=OR,zorder=3)
    ax.annotate(str(a["f"]),(a["cx"],a["cy"]),textcoords="offset points",xytext=(4,3),fontsize=6) if a["f"] not in (3,4,5,6) else None
ax.annotate("sampled frames 2-7\n(shark chain)",(800,330),xytext=(1050,330),fontsize=6,arrowprops=dict(arrowstyle="-",lw=.5,color=GR),va="center")
ax.invert_yaxis(); ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
ax.scatter([],[],s=26,color=BL,label="admitted (8)"); ax.scatter([],[],s=34,marker="X",color=OR,label="rejected as isolated (5)")
ax.legend(fontsize=6,frameon=False,loc="best")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
print(sum(a["adm"] for a in A),len(A))
fig.savefig("fig/fig7_chain_map.png",dpi=300,bbox_inches="tight",pad_inches=0.03); plt.close()

# ---- Fig 8: variant comparison
V=S["variants"]
fig,ax=plt.subplots(figsize=(W,2.7))
names=[f'{v["id"]}' for v in V][::-1]
lab={"A":"A  FLAIR baseline","B1":"B1  CLIP p>0.97","B2":"B2  CLIP p>0.99","B3":"B3  CLIP p>0.995","C1":"C1  margin≥4","C2":"C2  margin≥5","D1":"D1  TCC λ=1 (ours)","D2":"D2  TCC + last frame","D3":"D3  TCC + margin≥4"}
for k,v in enumerate(V[::-1]):
    left=0
    for key,col in (("new_call_passes",BL),("reprop_passes",LB)):
        val=v[key]
        ax.barh(k,val,left=left,color=col if v["id"]!="D1" or key!="new_call_passes" else OR,height=.62,edgecolor="white",lw=.4); left+=val
    ax.text(left+12,k,f'{v["total_passes"]}  ({-v["passes_saved_pct"]:.0f}%)' if v["passes_saved_pct"] else str(v["total_passes"]),va="center",fontsize=6)
ax.set_yticks(range(len(V))); ax.set_yticklabels([lab[v["id"]] for v in V[::-1]],fontsize=6.3)
ax.set_xlim(0,1500); ax.set_xlabel("SAM 2 propagation frame-passes (replay)")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BL,label="new-track calls"),Patch(color=OR,label="new-track calls (ours)"),Patch(color=LB,label="re-propagation")],fontsize=6,frameon=False,loc="lower right")
fig.savefig("fig/fig8_variants.png",dpi=300,bbox_inches="tight",pad_inches=0.03); plt.close()

# ---- Fig 9: sensitivity sweeps
OW=S["operating_windows"]
fig,axs=plt.subplots(3,1,figsize=(W,4.3),gridspec_kw={"hspace":.85})
for ax,key,title,xl,win in ((axs[0],"sweep_lambda","TCC gate","λ (box diagonals)",OW["lambda"]),
                            (axs[1],"sweep_tau","CLIP threshold","p(target) threshold",OW["clip_threshold"]),
                            (axs[2],"sweep_margin","Logit margin","margin threshold",OW["logit_margin"])):
    sw=S[key]; x=[r["value"] for r in sw]
    ax.axvspan(win[0],win[1],color="#e3eef7",lw=0)
    ax.plot(x,[r["passes_saved_pct"] for r in sw],"-o",color=BL,ms=2.5,lw=1)
    bad=[r for r in sw if (not r["shark_confirmed"]) or r["tracks_off_shark"]>0 and False]
    miss=[r for r in sw if not r["shark_confirmed"]]
    if miss: ax.plot([r["value"] for r in miss],[r["passes_saved_pct"] for r in miss],"X",color=OR,ms=5,ls="none",label="shark track lost")
    ax.set_ylim(-5,105); ax.set_ylabel("passes saved (%)"); ax.set_xlabel(xl); ax.set_title(title,fontsize=7,loc="left")
    if miss: ax.legend(fontsize=5.8,frameon=False,loc="center left")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.savefig("fig/fig9_sweeps.png",dpi=300,bbox_inches="tight",pad_inches=0.03); plt.close()

# ---- Fig 11 (new): which propagation calls disappear
cc={int(k):v for k,v in S["call_cost_by_sampled_frame"].items()}
base_calls=set(V[0]["calls"]); tcc=[v for v in V if v["id"]=="D1"][0]; tcc_calls=set(tcc["calls"])
fig,ax=plt.subplots(figsize=(W,2.1))
for f in range(8):
    if f in tcc_calls: ax.bar(f,cc[f],color=BL,width=.65)
    elif f in base_calls: ax.bar(f,cc[f],color="white",edgecolor=OR,hatch="////",lw=.8,width=.65)
    if f in base_calls: ax.text(f,cc[f]+5,str(cc[f]),ha="center",fontsize=6)
ax.set_xticks(range(8)); ax.set_xlabel("sampled frame that launches the call"); ax.set_ylabel("frame-passes"); ax.set_ylim(0,225)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BL,label="call still made with TCC"),Patch(facecolor="white",edgecolor=OR,hatch="////",label="call removed by TCC")],fontsize=6,frameon=False,loc="upper right",ncol=1)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.savefig("fig/fig10_calls.png",dpi=300,bbox_inches="tight",pad_inches=0.03); plt.close()

# ---- ground-truth Dice vs union Dice on released masks (read from the repo's offline gate summaries)
import numpy as np, os
V=os.environ.get("VAL",D+"track_validation")   # default: results/track_validation
s2=json.load(open(V+"/offline_gate/video2_summary.json")); s5=json.load(open(V+"/offline_gate/video5_summary.json"))
labs=["Video 2\nFLAIR (unpruned)","Video 2\nshark tracks only (oracle)","Video 5\nFLAIR (unpruned)"]
dice=[s2["baseline"]["authors_dice"],s2["oracle"]["authors_dice"],s5["baseline"]["authors_dice"]]
uni=[s2["baseline"]["union_dice"],s2["oracle"]["union_dice"],s5["baseline"]["union_dice"]]
fig,ax=plt.subplots(figsize=(W,2.2)); x=np.arange(3); bw=.36
b1=ax.bar(x-bw/2,dice,bw,color=LB,label="authors’ Dice (best track per shark)")
b2=ax.bar(x+bw/2,uni,bw,color=BL,label="union Dice (all tracks merged)")
for b,v in list(zip(b1,dice))+list(zip(b2,uni)): ax.text(b.get_x()+b.get_width()/2,v+.02,f"{v:.3f}",ha="center",fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=6.3); ax.set_ylim(0,1.18); ax.set_ylabel("Dice")
ax.legend(fontsize=5.8,frameon=False,loc="upper center",ncol=1,bbox_to_anchor=(.5,1.12))
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.savefig("fig/fig_gt_dice.png",dpi=300,bbox_inches="tight",pad_inches=0.03); plt.close()
