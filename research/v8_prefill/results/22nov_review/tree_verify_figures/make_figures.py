from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import rcParams

out=Path(sys.argv[1])
rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","svg.hashsalt":"tree-verify-m6","axes.unicode_minus":True})
ink="#101D38"; gray="#405367"; light="#E6ECED"; rule="#B6C4C7"; teal="#096453"; sea="#D8ECE6"; amber="#9A560F"; ochre="#F4D49D"; pale="#F2F6F5"; white="#FFFFFF"

def page(h):
    f=plt.figure(figsize=(12,h/100),facecolor=white)
    a=f.add_axes([0,0,1,1],xlim=(0,1200),ylim=(h,0));a.axis("off")
    return f,a

def t(a,x,y,s,size=13,color=ink,bold=False,align="left",v="center",**kw):
    return a.text(x,y,s,fontsize=size,color=color,fontweight="bold" if bold else "normal",ha=align,va=v,linespacing=1.38,**kw)

def line(a,x1,y1,x2,y2,c=rule,lw=1.1,**kw):
    a.plot([x1,x2],[y1,y2],color=c,lw=lw,solid_capstyle="round",**kw)

def rect(a,x,y,w,h,face,edge="none",lw=1):
    a.add_patch(Rectangle((x,y),w,h,facecolor=face,edgecolor=edge,lw=lw))
def dot(a,x,y,r,face,edge="none",lw=1):
    a.add_patch(Circle((x,y),r,facecolor=face,edgecolor=edge,lw=lw))
def heading(a,title,sub):
    t(a,42,43,title,27,bold=True)
    t(a,44,90,sub,13.5,gray)
    line(a,44,116,1156,116)
def save(f,name):
    vector = out / (name + ".svg")
    f.savefig(vector, format="svg", metadata={"Date": None})
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text().splitlines()) + "\n")
    f.savefig(out/(name+".png"),format="png",dpi=220,facecolor=white)
    plt.close(f)

# Row IDs correspond to a valid topological order. Path, matrix and compaction share this parent list.
f,a=page(590)
heading(a,"One call verifies a tree of candidate tokens","Tree8 fits the anchor and seven proposed tokens into the same eight rows as a chain.")
t(a,44,157,"Build the tree",18,bold=True)
t(a,454,157,"Mask each row",18,bold=True)
t(a,810,157,"Commit one path",18,bold=True)
line(a,424,144,424,465,light);line(a,782,144,782,465,light)
t(a,44,200,"Rank children by path probability.\nKeep the seven best expansions.",11.6,gray)
t(a,454,200,"Each row sees its own path\nand the prior KV history.",11.6,gray)
t(a,810,200,"The target picks one continuation.\nMove its rows into consecutive slots.",11.6,gray)
parent=[-1,0,0,1,2,4,1,3]
nodes=[(67,335),(157,283),(157,404),(245,259),(245,404),(335,404),(245,323),(335,259)]
path=[0,2,4,5]
for i in range(1,8):
    j=parent[i]; take=i in path and j in path
    line(a,*nodes[j],*nodes[i],teal if take else rule,3 if take else 1.8,zorder=1)
for i,(x,y) in enumerate(nodes):
    take=i in path
    dot(a,x,y,16,teal if take else white,teal if take else rule,1.5)
    t(a,x,y+.3,str(i),11,white if take else gray,True,align="center")
t(a,50,364,"anchor",9.6,gray)
t(a,44,451,"Numbers identify verifier rows.",10.6,gray)
cell=21; mx=553; my=269
t(a,637,237,"Visible block row",10.6,gray,align="center")
for j in range(8):
    t(a,mx+cell*(j+.5),my-11,str(j),9.4,gray,align="center")
for i in range(8):
    t(a,mx-14,my+cell*(i+.5),str(i),9.4,teal if i==5 else gray,i==5,align="center")
    seen=[]; j=i
    while j!=-1: seen.append(j);j=parent[j]
    for j in range(8):
        rect(a,mx+j*cell,my+i*cell,cell-2,cell-2,teal if j in seen and i==5 else (gray if j in seen else light))
t(a,487,350,"Query\nrow",10.6,gray,align="center")
t(a,454,451,"Row 5 sees rows 0, 2, 4 and 5.",10.6,teal,True)
sx=844; dx=68
t(a,810,259,"Accepted verifier rows",11.4,teal,True)
for i,row in enumerate(path):
    x=sx+i*dx
    if i: line(a,x-dx,309,x,309,teal,2)
for i,row in enumerate(path):
    x=sx+i*dx
    dot(a,x,309,19,teal);t(a,x,309,str(row),12,white,True,align="center")
    a.annotate("",xy=(x,370),xytext=(x,339),arrowprops={"arrowstyle":"-|>","color":gray,"lw":1.1,"mutation_scale":12})
    rect(a,x-24,376,48,36,sea,teal,.8)
    t(a,x,394,str(i),12,ink,True,align="center")
t(a,810,451,"New state slots for the next cycle.",10.6,gray)
rect(a,44,486,1112,80,pale)
t(a,62,507,"Chain-shaped draft",12,ink,True)
t(a,62,537,"When all seven nodes lie on one path, use the regular chain verifier.",11.4,gray)
t(a,691,507,"What gets compacted",12,ink,True)
t(a,691,537,"KV, DeltaNet tape, convolution rows and drafter taps.",10.55,gray)
save(f,"02-tree8-method")

f,a=page(595)
heading(a,"The benefit and cost of verifier width","The throughput comparison uses the full model. The width sweep times one four-layer verifier group.")
t(a,44,157,"Sampled decode speed",18,bold=True)
t(a,644,157,"Cost of widening the verifier",18,bold=True)
t(a,44,191,"Tree8 relative to the eight-row chain",12,gray)
t(a,644,191,"Milliseconds per layer group at 8K and 32K context",12,gray)
line(a,608,145,608,538,light)
cats=["Overall","8K context","32K context","Code","Tool agent"];gains=[8.5,9.0,11.6,10.9,-1.6]
def bx(x):return 245+21*x
for tick in (0,5,10):
    line(a,bx(tick),219,bx(tick),457,rule if tick==0 else light,1.1)
    t(a,bx(tick),481,f"{tick}%",10,gray,align="center")
for i,(c,v) in enumerate(zip(cats,gains)):
    y=246+i*47
    t(a,46,y,c,12,ink,c=="Overall")
    rect(a,bx(min(v,0)),y-9,21*abs(v),18,teal if v>0 else amber)
    t(a,bx(v)+(9 if v>0 else -8),y,("{:+.1f}%".format(v)).replace("-","−"),11.5,teal if v>0 else amber,True,align="left" if v>0 else "right")
# Width sweep measures actual group times (two timing blocks per width).
width=np.arange(8,17)
ms8=np.array([5.258,5.334,5.403,5.446,5.540,5.627,5.701,5.821,5.814])
ms32=np.array([6.022,6.168,6.267,6.409,6.515,6.694,6.750,7.089,7.096])
def wx(i):return 698+(i-8)*49
def wy(ms):return 463-(ms-5.0)*104
for val in (5.,5.5,6.,6.5,7.):
    line(a,693,wy(val),1100,wy(val),light)
    t(a,680,wy(val),f"{val:.1f}",10,gray,align="right")
for ms,c,label,marker in [(ms32,amber,"32K","s"),(ms8,teal,"8K","o")]:
    a.plot([wx(i) for i in width],[wy(v) for v in ms],color=c,lw=2.4,marker=marker,ms=5.4,mfc=white,mec=c,mew=1.55)
    t(a,1106,wy(ms[-1]),label,11.5,c,True)
for w in width:t(a,wx(w),483,str(w),10,gray,align="center")
t(a,896,509,"Verifier rows",10.8,gray,align="center")
t(a,44,531,"3.91 → 4.35 tokens per verify",13,teal,True)
t(a,44,559,"Cycle time rose 2.4% in this sampled block.",11.1,gray)
t(a,644,531,"At 32K, 14 → 15 rows added 0.34 ms per group.",12.2,amber,True)
t(a,644,559,"On greedy replay, every wider option lost throughput at 32K.",11,gray)
save(f,"03-tree8-speed-and-width")

f,a=page(590)
heading(a,"Sliced verification reads a shorter KV prefix","A fixed-width target entry can reuse the same cache buffers and skip keys beyond the selected prefix.")
t(a,44,157,"Choose the smallest eligible entry",17.5,bold=True)
t(a,640,157,"Decode throughput on the long session",17.5,bold=True)
line(a,607,145,607,475,light)
t(a,44,189,"Example with a 33K-token prefix",12,gray)
t(a,640,189,"Change versus tree8 with full-width KV entries",12,gray)
def kx(x):return 48+x*8.0
for y,label,max_k in [(252,"Original 64K entry",64),(348,"Sliced 40K entry",40)]:
    t(a,48,y-31,label,12,ink,True)
    rect(a,kx(0),y,8*max_k,24,ochre)
    rect(a,kx(0),y,8*33,24,teal)
    if max_k<64:
        rect(a,kx(max_k),y,8*(64-max_k),24,pale)
    line(a,kx(max_k),y-3,kx(max_k),y+26,gray,.9)
    for k in (0,max_k):t(a,kx(k),y+41,f"{k}K" if k else "0",9.7,gray,align="center")
for x,c,lab in [(48,teal,"Occupied KV"),(220,ochre,"Extra keys read"),(425,pale,"Skipped")]:
    rect(a,x,410,14,14,c,rule,.6);t(a,x+22,417,lab,10.2,gray)
t(a,48,458,"Prefix position + 8 verifier rows must fit the entry.",11.4,gray)
right=1125;left=797
def xx(v):return left+(v+0.65)*(right-left)/7.3
for v in (0,2,4,6):
    line(a,xx(v),240,xx(v),423,rule if v==0 else light,1.1)
    t(a,xx(v),448,f"{v}%",10,gray,align="center")
bands=["8–16K","16–32K","32–57K","All turns"]
point1=[1.24,2.72,5.69,3.40]; point2=[-.16,1.51,5.42,2.49]
for i,(b,v,w) in enumerate(zip(bands,point1,point2)):
    y=252+i*53
    t(a,640,y,b,12,ink,i==3)
    line(a,xx(w),y+6,xx(v),y-6,rule,1.3)
    a.plot(xx(v),y-6,marker="o",mfc=teal,mec=teal,ms=7)
    a.plot(xx(w),y+6,marker="D",mfc=white,mec=amber,ms=6.3,mew=1.6)
a.plot(655,216,marker="o",ms=7,mfc=teal,mec=teal)
t(a,672,216,"All sets",10.8,teal,True)
a.plot(807,216,marker="D",ms=6.3,mfc=white,mec=amber,mew=1.6)
t(a,824,216,"Excluding seed 7100",10.8,amber,True)
rect(a,44,490,1112,74,pale)
t(a,60,515,"+3.40% pooled   /   +2.49% excluding seed 7100",15,teal,True)
t(a,60,546,"One baseline launch ran slow on early turns.",11.2,gray)
t(a,742,515,"152/152 replies matched",15,ink,True)
t(a,742,546,"471 parity calls had zero maximum logit difference.",10.7,gray)
save(f,"04-sliced-kv")
