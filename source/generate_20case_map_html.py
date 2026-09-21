#!/usr/bin/env python3
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Selected20Case_GeoreferencedTrajectories.json"
OUT = ROOT / "cases20" / "MAP_20_CASES.html"

cases = json.loads(DATA.read_text(encoding="utf-8"))
if len(cases) != 20:
    raise RuntimeError(f"Expected 20 cases, got {len(cases)}")

cases_json = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))

html = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>20 AIS cases for HINN figure selection - geographic map edition</title>
<style>
:root{--blue:#1f77b4;--black:#111;--green:#2ca02c;--orange:#ff7f0e;--purple:#9467bd;--red:#d62728}
*{box-sizing:border-box}
body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f4f5f7;color:#111}
header{background:#fff;padding:13px 18px;border-bottom:1px solid #d8d8d8;position:sticky;top:0;z-index:50;box-shadow:0 1px 5px rgba(0,0,0,.08)}
h1{margin:0 0 5px;font-size:21px}
.note{font-size:13px;line-height:1.45}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px}
select,button{font:inherit;padding:5px 8px;border:1px solid #aaa;background:#fff;border-radius:4px}
.status{font-size:12px;color:#444}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}
.card{background:#fff;border:1px solid #d6d6d6;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.title{padding:8px 10px;border-bottom:1px solid #ececec;font-size:14px;line-height:1.4}
.map{height:560px;position:relative;overflow:hidden;background:#dcecf3}
.tile{position:absolute;width:256px;height:256px;user-select:none;-webkit-user-drag:none;image-rendering:auto}
.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.local{height:560px;display:flex;align-items:center;justify-content:center;padding:4px}
.local img{max-width:100%;max-height:100%;object-fit:contain}
.legend{position:absolute;right:8px;top:8px;background:rgba(255,255,255,.94);padding:6px 8px;border:1px solid #999;font-size:10px;line-height:1.55;z-index:5;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.sw{display:inline-block;width:19px;height:4px;border-radius:3px;margin-right:5px;vertical-align:middle}
.attrib{position:absolute;right:3px;bottom:2px;background:rgba(255,255,255,.88);font-size:9px;padding:2px 4px;z-index:5}
.waiting{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#53616a;font-size:13px;background:linear-gradient(135deg,#eaf4f8,#d9ebf2)}
@media(max-width:1100px){.row{grid-template-columns:1fr}.map,.local{height:520px}}
</style>
</head>
<body>
<header>
  <h1>20 AIS cases for HINN figure selection - geographic map edition</h1>
  <div class="note">
    Cases 1-12: HINN-leading. Cases 13-20: near-tie/challenging.
    Trajectories and comparison plots are local. Internet is required only for the geographic basemap.
    The default map is Esri World Street Map, with Esri World Topographic fallback.
    Maps are rendered lazily so only visible cases request tiles.
  </div>
  <div class="toolbar">
    <label>Preferred basemap:
      <select id="providerSelect">
        <option value="esri" selected>Esri World Street Map - recommended</option>
        <option value="topo">Esri World Topographic Map</option>
        <option value="osm">OpenStreetMap Standard - optional fallback only</option>
      </select>
    </label>
    <button id="rerenderBtn" type="button">Reload visible maps</button>
    <span class="status">Automatic fallback: Esri Street -> Esri Topographic. No API key, Leaflet, or CDN JavaScript is required.</span>
  </div>
</header>
<div id="root"></div>

<script>
const CASES=__CASES__;
const C={history:"#1f77b4",truth:"#111111",rf:"#2ca02c",rnn:"#ff7f0e",lstm:"#9467bd",hinn:"#d62728"};

const PROVIDERS={
 esri:{
   label:"Esri World Street Map",
   url:(z,x,y)=>`https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/${z}/${y}/${x}`,
   attrib:"Map tiles © Esri and contributors"
 },
 topo:{
   label:"Esri World Topographic Map",
   url:(z,x,y)=>`https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/${z}/${y}/${x}`,
   attrib:"Map tiles © Esri and contributors"
 },
 osm:{
   label:"OpenStreetMap Standard",
   url:(z,x,y)=>`https://tile.openstreetmap.org/${z}/${x}/${y}.png`,
   attrib:"© OpenStreetMap contributors"
 }
};

function providerOrder(){
 const p=document.getElementById("providerSelect").value;
 if(p==="topo") return ["topo","esri"];
 if(p==="osm") return ["osm","esri","topo"];
 return ["esri","topo"];
}

function merc(lat,lon,z){
 const s=256*Math.pow(2,z),x=(lon+180)/360*s,si=Math.sin(lat*Math.PI/180),
 y=(.5-Math.log((1+si)/(1-si))/(4*Math.PI))*s;
 return [x,y];
}
function zoomFor(all,w,h){
 for(let z=18;z>=3;z--){
   const q=all.map(p=>merc(p[0],p[1],z)),xs=q.map(p=>p[0]),ys=q.map(p=>p[1]);
   if(Math.max(...xs)-Math.min(...xs)<w*.72 && Math.max(...ys)-Math.min(...ys)<h*.72) return z;
 }
 return 3;
}
function pathD(a,z,ox,oy){
 return a.map((p,i)=>{
   const q=merc(p[0],p[1],z);
   return (i?"L":"M")+(q[0]-ox).toFixed(1)+","+(q[1]-oy).toFixed(1);
 }).join(" ");
}
function addPath(svg,a,z,ox,oy,color,lw,opacity){
 const p=document.createElementNS("http://www.w3.org/2000/svg","path");
 p.setAttribute("d",pathD(a,z,ox,oy));
 p.setAttribute("fill","none");
 p.setAttribute("stroke",color);
 p.setAttribute("stroke-width",lw);
 p.setAttribute("stroke-linecap","round");
 p.setAttribute("stroke-linejoin","round");
 p.setAttribute("opacity",opacity);
 svg.appendChild(p);
}
function addAnchor(svg,latlon,z,ox,oy){
 const q=merc(latlon[0],latlon[1],z);
 const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
 c.setAttribute("cx",(q[0]-ox).toFixed(1)); c.setAttribute("cy",(q[1]-oy).toFixed(1));
 c.setAttribute("r","5.2"); c.setAttribute("fill","#000"); c.setAttribute("stroke","#fff"); c.setAttribute("stroke-width","1");
 svg.appendChild(c);
}

function renderMap(el,c,force=false){
 if(el.dataset.rendered==="1" && !force) return;
 el.dataset.rendered="1";
 el.innerHTML="";
 const waiting=document.createElement("div");
 waiting.className="waiting";
 waiting.textContent="Loading geographic basemap...";
 el.appendChild(waiting);

 const w=el.clientWidth,h=el.clientHeight;
 const all=[...c.history,...c.truth,...c.rf,...c.rnn,...c.lstm,...c.hinn];
 const z=zoomFor(all,w,h),q=all.map(p=>merc(p[0],p[1],z));
 const xs=q.map(p=>p[0]),ys=q.map(p=>p[1]);
 const cx=(Math.min(...xs)+Math.max(...xs))/2, cy=(Math.min(...ys)+Math.max(...ys))/2;
 const ox=cx-w/2, oy=cy-h/2, n=Math.pow(2,z);
 const order=providerOrder();
 let loaded=0,failed=0,total=0;

 function updateWaiting(){
   if(loaded>0 && waiting.parentNode) waiting.remove();
   if(loaded===0 && failed===total && total>0){
     waiting.textContent="Basemap could not be reached. Connect to Internet or switch provider.";
   }
 }

 for(let tx=Math.floor(ox/256)-1;tx<=Math.floor((ox+w)/256)+1;tx++){
   for(let ty=Math.floor(oy/256)-1;ty<=Math.floor((oy+h)/256)+1;ty++){
     if(ty<0||ty>=n) continue;
     total++;
     const wx=((tx%n)+n)%n;
     const img=document.createElement("img");
     img.className="tile"; img.loading="lazy"; img.decoding="async";
     img.style.left=(tx*256-ox)+"px"; img.style.top=(ty*256-oy)+"px";
     img.dataset.pi="0";
     const setSrc=()=>{
       const pi=Number(img.dataset.pi);
       img.src=PROVIDERS[order[pi]].url(z,wx,ty);
     };
     img.onload=()=>{loaded++;updateWaiting();};
     img.onerror=()=>{
       let pi=Number(img.dataset.pi)+1;
       if(pi<order.length){img.dataset.pi=String(pi);setSrc();}
       else{failed++;updateWaiting();}
     };
     setSrc();
     el.appendChild(img);
   }
 }

 const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
 svg.setAttribute("class","overlay"); svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
 addPath(svg,c.history,z,ox,oy,C.history,3.5,.96);
 addPath(svg,c.truth,z,ox,oy,C.truth,5.0,1);
 addPath(svg,c.rf,z,ox,oy,C.rf,2.7,.95);
 addPath(svg,c.rnn,z,ox,oy,C.rnn,2.7,.95);
 addPath(svg,c.lstm,z,ox,oy,C.lstm,2.7,.95);
 addPath(svg,c.hinn,z,ox,oy,C.hinn,4.4,1);
 addAnchor(svg,c.history[c.history.length-1],z,ox,oy);
 el.appendChild(svg);

 const leg=document.createElement("div"); leg.className="legend";
 leg.innerHTML=`<div><span class=sw style="background:${C.history}"></span>10-min history</div>
 <div><span class=sw style="background:${C.truth}"></span>Ground truth</div>
 <div><span class=sw style="background:${C.rf}"></span>Random Forest</div>
 <div><span class=sw style="background:${C.rnn}"></span>RNN</div>
 <div><span class=sw style="background:${C.lstm}"></span>LSTM</div>
 <div><span class=sw style="background:${C.hinn}"></span><b>HINN</b></div>
 <div>● Prediction anchor</div>`;
 el.appendChild(leg);

 const at=document.createElement("div"); at.className="attrib";
 const pref=document.getElementById("providerSelect").value;
 at.textContent=(PROVIDERS[pref]||PROVIDERS.esri).attrib;
 el.appendChild(at);
}

const root=document.getElementById("root");
CASES.forEach((c,i)=>{
 const row=document.createElement("div"); row.className="row";
 const tag=c.recommended?"RECOMMENDED":c.group;
 row.innerHTML=`<div class=card>
   <div class=title><b>Case ${String(c.case).padStart(2,"0")} - ${c.scenario}</b> - MMSI ${c.mmsi} - ${tag}<br>
   HINN ADE/FDE ${c.hade.toFixed(1)}/${c.hfde.toFixed(1)} m; gain vs best baseline ${c.gade.toFixed(1)}%/${c.gfde.toFixed(1)}%</div>
   <div id=m${i} class=map data-case="${i}"><div class=waiting>Map will load when visible...</div></div>
 </div>
 <div class=card>
   <div class=title>Static local comparison</div>
   <div class=local><img loading="lazy" src="${c.local_img}"></div>
 </div>`;
 root.appendChild(row);
});

const observer=new IntersectionObserver(entries=>{
 entries.forEach(e=>{
   if(e.isIntersecting){
     const idx=Number(e.target.dataset.case);
     renderMap(e.target,CASES[idx]);
   }
 });
},{rootMargin:"700px 0px",threshold:0.01});

document.querySelectorAll(".map").forEach(el=>observer.observe(el));

function rerenderVisible(){
 document.querySelectorAll(".map").forEach(el=>{
   const r=el.getBoundingClientRect();
   if(r.bottom>-700 && r.top<window.innerHeight+700){
     renderMap(el,CASES[Number(el.dataset.case)],true);
   }else{
     el.dataset.rendered="0";
     el.innerHTML='<div class="waiting">Map will load when visible...</div>';
   }
 });
}
document.getElementById("providerSelect").addEventListener("change",rerenderVisible);
document.getElementById("rerenderBtn").addEventListener("click",rerenderVisible);
</script>
</body>
</html>'''.replace("__CASES__", cases_json)

OUT.write_text(html, encoding="utf-8")
print(f"Wrote: {OUT}")
