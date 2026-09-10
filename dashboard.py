#!/usr/bin/env python3
"""桐洲岛淹没预警 Dashboard — stdlib HTTP 服务, 复用 flood_warning/typhoon

  python3 dashboard.py [--port 8787]

后台线程每小时自动拉取实况 (4次API限频约2分钟), 缓存 data/live_cache.json;
打开网页即时显示最近快照 + 近72h水位曲线 + 台风预备级 + 两次事件复盘 (零API)。
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

import flood_warning as FW
import typhoon as TYPH

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data" / "live_cache.json"
REFRESH_SEC = 3600

EVENTS = {  # 事件复盘窗口 (起点前48h含预警提前量)
    "2026-07-13": ("2026-07-10", "2026-07-16"),
    "2026-08-09": ("2026-08-07", "2026-08-13"),
}

_state = {"running": False, "fetched_at": None, "error": None, "lock": threading.Lock()}
_hist_cache = None


def _load_cache() -> dict | None:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return None
    return None


def _snap_to_json(snap: dict, ty: dict) -> dict:
    sig, r = snap["sig"], snap["assess"]
    return {
        "fetched_at": snap["time"],
        "level": r["level"], "reasons": r["reasons"],
        "est_mom": None if np.isnan(r["est_mom"]) else round(r["est_mom"], 2),
        "est_dry": None if np.isnan(r["est_dry"]) else round(r["est_dry"], 2),
        "stations": {"baxia": _r(sig.baxia), "baxia24": _r(sig.baxia24),
                     "luzhu": _r(sig.lz), "lz_r6": _r(sig.lz_r6),
                     "lzz_r6": _r(sig.lzz_r6), "zk": _r(sig.zk)},
        "series": snap["series"], "typhoon": ty,
    }


def _r(v) -> float | None:
    return None if v is None or np.isnan(v) else round(float(v), 2)


def _fetch_once():
    """一次完整实况拉取 (水位4次API+台风) → 缓存文件."""
    with _state["lock"]:
        if _state["running"]:
            return
        _state["running"] = True
    try:
        ty = {"typhoons": [], "prealert": False}
        try:
            ty = TYPH.live_threat()
        except Exception as e:
            ty = {"typhoons": [], "prealert": False, "error": repr(e)[:80]}
        out = _snap_to_json(FW.live_snapshot(), ty)
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(out, ensure_ascii=False))
        _state["fetched_at"], _state["error"] = out["fetched_at"], None
    except Exception as e:
        _state["error"] = repr(e)[:120]
    finally:
        _state["running"] = False


def _bg_loop():
    while True:
        c = _load_cache()
        if c is None or _stale(c):
            _fetch_once()
        time.sleep(max(60, REFRESH_SEC - int(time.time() % REFRESH_SEC)))


def _stale(cache: dict) -> bool:
    try:
        return pd.Timestamp.now() - pd.Timestamp(cache["fetched_at"]) > pd.Timedelta(seconds=REFRESH_SEC)
    except Exception:
        return True


def history_range(start: str, end: str) -> dict:
    """归档区间 → 曲线 + 逐时等级 (零API, 全样本内存缓存)."""
    global _hist_cache
    if _hist_cache is None:
        h = FW.load_archive_hours()
        lv = FW.signals(h).apply(FW.assess, axis=1)
        _hist_cache = (h, [d["level"] for d in lv])
    h, levels = _hist_cache
    m = (h.index >= f"{start} 00:00") & (h.index <= f"{end} 23:00")
    hh = h[m]
    def col(c):
        return [[t.strftime("%Y-%m-%dT%H:%M"), None if np.isnan(v) else round(v, 3)]
                for t, v in hh[c].items()]
    lv = [levels[i] for i in np.flatnonzero(m)]
    ts = [t.strftime("%Y-%m-%dT%H:%M") for t in hh.index]
    return {"ts": ts, "series": {"baxia": col("baxia"), "luzhu": col("luzhu"),
                                 "lzz": col("lzz"), "zk": col("zk"), "xt": col("xt")},
            "levels": lv}


PAGE = """<!doctype html>
<html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>桐洲岛淹没预警</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root{--bg:#0d1420;--panel:#141d2e;--line:#233250;--txt:#dce6f5;--dim:#7d8fa8;
  --lvG:#2fa46a;--lvY:#e0a63a;--lvR:#e05252;--lvB:#4a8fd0;--acc:#5aa2ff}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--txt);
  font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;padding:24px}
.wrap{max-width:1080px;margin:0 auto;display:grid;gap:16px}
header{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:8px}
h1{font-size:20px;letter-spacing:.5px}
h1 small{color:var(--dim);font-weight:400;margin-left:10px;font-size:13px}
#meta{color:var(--dim);font-size:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
.badge{display:inline-block;font-size:34px;font-weight:700;padding:8px 26px;border-radius:10px;color:#fff}
.b-绿{background:var(--lvG)}.b-黄{background:var(--lvY)}.b-红{background:var(--lvR)}
.lvname{font-size:15px;color:var(--dim);margin-right:14px}
#reasons{margin-top:12px;display:grid;gap:4px;font-size:14px;color:var(--txt)}
#stations{margin-top:14px;display:flex;flex-wrap:wrap;gap:10px}
.st{background:#0d1420;border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:13px}
.st b{display:block;font-size:18px;margin-top:2px}
.st span{color:var(--dim)}
#tyline{margin-top:12px;padding:8px 12px;border-radius:8px;font-size:13px;display:none}
.ty-on{background:rgba(74,143,208,.15);border:1px solid var(--lvB);color:#bcd8f5;display:block}
button{background:var(--acc);border:0;color:#fff;border-radius:8px;padding:8px 18px;
  font-size:14px;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.err{color:var(--lvR)}
h2{font-size:15px;color:var(--dim);font-weight:600;margin-bottom:10px}
#chart,#hchart{width:100%;height:340px}
select{background:#0d1420;color:var(--txt);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
.row{display:flex;justify-content:space-between;align-items:center}
</style></head><body><div class="wrap">
<header><h1>桐洲岛淹没预警<small>淹没阈值 新桐乡 黄6.5m / 红7.0m</small></h1>
<div><span id="meta">加载中…</span> <button id="btn" onclick="refresh()">立即刷新</button></div></header>

<div class="card"><div class="row">
  <div><span class="lvname">当前风险等级</span><span id="badge" class="badge b-绿">绿</span>
    <span id="est" style="margin-left:12px;color:var(--dim);font-size:13px"></span></div>
</div><div id="reasons"></div><div id="stations"></div><div id="tyline"></div></div>

<div class="card"><h2>近72小时水位（实况拉取）</h2><div id="chart"></div></div>

<div class="card"><h2>事件复盘（历史样本）</h2>
<div class="row" style="margin-bottom:10px">
  <select id="evsel" onchange="loadHist(this.value)">
    <option value="2026-07-13">2026-07-13 台风+泄洪</option>
    <option value="2026-08-09">2026-08-09 台风外围</option>
  </select><span style="color:var(--dim);font-size:12px">色带 = 逐时预警等级</span></div>
<div id="hchart"></div></div>
</div>
<script>
const charts = {live: echarts.init(document.getElementById('chart')),
                hist: echarts.init(document.getElementById('hchart'))};
const LV = {绿:'#2fa46a',黄:'#e0a63a',红:'#e05252'};
const NAMES = {baxia:'坝下',luzhu:'渌渚',lzz:'渌渚镇',zk:'闸口',xt:'新桐乡',baxia24:'坝下24h均值'};

function baseOpt(series, markLines){
  return {backgroundColor:'transparent',tooltip:{trigger:'axis'},
  legend:{textStyle:{color:'#7d8fa8'},top:0},
  grid:{left:44,right:16,top:34,bottom:44},
  xAxis:{type:'time',axisLine:{lineStyle:{color:'#233250'}},axisLabel:{color:'#7d8fa8'}},
  yAxis:{scale:true,axisLabel:{color:'#7d8fa8'},splitLine:{lineStyle:{color:'#1a2540'}}},
  series:series.map(s=>({name:s.name,type:'line',showSymbol:false,data:s.data,
    lineStyle:{width:s.wide?2.5:1.4,color:s.color||'#5aa2ff'},color:s.color,
    markLine:s.mark?{symbol:'none',silent:true,label:{color:s.markColor||'#e0a63a',formatter:s.mark},
      lineStyle:{color:s.markColor||'#e0a63a',type:'dashed'},data:[{yAxis:s.mark}]}:undefined}))};
}

async function loadLatest(){
  const d = await (await fetch('/api/latest')).json();
  const st = await (await fetch('/api/status')).json();
  const meta = document.getElementById('meta');
  if(!d){ meta.innerHTML = st.running ? '首次拉取中（约2分钟，受接口限频）…' :
    `<span class="err">尚无数据</span>`; return; }
  const ago = st.running ? ' [刷新中…]' : '';
  meta.innerHTML = st.error ? `<span class="err">上次成功 ${d.fetched_at}${ago} · 拉取失败: ${st.error}</span>`
    : `数据时间 ${d.fetched_at}${ago}`;
  const b = document.getElementById('badge');
  b.textContent = d.level; b.className = 'badge b-' + d.level;
  document.getElementById('est').textContent =
    [d.est_dry!=null?`干流回归峰值≈${d.est_dry}m`:'', d.est_mom!=null?`动量外推峰值≈${d.est_mom}m`:''].filter(x=>x).join(' · ');
  document.getElementById('reasons').innerHTML =
    d.reasons.length ? d.reasons.map(r=>'· '+r).join('<br>') : '<span style="color:var(--dim)">各站低于黄警阈值</span>';
  const ST = {baxia:'坝下',baxia24:'坝下24h均值',luzhu:'渌渚',zk:'闸口'};
  const R6 = {lz_r6:'渌渚6h',lzz_r6:'渌渚镇6h'};
  document.getElementById('stations').innerHTML =
    Object.entries(ST).map(([k,n])=>`<div class="st"><span>${n}</span><b>${d.stations[k]??'—'} m</b></div>`).join('')
    + Object.entries(R6).map(([k,n])=>`<div class="st"><span>${n}涨幅</span><b>${d.stations[k]!=null?(d.stations[k]>0?'+':'')+d.stations[k]+' m':'—'}</b></div>`).join('');
  const ty = document.getElementById('tyline');
  if(d.typhoon && d.typhoon.typhoons && d.typhoon.typhoons.length){
    ty.className = 'tyline' + (d.typhoon.prealert?' ty-on':'');
    ty.innerHTML = '台风: ' + d.typhoon.typhoons.map(t=>t.error ? `${t.name} 拉取失败` :
      `${t.name}(${t.num}) ${t.grade} 距岛${t.cur_km}km${t.fc_min_km!=null?` · 72h预报最近${t.fc_min_km}km`:''}`).join('；')
      + (d.typhoon.prealert?' — <b>预备级: 台风将影响, 预期水库预泄+支流涨水, 提前巡查</b>':'');
  } else ty.className='tyline';
  const S = d.series;
  charts.live.setOption(baseOpt([
    {name:NAMES.baxia,data:S.baxia,color:'#5aa2ff'},
    {name:NAMES.baxia24,data:S.baxia24,color:'#9d7ae8',wide:true,mark:7.0},
    {name:NAMES.luzhu,data:S.luzhu,color:'#e8c26a',wide:true,mark:7.0,markColor:'#e0a63a'},
    {name:NAMES.zk,data:S.zk,color:'#6acfe8',mark:6.0},
  ].filter(s=>s.data)),true);
}

const WIN = {'2026-07-13':['2026-07-10','2026-07-16'],'2026-08-09':['2026-08-07','2026-08-13']};
async function loadHist(day){
  const [a,b] = WIN[day];
  const d = await (await fetch(`/api/history?start=${a}&end=${b}`)).json();
  const lv = d.levels;
  const pieces = []; let s0 = 0;
  for(let i=1;i<=lv.length;i++) if(i===lv.length||lv[i]!==lv[s0]){
    pieces.push([{xAxis:d.ts[s0],itemStyle:{color:LV[lv[s0]]+'18'},
                  label:{show:i-s0>8,color:LV[lv[s0]],formatter:lv[s0],position:'insideTop'}},
                 {xAxis:i<lv.length?d.ts[i]:d.ts[lv.length-1]}]);
    s0=i;}
  charts.hist.setOption(baseOpt([
    {name:NAMES.baxia,data:d.series.baxia,color:'#5aa2ff'},
    {name:NAMES.luzhu,data:d.series.luzhu,color:'#e8c26a',wide:true,mark:7.0},
    {name:NAMES.xt,data:d.series.xt,color:'#ff9d5c',wide:true,mark:6.5},
    {name:NAMES.zk,data:d.series.zk,color:'#6acfe8'},
  ]));
  charts.hist.setOption({series:[{},{markArea:{silent:true,data:pieces}},{},{}]});
}

async function refresh(){
  const btn = document.getElementById('btn'); btn.disabled = true;
  await fetch('/api/refresh',{method:'POST'});
  const t0 = Date.now();
  const poll = async () => {
    const st = await (await fetch('/api/status')).json();
    if(!st.running || Date.now()-t0 > 240000){ btn.disabled=false; loadLatest(); return; }
    document.getElementById('meta').innerHTML = '拉取中（约2分钟，受接口限频）…';
    setTimeout(poll, 5000);
  };
  poll();
}

loadLatest(); loadHist('2026-07-13');
setInterval(loadLatest, 60000);
window.onresize = () => Object.values(charts).forEach(c=>c.resize());
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/latest":
            self._json(_load_cache())
        elif self.path == "/api/status":
            with _state["lock"]:
                self._json({"running": _state["running"], "fetched_at": _state["fetched_at"],
                            "error": _state["error"]})
        elif self.path.startswith("/api/history"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            try:
                self._json(history_range(q["start"][0], q["end"][0]))
            except Exception as e:
                self._json({"error": repr(e)[:120]}, 400)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/refresh":
            threading.Thread(target=_fetch_once, daemon=True).start()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *a):  # 静默访问日志
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    c = _load_cache()
    _state["fetched_at"] = c["fetched_at"] if c else None
    threading.Thread(target=_bg_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"Dashboard → http://localhost:{a.port}  (每小时自动拉取实况, Ctrl+C 退出)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出")


if __name__ == "__main__":
    main()
