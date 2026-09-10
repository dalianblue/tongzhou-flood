#!/usr/bin/env python3
"""台风因素 (typhoon.py) — 中央气象台台风网公开接口 (typhoon.nmc.cn)

背景: 2026年两次桐洲岛淹没均有台风背景 —
  7/12 台风"巴威"(BAVI/2609) 正面过境(距岛13km, STS级) + 水库台风前预泄;
  8/9 台风"白海豚"(DOLPHIN/2613) 外围雨带(距岛205km) → 渌渚江支流暴涨.
水库接到台风预警会提前放水, 坝下抬升先于台风到达 — 该信号已由泄洪型判据捕捉;
台风模块的作用是提供更早(48~72h)的"预备级"提示.

接口(公开JSONP, 无需登录):
  list_default → 当年全部台风(含停止编号): t=[id, 名en, 名cn, 编号, ?, ?, 寓意, status]
  view_{id}    → ty=[id,名en,名cn,编号,...,status, 轨迹点[8], 字段映射[9]]
  轨迹点: [id, 'YYYYMMDDHHMM', epoch_ms, 强度, 经度, 纬度, 气压, 风速, 移向, 移速, 风圈, {机构:预报点}, ...]
  预报点: [+Nh, 时刻, 经度, 纬度, 气压, 风速, 机构, 强度]
用法:
  python typhoon.py --harvest    # 一次性收割全年轨迹 → data/typhoon_tracks_2026.csv (研究样本)
  python typhoon.py --live       # 当前活动台风 + 72h预报路径距岛威胁
"""
import argparse
import json
import re
import time
import urllib.request
from math import cos, hypot, radians
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRACKS_CSV = BASE / "data" / "typhoon_tracks_2026.csv"

ISLAND_LON, ISLAND_LAT = 119.93, 29.87  # 桐洲岛
THREAT_KM = 500.0   # 预备级: 72h预报路径距岛阈值
THREAT_H = 72


def _get_jsonp(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    txt = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")
    return json.loads(re.search(r"({.*})", txt.strip(), re.DOTALL).group(1))


def dist_km(lon: float, lat: float) -> float:
    return hypot((lon - ISLAND_LON) * 111 * cos(radians(ISLAND_LAT)),
                 (lat - ISLAND_LAT) * 111)


def list_typhoons() -> list:
    d = _get_jsonp("https://typhoon.nmc.cn/weatherservice/typhoon/jsons/list_default")
    return d.get("typhoonList", [])


def fetch_detail(tid: str) -> list:
    return _get_jsonp(f"https://typhoon.nmc.cn/weatherservice/typhoon/jsons/view_{tid}")["typhoon"]


def harvest():
    """当年全部台风实况轨迹 → CSV (tid,名称,编号,时刻,强度,经度,纬度,距岛km)."""
    rows = []
    for t in list_typhoons():
        tid, name, num = str(t[0]), t[2], t[3]
        ty = fetch_detail(tid)
        for p in ty[8]:
            rows.append((tid, name, num, p[1], p[3], p[4], p[5], round(dist_km(p[4], p[5]), 1)))
        time.sleep(1)  # 礼貌间隔
    TRACKS_CSV.parent.mkdir(exist_ok=True)
    with open(TRACKS_CSV, "w", encoding="utf-8") as f:
        f.write("tid,name,num,ts,grade,lng,lat,dist_km\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"{len(rows)} 轨迹点 / {len(set(r[0] for r in rows))} 台风 → {TRACKS_CSV}")


def load_tracks():
    import csv
    if not TRACKS_CSV.exists():
        return []
    with open(TRACKS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def live_threat() -> dict:
    """活动台风 + BABJ 72h预报路径最小距岛距离 → 预备级判定."""
    active = [t for t in list_typhoons() if t[7] != "stop"]
    out = {"typhoons": [], "prealert": False}
    for t in active:
        tid, name = str(t[0]), t[2]
        try:
            ty = fetch_detail(tid)
            pts = ty[8]
            last = pts[-1]
            cur = dist_km(last[4], last[5])
            fc_min, fc_at = None, ""
            for agency_pts in (last[11] or {}).values():  # 任一机构预报, 后取到的覆盖
                for fp in agency_pts:
                    h = fp[0]
                    if h <= THREAT_H and (fc_min is None or fp[2] is None is False):
                        d = dist_km(fp[2], fp[3])
                        if fc_min is None or d < fc_min:
                            fc_min, fc_at = d, f"+{h}h"
            rec = {"name": name, "num": ty[3], "grade": last[3], "cur_km": round(cur),
                   "fc_min_km": round(fc_min) if fc_min is not None else None, "fc_at": fc_at}
            out["typhoons"].append(rec)
            if (fc_min is not None and fc_min <= THREAT_KM) or cur <= THREAT_KM:
                out["prealert"] = True
        except Exception as e:
            out["typhoons"].append({"name": name, "error": repr(e)[:60]})
    return out


def _cmd_live():
    r = live_threat()
    if not r["typhoons"]:
        print("当前无活动台风")
        return
    for t in r["typhoons"]:
        if "error" in t:
            print(f"{t['name']}: 拉取失败 {t['error']}")
            continue
        print(f"{t['name']}({t['num']}) {t['grade']} 当前距岛 {t['cur_km']}km"
              + (f" | 72h预报路径最近 {t['fc_min_km']}km ({t['fc_at']})" if t["fc_min_km"] else " | 无预报"))
    print("\n预备级(蓝):", "触发 — 72h内台风影响≤500km, 水库大概率预泄, 关注坝下抬升与支流涨幅"
          if r["prealert"] else "未触发")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.harvest:
        harvest()
    elif a.live:
        _cmd_live()
    else:
        ap.print_help()
