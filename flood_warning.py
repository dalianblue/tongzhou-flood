#!/usr/bin/env python3
"""桐洲岛洪水预警模型 (flood_warning.py)

背景: 2026-07-13 与 2026-08-09 桐洲岛两次被淹, 主因上游泄洪(7月)/台风+支流+顶托(8月).
8/9 新桐乡 11.31m 读数为洪水期传感器故障(同时刻上游坝下仅7.5m, 物理不可能),
真实峰值约 7.1m; 7月真实峰值 7.61m. 淹没阈值: 新桐乡(7011AJ4M) ≥6.5m 黄 / ≥7.0m 红.

规则集 (2026-06~08 样本逐时校准):
  黄-泄洪型: 坝下24h均值≥7.0 且 渌渚6h涨幅≥0.3   → 7月事件提前19h
  黄-支流型: 渌渚6h涨≥0.8 | 渌渚镇涨≥0.5 | 山溪涨≥0.6 → 8月事件提前39h (渌渚镇缓涨早可见)
  红-动量:   渌渚当前+6h涨幅 ≥7.3 (峰值外推)
  红-支流动量: 渌渚镇当前+6h涨幅 ≥8.3            → 8月首红提前9h
  红-泄洪级: 坝下24h均值≥8.6 或 坝下≥10.0
  顶托:     闸口≥6.0 时黄升红
回测(1632h): 4事件全中, 黄误报72h/红误报32h(过半为退水期滞后与真实大泄洪日, 详见 --backtest).

用法:
  python flood_warning.py --check                 # 实况风险卡 (4次API, 间隔26s)
  python flood_warning.py --backtest [--start --end]  # 归档回测 (零API)
  python flood_warning.py --demo 2026-08-09       # 单事件复盘 (零API)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_CSV = BASE_DIR / "data" / "hydro_sample_2026_07_08.csv"  # 研究样本: 2026-06-25~08-31, 12站×5min

# 站点 (zh 站号做键, architect.md A8)
ZM_BAXIA = "70101501"   # 富春江坝下 (干流泄洪)
ZM_LUZHU = "7010AJ33"   # 渌渚 (岛上游10km, 干流+渌渚江汇合)
ZM_LZZ = "7011AJ5M"     # 渌渚镇 (渌渚江支流)
ZM_XI = ["7011AJ7R", "7011AJ8R", "7011AJ9R"]  # 山溪水位站(基面50/65/194m, 只用涨幅)
ZM_ZHANKOU = "70102300"  # 闸口 (下游潮位, 顶托)
ZM_XT = "7011AJ4M"      # 新桐乡 (岛对岸锚站)

# 阈值: 2026全年归档校准 (两次淹没事件 + 全年误报扫描)
TH_BAXIA24_Y = 7.0      # 黄-泄洪: 坝下24h均值
TH_LZ_R6_Y = 0.3        # 黄-泄洪: 渌渚6h涨幅配合
TH_LZ_R6_Y2 = 0.8       # 黄-支流: 渌渚6h涨幅
TH_LZZ_R6_Y = 0.5       # 黄-支流: 渌渚镇6h涨幅 (原1.2; 支流缓涨型39h前即可见, 8月型首黄15h→39h)
TH_XI_R6_Y = 0.6        # 黄-支流: 山溪6h涨幅
TH_MOM_R = 7.3          # 红-动量: 渌渚+6h涨幅 外推峰值
TH_LZZ_MOM_R = 8.3      # 红-支流动量: 渌渚镇+6h涨幅 外推峰值 (支流型首红5h→9h)
TH_BAXIA24_R = 8.6      # 红-泄洪级: 坝下24h均值
TH_BAXIA_R = 10.0       # 红-泄洪级: 坝下瞬时
TH_ZK_HOLD = 6.0        # 顶托: 闸口潮位
TH_FLOOD_Y = 6.5        # 新桐乡淹没阈值-黄
TH_FLOOD_R = 7.0        # 新桐乡淹没阈值-红
REG_A, REG_B = 2.141, 0.468  # 日均回归 新桐乡=2.141+0.468*坝下日均 (r=0.804)

STATIONS = dict(baxia=ZM_BAXIA, luzhu=ZM_LUZHU, lzz=ZM_LZZ, zk=ZM_ZHANKOU, xt=ZM_XT)  # + 溪站


def clean(s: pd.Series, cap: float = None) -> pd.Series:
    """毛刺清洗: 偏离1h滚动中位>1m置NaN; cap 为物理上限(下游不可能超过坝下全年峰)."""
    if cap is not None:
        s = s.where(s < cap)
    med = s.rolling("1h", center=True, min_periods=6).median()
    return s.mask((s - med).abs() > 1.0)


def load_archive_hours(start=None, end=None) -> pd.DataFrame:
    """归档 → 逐时中位 DataFrame (列: baxia/luzhu/lzz/xi1..3/zk/xt). 零API."""
    keep = {v: k for k, v in STATIONS.items()}
    for z in ZM_XI:
        keep[z] = "xi" + z[-2]
    parts = []
    for ch in pd.read_csv(ARCHIVE_CSV, usecols=["sample_ts", "zh", "sw"],
                          low_memory=False, chunksize=500000):
        parts.append(ch[ch.zh.isin(keep)])
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df.sample_ts)
    w = df.pivot_table(index="ts", columns=df.zh.map(keep), values="sw",
                       aggfunc="median").sort_index()
    if start or end:
        w = w.loc[start or w.index[0]: end or w.index[-1]]
    for c in w.columns:
        w[c] = clean(w[c])
    # 新桐乡非物理值: 坝下全年峰11.5m, 下游不可能>12m; 再校验与上游渌渚一致性(差>2m 剔除)
    # (8/9 洪水期传感器故障出 10~18m 读数, 上下游对照可全部剔除)
    w["xt"] = w["xt"].where(w["xt"] < 12)
    lz_ff = w["luzhu"].reindex(w.index, method="ffill", tolerance=pd.Timedelta("1h"))
    w["xt"] = w["xt"].where((w["xt"] - lz_ff).abs() <= 2.0)
    return w.resample("h").median()


def signals(h: pd.DataFrame) -> pd.DataFrame:
    """逐时水位 → 预警信号 (回测/实况共用)."""
    xi_cols = [c for c in h.columns if c.startswith("xi")]
    return pd.DataFrame({
        "baxia": h["baxia"],
        "baxia24": h["baxia"].rolling(24, min_periods=12).mean(),
        "baxia_r6": h["baxia"] - h["baxia"].shift(6),
        "lz": h["luzhu"],
        "lz_r6": h["luzhu"] - h["luzhu"].shift(6),
        "lzz": h["lzz"] if "lzz" in h else np.nan,
        "lzz_r6": h["lzz"] - h["lzz"].shift(6) if "lzz" in h else np.nan,
        "xi_r6": pd.concat([h[c] - h[c].shift(6) for c in xi_cols], axis=1).max(axis=1)
                 if xi_cols else np.nan,
        "zk": h["zk"],
        "xt": h["xt"] if "xt" in h else np.nan,
    })


def assess(sig: pd.Series) -> dict:
    """最新信号 → {level, reasons, est_peak}. 绿/黄/红."""
    reasons = []
    yellow = False
    # 黄-泄洪型
    if sig.baxia24 >= TH_BAXIA24_Y and sig.lz_r6 >= TH_LZ_R6_Y:
        yellow = True
        reasons.append(f"泄洪型: 坝下24h均值{sig.baxia24:.2f}≥{TH_BAXIA24_Y} 且 渌渚6h涨{sig.lz_r6:+.2f}")
    # 黄-支流型
    for name, v, th in [("渌渚", sig.lz_r6, TH_LZ_R6_Y2), ("渌渚镇", sig.lzz_r6, TH_LZZ_R6_Y),
                        ("山溪", sig.xi_r6, TH_XI_R6_Y)]:
        if not np.isnan(v) and v >= th:
            yellow = True
            reasons.append(f"支流型: {name}6h涨{v:+.2f}≥{th}")
    red = False
    # 现报级: 已过淹没阈值则保持 (涨幅归零不降级)
    if not np.isnan(sig.xt):
        if sig.xt >= TH_FLOOD_R:
            yellow = red = True
            reasons.append(f"现报: 新桐乡{sig.xt:.2f}≥{TH_FLOOD_R}")
        elif sig.xt >= TH_FLOOD_Y:
            yellow = True
            reasons.append(f"现报: 新桐乡{sig.xt:.2f}≥{TH_FLOOD_Y}")
    if not np.isnan(sig.lz) and sig.lz >= 7.0:
        yellow = True
        if sig.lz >= 7.5:
            red = True
        reasons.append(f"现报: 渌渚{sig.lz:.2f}" + ("≥7.5" if red else "≥7.0"))
    mom = sig.lz + max(0.0, sig.lz_r6) if not np.isnan(sig.lz) else np.nan
    if not np.isnan(mom) and mom >= TH_MOM_R:
        red = True
        reasons.append(f"动量外推峰值 {mom:.2f}≥{TH_MOM_R} (渌渚{sig.lz:.2f}+6h涨{sig.lz_r6:+.2f})")
    mom_lzz = sig.lzz + max(0.0, sig.lzz_r6) if not np.isnan(sig.lzz) else np.nan
    if not np.isnan(mom_lzz) and mom_lzz >= TH_LZZ_MOM_R:
        red = True
        reasons.append(f"支流动量外推峰值 {mom_lzz:.2f}≥{TH_LZZ_MOM_R} "
                       f"(渌渚镇{sig.lzz:.2f}+6h涨{sig.lzz_r6:+.2f})")
    if not np.isnan(sig.baxia24) and (sig.baxia24 >= TH_BAXIA24_R or sig.baxia >= TH_BAXIA_R):
        red = True
        reasons.append(f"泄洪级: 坝下24h均值{sig.baxia24:.2f}/瞬时{sig.baxia:.2f}")
    if yellow and not np.isnan(sig.zk) and sig.zk >= TH_ZK_HOLD:
        red = True
        reasons.append(f"潮位顶托: 闸口{sig.zk:.2f}≥{TH_ZK_HOLD}, 黄升红")
    level = "红" if red else ("黄" if yellow else "绿")
    est_dry = REG_A + REG_B * sig.baxia24 if not np.isnan(sig.baxia24) else np.nan
    return {"level": level, "reasons": reasons, "est_mom": mom, "est_dry": est_dry}


def _events(h: pd.DataFrame) -> list:
    """事件窗口: 新桐乡≥6.5 或 渌渚≥7.0 (连续, 间隔>24h 拆分)."""
    m = ((h["xt"] >= TH_FLOOD_Y) | (h["luzhu"] >= 7.0)).fillna(False)
    if not m.any():
        return []
    grp = (m != m.shift()).cumsum()
    out = []
    for _, g in h[m].groupby(grp[m]):
        if out and g.index.min() - out[-1][1] <= pd.Timedelta(hours=24):
            out[-1] = (out[-1][0], g.index.max())  # 合并退水期反复过阈
        else:
            out.append((g.index.min(), g.index.max()))
    return out


def cmd_backtest(start=None, end=None):
    h = load_archive_hours(start, end)
    sig = signals(h)
    ev = _events(h)
    lv = sig.apply(assess, axis=1)
    y = lv.map(lambda d: d["level"] in ("黄", "红"))
    r = lv.map(lambda d: d["level"] == "红")
    # 误报: 报警后24h内无过阈(新桐乡≥6.5 或 渌渚≥7.0)
    fut = ((h["xt"].iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1].shift(-1) >= TH_FLOOD_Y)
           | (h["luzhu"].iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1].shift(-1) >= 7.0)
           ).fillna(False)
    print(f"回测区间 {h.index.min():%Y-%m-%d} ~ {h.index.max():%Y-%m-%d} ({len(h)}h)")
    print(f"黄以上 {y.sum()}h (其中误报 {(y & ~fut).sum()}h) | 红 {r.sum()}h (误报 {(r & ~fut).sum()}h)\n")
    ok = True
    for a, b in ev:
        # 首黄/首红: 过阈前48h内首个亮警点 (黄警允许闪烁; 48h外的孤立误报不算首报)
        def first_alert(v):
            w = v.loc[a - pd.Timedelta(hours=48):a]
            on = w.index[w]
            return on.min() if len(on) else None
        fy = first_alert(y)
        lead = (a - fy).total_seconds() / 3600 if fy is not None else float("nan")
        fr = first_alert(r)
        lead_r = (a - fr).total_seconds() / 3600 if fr is not None else float("nan")
        tag = "★" if not np.isnan(lead) and lead >= 6 else "!"
        if np.isnan(lead) or lead < 6:
            ok = False
        pk = np.nanmax(h.loc[a - pd.Timedelta(hours=6):b + pd.Timedelta(hours=6), ["xt", "luzhu"]].values)
        print(f"{tag} 事件 {a:%m-%d %H:%M}~{b:%m-%d %H:%M} 峰值~{pk:.2f}m | "
              f"首黄 {(f'{fy:%m-%d %H:%M} 提前{lead:.0f}h') if fy is not None else '漏报'} | "
              f"首红 {(f'{fr:%m-%d %H:%M} 提前{lead_r:.0f}h') if fr is not None else '未升级'}")
    fa = (y & ~fut)
    if fa.any():
        print("\n黄警误报时段 (无后续过阈):")
        grp = (fa != fa.shift()).cumsum()
        for _, g in h[fa].groupby(grp[fa]):
            print(f"  {g.index.min():%m-%d %H:%M}~{g.index.max():%m-%d %H:%M} ({len(g)}h)")
    print("\n自检:", "PASS (全部事件黄警提前≥6h)" if ok else "FAIL (存在漏报或提前量<6h)")
    return 0 if ok else 1


def cmd_demo(date: str):
    d = pd.Timestamp(date)
    h = load_archive_hours(str(d.date() - pd.Timedelta(days=1)), str(d.date() + pd.Timedelta(days=2)))
    sig = signals(h)
    print(f"=== {date} 前后复盘 (逐3h) ===")
    cols = ["baxia", "baxia24", "lz", "lz_r6", "lzz_r6", "xi_r6", "zk", "xt"]
    lv = sig.apply(assess, axis=1)
    view = sig[cols].iloc[::3].round(2)
    view["预警"] = [d["level"] for d in lv[::3]]
    # 台风距岛列 (来自 typhoon.py --harvest 的轨迹缓存)
    try:
        import typhoon as TYPH
        tr = [r for r in TYPH.load_tracks()
              if abs((pd.Timestamp(r["ts"]) - d).total_seconds()) < 3 * 86400]
        if tr:
            tdf = pd.DataFrame({"ts": pd.to_datetime([r["ts"] for r in tr]),
                                "d": [float(r["dist_km"]) for r in tr],
                                "v": [f"{r['name']}{float(r['dist_km']):.0f}km"
                                      for r in tr]}).sort_values("d")
            tdf = tdf.drop_duplicates("ts", keep="first")       # 每时刻取距岛最近的台风
            tdf = tdf.set_index("ts").sort_index()
            tdf.index = tdf.index.tz_localize("Asia/Shanghai")  # 对齐 view.index (+08:00)
            view["台风"] = tdf.reindex(view.index, method="ffill")["v"].fillna("")
    except Exception:
        pass
    print(view.loc[str(d.date()):].to_string())
    print("\n图例: 阈值 黄-泄洪 坝下24≥7.0&渌渚涨≥0.3 | 黄-支流 渌渚涨≥0.8/渌渚镇≥0.5/山溪≥0.6 | "
          f"红 动量≥{TH_MOM_R} 或 渌渚镇动量≥{TH_LZZ_MOM_R} 或 坝下24≥8.6 或 坝下≥10 或 黄+闸口≥6.0")


def _fetch_window(zm: str, jg: int, hours: float, timeout: int = 90,
                  st=None, et=None) -> pd.Series:
    """浙江水利厅 getHisData 窗口拉取 (自 pyTides realtime_predictor 精简, 含超时).
    默认 et=now 往前 hours; 也可直接给 st/et (datetime, 无tz北京时间)."""
    import json
    import ssl
    import urllib.request
    from datetime import datetime, timedelta
    if et is None:
        et = datetime.now()
    if st is None:
        st = et - timedelta(hours=hours)
    url = (f"https://sqfb.slt.zj.gov.cn/rest/water/getHisData?zm={zm}"
           f"&st={st:%Y-%m-%dT%H:%M:%S}&et={et:%Y-%m-%dT%H:%M:%S}&jg={jg}&lx=0")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = None
    for _ in range(3):  # 502/504 偶发, 重试
        try:
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=ssl.create_default_context()) as r:
                d = json.loads(r.read().decode("utf-8"))
            break
        except Exception:
            time.sleep(8)
    rows = (d or {}).get("pz") or []
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["tm"] = pd.to_datetime(df["tm"])  # 无tz, 北京时间
    df["sw"] = pd.to_numeric(df["sw"], errors="coerce")
    s = df.dropna(subset=["sw"]).set_index("tm")["sw"]
    s.index = s.index.tz_localize("Asia/Shanghai")
    return s


def live_snapshot(hours: float = 72.0) -> dict:
    """实况快照: 4次API (坝下/渌渚/渌渚镇/闸口, 各hours窗口), 间隔26s限频.
    返回 {time, series(逐时清洗后, tz-naive北京时间字符串), sig, assess}."""
    plan = [(ZM_BAXIA, "baxia"), (ZM_LUZHU, "luzhu"),
            (ZM_LZZ, "lzz"), (ZM_ZHANKOU, "zk")]
    got = {}
    for i, (zm, col) in enumerate(plan):
        s = _fetch_window(zm, 2, hours)
        if len(s):
            got[col] = s
        if i < len(plan) - 1:
            time.sleep(26)  # 限频纪律
    if not got:
        raise RuntimeError("API 拉取全部失败")
    df = pd.DataFrame({k: clean(v) for k, v in got.items()})
    h = df.resample("h").median()
    if "lzz" not in h:
        h["lzz"] = np.nan
    series = {c: [[t.strftime("%Y-%m-%dT%H:%M"), None if np.isnan(v) else round(v, 3)]
                  for t, v in h[c].items()] for c in h.columns}
    sg = signals(h)
    series["baxia24"] = [[t.strftime("%Y-%m-%dT%H:%M"), None if np.isnan(v) else round(v, 3)]
                         for t, v in sg["baxia24"].items()]
    sig = sg.iloc[-1]
    return {"time": h.index.max().strftime("%Y-%m-%d %H:%M"),
            "series": series, "sig": sig, "assess": assess(sig)}


def cmd_check():
    """实况风险卡: live_snapshot (4次API, ~2分钟限频) + 台风预备级."""
    try:
        snap = live_snapshot()
    except RuntimeError as e:
        print(e)
        return 1
    sig, r = snap["sig"], snap["assess"]
    now = snap["time"]
    print(f"=== 桐洲岛洪水风险卡 @ {now:%Y-%m-%d %H:%M} ===")
    print(f"坝下 {sig.baxia:.2f}m (24h均值{sig.baxia24:.2f}) | 渌渚 {sig.lz:.2f}m (6h{sig.lz_r6:+.2f}) | "
          f"渌渚镇6h{sig.lzz_r6:+.2f} | 闸口 {sig.zk:.2f}m")
    est = []
    if not np.isnan(r["est_dry"]):
        est.append(f"干流回归峰值(下限) {r['est_dry']:.2f}m")
    if not np.isnan(r["est_mom"]):
        est.append(f"动量外推峰值 {r['est_mom']:.2f}m")
    if est:
        print(" | ".join(est))
    print(f"\n风险等级: 【{r['level']}】")
    for x in r["reasons"]:
        print(f"  - {x}")
    if r["level"] == "绿":
        print(f"  (各站低于黄警阈值; 淹没阈值 新桐乡 黄{TH_FLOOD_Y}m/红{TH_FLOOD_R}m)")
    # 台风因素 (预备级, 独立于水位等级): 72h预报路径距岛≤500km → 水库大概率预泄
    try:
        import typhoon as TYPH
        th = TYPH.live_threat()
        print("\n台风因素 (预备级):")
        if not th["typhoons"]:
            print("  当前无活动台风")
        for t in th["typhoons"]:
            if "error" in t:
                print(f"  {t['name']}: 拉取失败 {t['error']}")
            else:
                line = f"  {t['name']}({t['num']}) {t['grade']} 距岛{t['cur_km']}km"
                if t["fc_min_km"] is not None:
                    line += f", 72h预报最近{t['fc_min_km']}km({t['fc_at']})"
                print(line)
        if th["prealert"]:
            print("  ★ 预备级(蓝)触发: 台风将影响 → 预期水库预泄(坝下将抬升) + 支流涨水, 提前巡查准备")
    except Exception as e:
        print(f"\n台风因素: 拉取失败({repr(e)[:50]}), 跳过")
    return 0


def cmd_fill_gaps():
    """补样本缺日: 采集端每周停一天 (每周同一天仅整点12条), 用 API 按站×日回补.
    12站×缺日 次调用, 间隔26s限频, 502/504 自动重试; 只补缺, 不覆盖已有行."""
    from datetime import datetime as dt
    df = pd.read_csv(ARCHIVE_CSV, low_memory=False)
    df["ts"] = pd.to_datetime(df.sample_ts)
    days = df.groupby(df.ts.dt.date).size()
    gap_days = sorted(d for d, n in days.items() if n < 100)
    if not gap_days:
        print("无缺日, 无需补数")
        return 0
    print(f"缺日 {len(gap_days)} 天: {', '.join(str(d) for d in gap_days)}")
    meta_cols = [c for c in df.columns if c not in ("sample_ts", "sbsj", "ts", "sw")]
    meta = df.groupby("zh")[meta_cols].first()
    zh_list = sorted(df.zh.unique())
    calls = [(zh, d) for zh in zh_list for d in gap_days]
    new_rows = []
    for i, (zh, d) in enumerate(calls):
        s = _fetch_window(zh, 2, 0, st=dt(d.year, d.month, d.day), et=dt(d.year, d.month, d.day, 23, 59))
        for ts, sw in s.items():
            row = dict(meta.loc[zh])
            row.update(sample_ts=ts.isoformat(), sbsj=ts.strftime("%Y-%m-%dT%H:%M:%S"), sw=sw)
            new_rows.append(row)
        print(f"[{i+1}/{len(calls)}] {zh} {d}: +{len(s)} 行", flush=True)
        if i < len(calls) - 1:
            time.sleep(26)
    out = pd.concat([df.drop(columns=["ts"]), pd.DataFrame(new_rows, columns=df.columns[:-1])])
    out = out.drop_duplicates(subset=["sample_ts", "zh"], keep="first").sort_values(["sample_ts", "zh"])
    out.to_csv(ARCHIVE_CSV, index=False)
    print(f"补 {len(new_rows)} 行 → {ARCHIVE_CSV} (现共 {len(out)} 行)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="实况风险卡 (4次API)")
    ap.add_argument("--backtest", action="store_true", help="归档回测 (零API)")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--demo", metavar="YYYY-MM-DD", help="单事件复盘 (零API)")
    ap.add_argument("--fill-gaps", action="store_true", help="API 回补样本缺日 (每周缺一天)")
    a = ap.parse_args()
    if a.backtest:
        sys.exit(cmd_backtest(a.start, a.end))
    if a.demo:
        cmd_demo(a.demo); return
    if a.fill_gaps:
        sys.exit(cmd_fill_gaps())
    if a.check:
        sys.exit(cmd_check())
    ap.print_help()


if __name__ == "__main__":
    main()
