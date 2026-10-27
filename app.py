"""
Dice Battle Live  -  Python + Flask + Pillow + numpy + ffmpeg

Termux install:
    pkg update
    pkg install python python-pillow python-numpy ffmpeg
    pip install flask gTTS          # gTTS = Hindi awaaz (internet chahiye)
    (gTTS na chale to:  pkg install espeak  -> robot jaisi Hindi awaaz)
Run:    python app.py
Open:   http://127.0.0.1:5000   -> Admin panel + Settings

Game kabhi apne aap khatam nahi hota: points lagatar jodte rehte hain jab tak
server on hai. Aakhir mein Admin panel ka "FINAL RESULT" button dabate hi
winner dikhta hai. "NAYA GAME" se sab 0 se shuru.
"""
import collections
import hashlib
import io
import json
import math
import os
import queue
import random
import re
import shutil
import struct
import subprocess
import threading
import time

import numpy as np
from flask import Flask, Response, abort, jsonify, redirect, render_template_string, request, send_file
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
TTS_DIR = os.path.join(DATA, "tts")
os.makedirs(TTS_DIR, exist_ok=True)
CFG_PATH = os.path.join(DATA, "config.json")
STATE_PATH = os.path.join(DATA, "state.json")
LOG_PATH = os.path.join(DATA, "ffmpeg.log")

SR = 44100

# quality: (width, height, fps, bitrate, x264 preset)
QUALITY = {
    "lite": (480, 854, 24, "1300k", "ultrafast"),
    "low": (540, 960, 30, "2000k", "veryfast"),
    "hd": (720, 1280, 30, "3000k", "superfast"),
}

DEFAULTS = {
    "stream_key": "",
    "title": "DICE BATTLE",
    "red_name": "RED",
    "blue_name": "BLUE",
    "red_say": "",      # awaaz mein bolne ka naam (Hindi mein likho to behtar)
    "blue_say": "",
    "quality": "low",
    "auto_live": False,
    "voice_on": True,
    "music_vol": 45,
    "sfx_vol": 85,
    "admin_pass": "",
}
_cfg_lock = threading.Lock()


def load_cfg():
    cfg = dict(DEFAULTS)
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    if cfg.get("quality") not in QUALITY:
        cfg["quality"] = "low"
    w, h, fps, br, preset = QUALITY[cfg["quality"]]
    cfg.update(width=w, height=h, fps=fps, bitrate=br, preset=preset)
    return cfg


def save_cfg(cfg):
    keep = {k: cfg[k] for k in DEFAULTS}
    with _cfg_lock:
        tmp = CFG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(keep, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CFG_PATH)


# =========================================================== Hindi numbers
H100 = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस "
        "ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
        "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस "
        "इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस "
        "इकतालीस बयालीस तैंतालीस चौवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
        "इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ "
        "इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर "
        "इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी अस्सी "
        "इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे "
        "इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()


def hn(n):
    n = max(0, int(n))
    if n < 100:
        return H100[n]
    if n < 1000:
        h, r = divmod(n, 100)
        return H100[h] + " सौ" + (" " + H100[r] if r else "")
    th, r = divmod(n, 1000)
    if th < 100:
        return H100[th] + " हज़ार" + (" " + hn(r) if r else "")
    return str(n)


def say_name(cfg, side):
    s = (cfg.get(side + "_say") or "").strip()
    if s:
        return s
    n = (cfg.get(side + "_name") or "").strip()
    if n.upper() == "RED":
        return "रेड"
    if n.upper() == "BLUE":
        return "ब्लू"
    return n or ("रेड" if side == "red" else "ब्लू")


def roll_text(name, n, total, extra=""):
    if n == 1:
        head = "%s को सिर्फ़ एक पॉइंट मिला।" % name
    elif n == 6:
        head = "छक्का! %s को पूरे छह पॉइंट मिले।" % name
    else:
        head = "%s को %s पॉइंट मिले।" % (name, hn(n))
    return ("%s कुल %s पॉइंट। %s" % (head, hn(total), extra)).strip()


# ================================================================== audio
def _lp(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def _hp(x, k=8):
    return x - _lp(x, k)


def _sweep(freqs):
    return np.sin(2 * np.pi * np.cumsum(freqs) / SR)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def _track_lofi():
    """8 bar ka halka upbeat lofi loop (kisi se copy nahi, khud synthesize)."""
    rng = np.random.default_rng(1)
    bpm = 92.0
    beat = 60.0 / bpm
    bar = int(round(4 * beat * SR))
    prog = [(36, [60, 64, 67, 71]), (33, [57, 60, 64, 67]), (41, [57, 60, 65, 69]), (31, [55, 59, 62, 65])]
    bars = 8
    N = bar * bars
    out = np.zeros(N)

    def put(sig, pos):
        pos %= N
        e = pos + len(sig)
        if e <= N:
            out[pos:e] += sig
        else:
            k = N - pos
            out[pos:] += sig[:k]
            out[:e - N] += sig[k:]

    def at(b):
        return b * bar

    def B(x):
        return int(x * beat * SR)

    tb = np.arange(bar) / SR
    pad_env = np.minimum(1, tb / 0.35) * np.minimum(1, (bar / SR - tb) / 0.55)
    kt = np.arange(int(0.30 * SR)) / SR
    kick = _sweep(48 + 120 * np.exp(-kt * 30)) * np.exp(-kt * 10) * 0.60
    st = np.arange(int(0.22 * SR)) / SR
    snare = (_hp(rng.uniform(-1, 1, len(st))) * np.exp(-st * 22) * 0.20 +
             np.sin(2 * np.pi * 190 * st) * np.exp(-st * 30) * 0.12)
    ht = np.arange(int(0.06 * SR)) / SR
    hat = _hp(rng.uniform(-1, 1, len(ht)), 4) * np.exp(-ht * 85)
    for b in range(bars):
        root, chord = prog[b % 4]
        v = b // 4
        base = at(b)
        for m in chord:
            f = mtof(m)
            put((np.sin(2 * np.pi * f * tb) + 0.5 * np.sin(2 * np.pi * f * 1.004 * tb)) * pad_env * 0.034, base)
        for bt, ro in ((0, 0), (1.5, 0), (2, 7), (3.5, 0)):
            f = mtof(root + ro)
            n = B(0.75)
            t = np.arange(n) / SR
            put((np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 4.2) *
                np.minimum(1, t * 250) * 0.34, base + B(bt))
        order = [0, 1, 2, 3, 2, 1, 2, 3] if v == 0 else [0, 2, 1, 3, 2, 3, 1, 2]
        for i, ci in enumerate(order):
            f = mtof(chord[ci] + (12 if i % 4 == 3 else 0))
            n = int(0.55 * SR)
            t = np.arange(n) / SR
            ph = 2 * np.pi * f * t
            p = (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)) * np.exp(-t * 7.5) * np.minimum(1, t * 400)
            put(p * (0.085 if i % 2 == 0 else 0.06), base + B(i * 0.5))
            put(p * 0.03, base + B(i * 0.5 + 0.375))
        for bt in (0, 2) + ((3.5,) if v == 1 else ()):
            put(kick, base + B(bt))
        for bt in (1, 3):
            put(snare, base + B(bt))
        for i in range(8):
            put(hat * (0.11 if i % 2 else 0.065), base + B(i * 0.5))
    out *= 0.72 / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


def _put_loop(out, sig, pos):
    N = len(out)
    pos %= N
    e = pos + len(sig)
    if e <= N:
        out[pos:e] += sig
    else:
        k = N - pos
        out[pos:] += sig[:k]
        out[:e - N] += sig[k:]


def _track_arcade():
    """120 bpm khushnuma 'arcade / synth-pop' loop (8 bar)."""
    rng = np.random.default_rng(3)
    beat = 60.0 / 120.0
    bar = int(round(4 * beat * SR))
    bars = 8
    out = np.zeros(bar * bars)

    def B(x):
        return int(x * beat * SR)

    prog = [(36, [60, 64, 67]), (43, [59, 62, 67]), (45, [57, 60, 64]), (41, [57, 60, 65])]
    kt = np.arange(int(0.28 * SR)) / SR
    kick = _sweep(50 + 130 * np.exp(-kt * 28)) * np.exp(-kt * 11) * 0.62
    ct = np.arange(int(0.16 * SR)) / SR
    clap = _ff(rng.uniform(-1, 1, len(ct)), 900, 5200) * np.exp(-ct * 28) * 0.55
    ht = np.arange(int(0.05 * SR)) / SR
    hat = _ff(rng.uniform(-1, 1, len(ht)), 5000, None) * np.exp(-ht * 100)
    ot = np.arange(int(0.18 * SR)) / SR
    ohat = _ff(rng.uniform(-1, 1, len(ot)), 5000, None) * np.exp(-ot * 18)
    tb = np.arange(bar) / SR
    pad_env = np.minimum(1, tb / 0.2) * np.minimum(1, (bar / SR - tb) / 0.3)
    arp_order = [0, 1, 2, 1, 0, 1, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1]
    for b in range(bars):
        root, chord = prog[b % 4]
        base = b * bar
        for i in range(4):
            _put_loop(out, kick, base + B(i))
        for bt in (1, 3):
            _put_loop(out, clap, base + B(bt))
        for i in range(8):
            _put_loop(out, hat * (0.10 if i % 2 == 0 else 0.06), base + B(i * 0.5))
        _put_loop(out, ohat * 0.08, base + B(1.5))
        _put_loop(out, ohat * 0.08, base + B(3.5))
        for i in range(8):
            f = mtof(root + (12 if i in (3, 7) else 0))
            n = B(0.45)
            t = np.arange(n) / SR
            s = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) +
                 0.25 * np.sin(6 * np.pi * f * t)) * np.exp(-t * 6) * np.minimum(1, t * 300)
            _put_loop(out, s * 0.26, base + B(i * 0.5))
        for i, ci in enumerate(arp_order):
            f = mtof(chord[ci] + 12)
            n = B(0.4)
            t = np.arange(n) / SR
            ph = 2 * np.pi * f * t
            s = (np.sin(ph) + np.sin(3 * ph) / 3 + np.sin(5 * ph) / 5) * np.exp(-t * 13) * np.minimum(1, t * 400)
            _put_loop(out, s * 0.055, base + B(i * 0.25))
            _put_loop(out, s * 0.018, base + B(i * 0.25 + 0.375))
        for m in chord:
            f = mtof(m)
            _put_loop(out, (np.sin(2 * np.pi * f * tb) + 0.5 * np.sin(2 * np.pi * f * 1.005 * tb)) * pad_env * 0.03, base)
    out *= 0.72 / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


def _track_piano():
    """Narm 'piano / chill' loop, 74 bpm (8 bar) - raat ke liye."""
    rng = np.random.default_rng(11)
    beat = 60.0 / 74.0
    bar = int(round(4 * beat * SR))
    bars = 8
    out = np.zeros(bar * bars)

    def B(x):
        return int(x * beat * SR)

    prog = [(38, [62, 65, 69, 72]), (46, [58, 62, 65, 69]), (41, [57, 60, 65, 69]), (36, [60, 64, 67, 69])]
    pent = [74, 77, 79, 81, 84]
    tb = np.arange(bar) / SR
    pad_env = np.minimum(1, tb / 0.6) * np.minimum(1, (bar / SR - tb) / 0.9)
    kt = np.arange(int(0.3 * SR)) / SR
    kick = _sweep(45 + 90 * np.exp(-kt * 26)) * np.exp(-kt * 9) * 0.30
    ht = np.arange(int(0.07 * SR)) / SR
    brush = _ff(rng.uniform(-1, 1, len(ht)), 3500, 9000) * np.exp(-ht * 55)
    for b in range(bars):
        root, chord = prog[b % 4]
        base = b * bar
        for m in chord:
            f = mtof(m - 12)
            _put_loop(out, (np.sin(2 * np.pi * f * tb) + 0.4 * np.sin(2 * np.pi * f * 1.003 * tb)) * pad_env * 0.045, base)
        f = mtof(root)
        n = B(3.8)
        t = np.arange(n) / SR
        _put_loop(out, (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 1.1) *
                  np.minimum(1, t * 120) * 0.30, base)
        order = [0, 1, 2, 3, 2, 1, 2, 3]
        for i, ci in enumerate(order):
            f = mtof(chord[ci])
            n = int(1.0 * SR)
            t = np.arange(n) / SR
            ph = 2 * np.pi * f * t
            p = (np.sin(ph) + 0.3 * np.sin(2 * ph) * np.exp(-t * 6) + 0.12 * np.sin(3 * ph) * np.exp(-t * 9)) * \
                np.exp(-t * 4.2) * np.minimum(1, t * 300)
            _put_loop(out, p * (0.10 if i % 2 == 0 else 0.07), base + B(i * 0.5))
        for bt in (1.5, 2.5, 3.5):
            if rng.random() < 0.7:
                f = mtof(int(rng.choice(pent)))
                n = int(1.2 * SR)
                t = np.arange(n) / SR
                s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 6)) * np.exp(-t * 3.5)
                _put_loop(out, s * np.minimum(1, t * 800) * 0.06, base + B(bt))
        _put_loop(out, kick, base)
        _put_loop(out, kick * 0.8, base + B(2))
        for i in range(8):
            _put_loop(out, brush * (0.05 if i % 2 == 0 else 0.03), base + B(i * 0.5))
    out *= 0.72 / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


def make_music():
    """3 alag gaane (lofi, arcade, piano) ek ke baad ek, beech mein crossfade - kabhi ruka nahi."""
    tr = [_track_lofi(), _track_arcade(), _track_piano()]
    tr = [t / (np.sqrt(np.mean(t.astype(np.float64) ** 2)) + 1e-9) * 0.12 for t in tr]
    order = [0, 0, 1, 1, 2, 2]
    cf = int(2.0 * SR)
    fade = np.linspace(0.0, 1.0, cf)
    fin, fout = np.sin(fade * np.pi / 2), np.cos(fade * np.pi / 2)
    out = tr[order[0]].astype(np.float64)
    for prev, cur in zip(order, order[1:]):
        nxt = tr[cur].astype(np.float64)
        if prev == cur:
            out = np.concatenate((out, nxt))
        else:
            mid = out[-cf:] * fout + nxt[:cf] * fin
            out = np.concatenate((out[:-cf], mid, nxt[cf:]))
    head = out[:cf] * fin + out[-cf:] * fout     # aakhri gaana wapas pehle mein ghul jata hai
    out = np.concatenate((head, out[cf:-cf]))
    return out.astype(np.float32)


# ------------------------------------------------------------ sound effects
def _tt(dur):
    return np.arange(int(dur * SR)) / SR


def _ff(x, lo=None, hi=None, order=2):
    """FFT se band-filter (zero-phase): lo se neeche aur hi se upar ki awaaz kaat deta hai."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    g = np.ones(len(f))
    if lo:
        g = g / (1.0 + (lo / np.maximum(f, 1.0)) ** (2 * order))
    if hi:
        g = g / (1.0 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * g, len(x))


def _nrm(x, peak=0.9):
    m = float(np.max(np.abs(x))) or 1.0
    return (x * (peak / m)).astype(np.float32)


def _place(buf, sig, t, gain=1.0):
    i = int(t * SR)
    if i < 0 or i >= len(buf):
        return
    k = min(len(sig), len(buf) - i)
    if k > 0:
        buf[i:i + k] += sig[:k] * gain


def _knock(seed, size=0.22, bright=1.0):
    """Lakdi ki table par dice ka takrana: gehri thap + lakdi ki goonj + chhota click."""
    r = np.random.default_rng(seed)
    n = int(SR * size)
    t = np.arange(n) / SR
    f0 = r.uniform(140, 210)
    thump = np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.9 * np.exp(-t * 70))) / SR) * np.exp(-t * 34)
    modes = ((r.uniform(430, 560), 0.55, 62), (r.uniform(780, 1050), 0.45, 85),
             (r.uniform(1500, 2100), 0.30, 120), (r.uniform(2800, 3800), 0.20 * bright, 170))
    body = sum(a * np.sin(2 * np.pi * f * t + r.uniform(0, 6.28)) * np.exp(-t * d) for f, a, d in modes)
    nz = _ff(r.uniform(-1, 1, n), 900, 8000) * np.exp(-t * 260) * bright
    s = (1.0 * thump + 0.8 * body + 0.55 * nz) * np.minimum(1, t * 1800)
    return _nrm(s, 0.95)


def _tick(seed, size=0.05):
    """Plastic dice ka chhota 'tik'."""
    r = np.random.default_rng(seed)
    n = int(SR * size)
    t = np.arange(n) / SR
    f = r.uniform(1800, 3600)
    s = np.sin(2 * np.pi * f * t) * np.exp(-t * r.uniform(150, 260)) + \
        0.5 * np.sin(2 * np.pi * f * 1.9 * t) * np.exp(-t * 300)
    s = s + _ff(r.uniform(-1, 1, n), 2500, 9000) * np.exp(-t * 420) * 0.8
    return _nrm(s * np.minimum(1, t * 3000), 0.8)


def _rattle(seed, dur=0.55):
    """Haath/cup mein dice khadkhadane ki awaaz."""
    r = np.random.default_rng(seed)
    total = int(SR * (dur + 0.12))
    out = np.zeros(total)
    ticks = [_tick(seed * 10 + i) for i in range(6)]
    for _ in range(22):
        _place(out, ticks[int(r.integers(6))], dur * float(r.beta(2, 2)), float(r.uniform(0.25, 1.0)))
    for j in range(4):
        _place(out, _knock(seed * 7 + j, 0.12, 0.6), dur * float(r.uniform(0.1, 0.9)), float(r.uniform(0.2, 0.45)))
    tb = np.arange(total) / SR
    bed = _ff(r.uniform(-1, 1, total), 1500, 6500) * (np.sin(np.pi * np.clip(tb / dur, 0, 1)) ** 1.5) * 0.12
    return _nrm(out + bed, 0.8)


def _roll(seed, dur=1.1):
    """Table par dice ka ludhakna aur ruk jana."""
    r = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    p = 1 - t / dur
    bed = _ff(r.uniform(-1, 1, n), 200, 1300) * p ** 1.6 * (0.65 + 0.35 * np.sin(2 * np.pi * (9 + 6 * p) * t))
    out = bed / (np.max(np.abs(bed)) + 1e-9) * 0.35
    times = [0.10, 0.27, 0.42, 0.54, 0.64, 0.72, 0.78]
    for i, tm in enumerate(times):
        _place(out, _knock(seed * 13 + i, 0.12, 0.5), tm, 0.5 * (1 - i / len(times) * 0.8))
    return _nrm(out, 0.7)


def _whoosh(dur=0.55):
    r = np.random.default_rng(3)
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = r.uniform(-1, 1, n)
    p = t / dur
    lo = _ff(x, 150, 700) * np.sin(np.pi * p) ** 2
    hi = _ff(x, 1800, 6000) * np.sin(np.pi * np.clip(p * 1.6, 0, 1)) ** 2 * (1 - p)
    return _nrm(lo / (np.max(np.abs(lo)) + 1e-9) * 0.9 + hi / (np.max(np.abs(hi)) + 1e-9) * 0.5, 0.7)


def _plink(f, dur=0.5, amp=1.0, decay=9.0):
    t = _tt(dur)
    s = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * decay * 0.5) +
         0.2 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * decay)) * np.exp(-t * decay)
    return s * np.minimum(1, t * 1500) * amp


_PENT = [0, 2, 4, 7, 9, 12]


def _count(n):
    """n point = n plink (sikke ginne jaisa), aakhri plink lamba."""
    f0 = 523.25
    out = np.zeros(int(SR * (0.075 * n + 1.1)))
    for i in range(n):
        f = f0 * 2 ** (_PENT[i] / 12.0)
        last = i == n - 1
        _place(out, _plink(f, 1.0 if last else 0.35, 1.0 if last else 0.7, 5.5 if last else 11), 0.075 * i)
        _place(out, _tick(100 + i) * 0.35, 0.075 * i)
    f = f0 * 2 ** (_PENT[n - 1] / 12.0)
    _place(out, _plink(f * 2, 0.9, 0.35, 6), 0.075 * (n - 1))
    return _nrm(out, 0.62)


def _six():
    r = np.random.default_rng(6)
    out = np.zeros(int(SR * 1.8))
    n = int(SR * 0.45)
    t = np.arange(n) / SR
    riser = _ff(r.uniform(-1, 1, n), 1500, 9000) * (t / 0.45) ** 2
    _place(out, riser / (np.max(np.abs(riser)) + 1e-9) * 0.5, 0.0)
    for f in (1046.5, 1318.5, 1568.0):
        _place(out, _plink(f, 1.3, 0.5, 4.0), 0.45)
    for _ in range(16):
        _place(out, _plink(float(r.uniform(1800, 4200)), 0.3, 0.25, 18), 0.45 + float(r.uniform(0, 0.95)))
    return _nrm(out, 0.75)


def _one():
    out = np.zeros(int(SR * 0.9))
    for i, f in enumerate((392.0, 293.66)):
        t = _tt(0.5)
        s = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 30)) * np.exp(-t * 7)
        _place(out, s * np.minimum(1, t * 1500), 0.16 * i)
    return _nrm(out, 0.5)


def _lead():
    out = np.zeros(int(SR * 1.4))
    _place(out, _whoosh(0.45), 0.0, 0.7)
    for k, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
        _place(out, _plink(f, 1.0, 0.6, 5.0), 0.3 + 0.03 * k)
    return _nrm(out, 0.7)


def _tie():
    out = np.zeros(int(SR * 1.2))
    for i in range(2):
        _place(out, _plink(880.0, 0.8, 0.8, 6.0), 0.22 * i)
        _place(out, _plink(587.33, 0.8, 0.5, 6.0), 0.22 * i)
    return _nrm(out, 0.6)


def _roar(seed, dur=2.2):
    """Bheed ka shor (cheer)."""
    r = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = _ff(r.uniform(-1, 1, n), 350, 3800)
    x = x + 0.5 * _ff(r.uniform(-1, 1, n), 700, 1300)
    mod = _ff(r.uniform(-1, 1, n), None, 8)
    mod = mod / (np.max(np.abs(mod)) + 1e-9)
    env = np.minimum(1, t / 0.5) ** 1.5 * np.minimum(1, (dur - t) / 0.9) * (0.8 + 0.2 * mod)
    return _nrm(x * env, 0.7)


def _applause(dur=4.5, seed=2):
    """Taaliyan."""
    r = np.random.default_rng(seed)
    out = np.zeros(int(SR * (dur + 0.3)))
    claps = []
    for _ in range(8):
        m = int(SR * 0.03)
        t = np.arange(m) / SR
        nz = _ff(r.uniform(-1, 1, m), float(r.uniform(600, 1200)), float(r.uniform(3500, 7000))) * \
            np.exp(-t * float(r.uniform(120, 220)))
        claps.append(_nrm(nz, 1.0))
    for ti in np.arange(0, dur, 0.005):
        d = 160 * min(1.0, ti / 0.5) * min(1.0, (dur - ti) / 1.5)
        for _ in range(int(r.poisson(d * 0.005))):
            _place(out, claps[int(r.integers(8))], ti + float(r.uniform(0, 0.005)), float(r.uniform(0.1, 0.7)))
    return _nrm(out, 0.7)


def _pop(seed):
    """Party popper / confetti."""
    r = np.random.default_rng(seed)
    n = int(SR * 0.35)
    t = np.arange(n) / SR
    bang = _ff(r.uniform(-1, 1, n), 200, 5000) * np.exp(-t * 60)
    thump = np.sin(2 * np.pi * np.cumsum(180 * np.exp(-t * 30) + 60) / SR) * np.exp(-t * 25) * 0.8
    rustle = _ff(r.uniform(-1, 1, n), 3000, 9000) * np.exp(-t * 9) * 0.25 * (t > 0.03)
    return _nrm((bang + thump + rustle) * np.minimum(1, t * 3000), 0.85)


def build_ambient(dur=12.0):
    """Halki bheed/mahaul ki awaaz (seamless loop) - hamesha peeche chalti hai."""
    r = np.random.default_rng(5)
    n = int(SR * dur)
    X = np.fft.rfft(r.uniform(-1, 1, n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    X = X / np.sqrt(np.maximum(f, 40.0) / 40.0)
    x = _ff(np.fft.irfft(X, n), 150, 2600)
    mod = _ff(r.uniform(-1, 1, n), None, 5)
    mod = mod / (np.max(np.abs(mod)) + 1e-9)
    x = x * (0.75 + 0.25 * mod)
    x = x / np.sqrt(np.mean(x ** 2)) * 0.10
    return x.astype(np.float32)


def build_sfx():
    lib = {}
    lib["hit"] = [_knock(s) for s in (11, 12, 13, 14, 15, 16)]
    lib["click"] = [_tick(s) for s in (21, 22, 23, 24, 25)]
    lib["rattle"] = [_rattle(s) for s in (31, 32, 33)]
    lib["roll"] = [_roll(s) for s in (41, 42)]
    lib["settle"] = [_knock(51, 0.16, 0.5) * 0.7]
    lib["whoosh"] = _whoosh()
    for i in range(6):
        lib["ding%d" % (i + 1)] = _count(i + 1)
    lib["six"] = _six()
    lib["one"] = _one()
    lib["lead"] = _lead()
    lib["tie"] = _tie()
    lib["roar"] = [_roar(1), _roar(2, 2.6)]
    lib["applause"] = _applause()
    lib["pop"] = [_pop(s) for s in (61, 62, 63)]

    def note(m, dur):
        n = int(SR * dur)
        t = np.arange(n) / SR
        f = mtof(m)
        s = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 7))
        return s * np.minimum(1, t * 120) * np.minimum(1, (dur - t) / 0.2) * 0.12

    rng = np.random.default_rng(7)
    total = int(SR * 2.8)
    fan = np.zeros(total)
    for i, m in enumerate((72, 76, 79, 84)):
        s = note(m, 0.22)
        p = int(i * 0.14 * SR)
        fan[p:p + len(s)] += s
    chord_start = int(0.56 * SR)
    for m in (72, 76, 79, 84, 88):
        s = note(m, 1.9) * (1 + 0.15 * np.sin(2 * np.pi * 6 * np.arange(int(SR * 1.9)) / SR))
        fan[chord_start:chord_start + len(s)] += s * 0.8
    for _ in range(14):
        p = int(rng.uniform(0.5, 2.2) * SR)
        s = _plink(float(rng.choice([1046.5, 1318.5, 1568.0, 2093.0])), 0.4, 0.4, 8.0)
        k = min(len(s), total - p)
        if k > 0:
            fan[p:p + k] += s[:k]
    fan *= 0.85 / (np.max(np.abs(fan)) + 1e-9)
    lib["fanfare"] = fan.astype(np.float32)
    return lib


class Mixer:
    def __init__(self, music, sfx, amb=None):
        self.music = music
        self.sfx = sfx
        self.amb = amb if amb is not None else np.zeros(SR, dtype=np.float32)
        self.mp = 0
        self.ap = 0
        self.act = []          # [arr, pos, gain, is_voice]
        self.duck = 1.0
        self.music_vol = 0.45
        self.amb_gain = 0.2
        self.sfx_gain = 0.85

    def trigger_sfx(self, name, gain=1.0, delay=0.0):
        a = self.sfx.get(name)
        if a is None:
            return
        if isinstance(a, list):
            a = random.choice(a)
        self.act.append([a, -int(max(0.0, delay) * SR), gain * self.sfx_gain, False])

    def trigger_voice(self, arr, delay=0.0):
        self.act.append([arr, -int(max(0.0, delay) * SR), 1.0, True])

    @staticmethod
    def _loop(arr, p, n):
        L = len(arr)
        if p + n <= L:
            return arr[p:p + n], (p + n) % L
        return np.concatenate((arr[p:], arr[:p + n - L])), (p + n) % L

    def read(self, n):
        m, self.mp = self._loop(self.music, self.mp, n)
        a, self.ap = self._loop(self.amb, self.ap, n)
        out = np.zeros(n, dtype=np.float32)
        voice = False
        keep = []
        for act in self.act:
            arr, pos, g, isv = act
            if pos < 0:
                off = -pos
                if off >= n:
                    act[1] = pos + n
                    keep.append(act)
                    continue
                k = min(n - off, len(arr))
                out[off:off + k] += arr[:k] * g
                act[1] = k
            else:
                k = min(n, len(arr) - pos)
                if k > 0:
                    out[:k] += arr[pos:pos + k] * g
                    act[1] = pos + k
            if act[1] < len(arr):
                keep.append(act)
                if isv and act[1] >= 0:
                    voice = True
        self.act = keep
        target = 0.30 if voice else 1.0
        d0 = self.duck
        self.duck = d0 + (target - d0) * 0.25
        mg = np.linspace(d0, self.duck, n, dtype=np.float32)
        x = m * mg * self.music_vol + a * mg * self.amb_gain + out
        y = np.tanh(x * 1.15)
        return (y * 31000).astype("<i2").tobytes()


# ----------------------------------------------------------- Hindi awaaz
class VoiceBank:
    """Text -> awaaz (numpy). Pehle gTTS (internet), nahi to espeak. Background mein banta hai."""

    def __init__(self):
        self.cache = {}
        self.fail_t = {}
        self.inflight = set()
        self.q = queue.Queue()
        self.last_error = ""
        self.engine = self._detect()
        self.synth_hook = None  # test ke liye
        if self.engine:
            threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _detect():
        if not shutil.which("ffmpeg"):
            return None
        try:
            import gtts  # noqa
            return "gTTS"
        except Exception:
            pass
        for b in ("espeak-ng", "espeak"):
            if shutil.which(b):
                return b
        return None

    def status(self):
        if not self.engine:
            return "band (pip install gTTS ya pkg install espeak)"
        s = self.engine
        if self.last_error:
            s += " - dikkat: " + self.last_error[:80]
        return s

    def ask(self, text):
        if not self.engine or not text:
            return
        if self.cache.get(text) is False and time.time() - self.fail_t.get(text, 0) > 20:
            del self.cache[text]
        if text in self.cache or text in self.inflight:
            return
        self.inflight.add(text)
        self.q.put(text)

    def get(self, text):
        """array = ready, None = abhi ban raha hai, False = ban nahi paya"""
        if not self.engine:
            return False
        return self.cache.get(text)

    def _worker(self):
        while True:
            text = self.q.get()
            try:
                arr = self._make(text)
                self.cache[text] = arr
                self.last_error = ""
            except Exception as e:  # noqa
                self.cache[text] = False
                self.fail_t[text] = time.time()
                self.last_error = str(e)
            finally:
                self.inflight.discard(text)

    def _make(self, text):
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        pcm = os.path.join(TTS_DIR, key + ".pcm")
        if os.path.exists(pcm):
            raw = open(pcm, "rb").read()
        else:
            if self.synth_hook:
                src = self.synth_hook(text)
            else:
                src = self._synth(text, key)
            af = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                  "silenceremove=start_periods=1:start_threshold=-45dB,areverse,atempo=1.08")
            r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-af", af,
                                "-f", "s16le", "-ar", str(SR), "-ac", "1", "-"],
                               capture_output=True, timeout=40)
            raw = r.stdout
            if len(raw) < SR // 4:
                raise RuntimeError("awaaz khali aayi")
            with open(pcm, "wb") as f:
                f.write(raw)
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(arr))) or 1.0
        return arr * (0.92 / peak)

    def _synth(self, text, key):
        errs = []
        if self.engine == "gTTS":
            try:
                from gtts import gTTS
                p = os.path.join(TTS_DIR, key + ".mp3")
                gTTS(text, lang="hi").save(p)
                return p
            except Exception as e:  # noqa
                errs.append("gTTS: %s" % e)
        for b in ("espeak-ng", "espeak"):
            if shutil.which(b):
                p = os.path.join(TTS_DIR, key + ".wav")
                r = subprocess.run([b, "-v", "hi", "-s", "150", "-w", p, text], capture_output=True, timeout=30)
                if r.returncode == 0 and os.path.exists(p):
                    return p
                errs.append(b)
        raise RuntimeError("; ".join(errs) or "koi awaaz engine nahi")


# =================================================================== fonts
RED = (214, 36, 36)
BLUE = (34, 104, 232)
COLORS = {"red": RED, "blue": BLUE}
GLOW = {"red": (255, 82, 70), "blue": (62, 138, 255)}
GOLD = (255, 214, 64)

# Hindi naam chahiye to data/font.ttf mein koi Devanagari font rakh do
FONT_PATHS = [
    os.path.join(DATA, "font.ttf"),
    "/system/fonts/Roboto-Bold.ttf",
    "/system/fonts/DroidSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
_FONTS = {}


def font(size):
    size = max(8, int(size))
    if size in _FONTS:
        return _FONTS[size]
    f = None
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:
                pass
    if f is None:
        try:
            f = ImageFont.load_default(size)
        except TypeError:
            f = ImageFont.load_default()
    _FONTS[size] = f
    return f


def text_c(d, cx, cy, s, size, fill, stroke=0, stroke_fill=None):
    f = font(size)
    l, t, r, b = d.textbbox((0, 0), s, font=f, stroke_width=stroke)
    d.text((cx - (r - l) / 2 - l, cy - (b - t) / 2 - t), s, font=f, fill=fill,
           stroke_width=stroke, stroke_fill=stroke_fill)


def fit_size(s, maxw, size):
    while size > 14:
        l, t, r, b = font(size).getbbox(s)
        if r - l <= maxw:
            break
        size -= 2
    return size


# ================================================================ 3D dice
def mm(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def mv(M, v):
    return (M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
            M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
            M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def rot_axis(ax, a):
    x, y, z = ax
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    return ((c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C))


IDENT = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
FACES = {
    1: ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    6: ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    2: ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    5: ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    3: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    4: ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
}
R_FRONT = {1: IDENT, 6: rot_y(math.pi), 2: rot_x(math.pi / 2), 5: rot_x(-math.pi / 2),
           3: rot_y(-math.pi / 2), 4: rot_y(math.pi / 2)}
R_TILT = mm(rot_x(0.34), rot_y(-0.44))
PIPS = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
    5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
    6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
}
_L = (-0.45, 0.65, 0.62)
_ln = math.sqrt(sum(c * c for c in _L))
LIGHT = tuple(c / _ln for c in _L)


def _outline(r=0.13, seg=4):
    pts = []
    for sx, sy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for i in range(seg + 1):
            ang = math.radians(a0 + 90.0 * i / seg)
            pts.append((sx * (1 - r) + r * math.cos(ang), sy * (1 - r) + r * math.sin(ang)))
    return pts


FACE_OUTLINE = _outline()
CIRCLE = [(math.cos(2 * math.pi * i / 14), math.sin(2 * math.pi * i / 14)) for i in range(14)]
CORNERS = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]


def _hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def dice_sprite(M, color, size, ss=2):
    C = int(size * 1.7)
    S = C * ss
    im = Image.new("RGBA", (S, S), color + (0,))
    d = ImageDraw.Draw(im)
    unit = S / 4.3
    cx = cy = S / 2.0
    cam = 10.0

    def proj(p):
        x, y, z = mv(M, p)
        f = cam / (cam - z)
        return (cx + x * unit * f, cy - y * unit * f)

    body = tuple(int(c * 0.5) for c in color) + (255,)
    d.polygon(_hull([proj((a * 0.95, b * 0.95, c * 0.95)) for a, b, c in CORNERS]), fill=body)
    for val, (n, u, v) in FACES.items():
        nr = mv(M, n)
        if nr[2] <= 0.03:
            continue
        lam = max(0.0, nr[0] * LIGHT[0] + nr[1] * LIGHT[1] + nr[2] * LIGHT[2])
        shade = 0.52 + 0.48 * lam
        spec = int(80 * lam ** 12)
        fc = tuple(min(255, int(c * shade) + spec) for c in color) + (255,)
        ec = tuple(int(c * 0.42 * shade) for c in color) + (255,)
        pts = [proj((n[0] + a * u[0] + b * v[0], n[1] + a * u[1] + b * v[1], n[2] + a * u[2] + b * v[2]))
               for a, b in FACE_OUTLINE]
        d.polygon(pts, fill=fc, outline=ec)
        ps = 0.72 + 0.28 * lam
        pc = (int(250 * ps), int(250 * ps), int(250 * ps), 255)
        for gx, gy in PIPS[val]:
            ox, oy = gx * 0.56, gy * 0.56
            for rad, col in ((0.215, ec), (0.175, pc)):
                pp = [proj((n[0] + (ox + rad * ca) * u[0] + (oy + rad * sa) * v[0],
                            n[1] + (ox + rad * ca) * u[1] + (oy + rad * sa) * v[1],
                            n[2] + (ox + rad * ca) * u[2] + (oy + rad * sa) * v[2]))
                      for ca, sa in CIRCLE]
                d.polygon(pp, fill=col)
    return im.reduce(ss) if ss > 1 else im


_REST = {}


def rest_sprite(value, side, size):
    key = (value, side, size)
    sp = _REST.get(key)
    if sp is None:
        sp = dice_sprite(mm(R_TILT, R_FRONT[value]), COLORS[side], size, 2)
        _REST[key] = sp
    return sp


# uchhal (pixel, 720 chauda screen ke hisaab se)
H0 = 215.0
T0 = 0.40
E = 0.58


def bounce_h(t):
    if t < T0:
        return H0 * (1 - (t / T0) ** 2)
    t -= T0
    for n in range(1, 9):
        A = 2 * T0 * (E ** n)
        Hn = H0 * (E ** (2 * n))
        if t < A:
            s = t / A
            return Hn * (1 - (2 * s - 1) ** 2)
        t -= A
    return 0.0


def impact_times(limit):
    out = []
    t = T0
    n = 0
    while t < limit:
        out.append((t, E ** n))
        n += 1
        t += 2 * T0 * (E ** n)
        if n > 8:
            break
    return out


def circle_photo(path, diameter, color, ring):
    D = diameter
    big = D * 3
    src = None
    if path and os.path.exists(path):
        try:
            src = ImageOps.fit(Image.open(path).convert("RGB"), (big, big), Image.LANCZOS)
        except Exception:
            src = None
    if src is None:
        src = Image.new("RGB", (big, big), tuple(int(c * 0.45) for c in color))
        text_c(ImageDraw.Draw(src), big / 2, big / 2, "?", int(big * 0.5), (255, 255, 255))
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, big - 1, big - 1), fill=255)
    out = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    out.paste(src, (0, 0), mask)
    od = ImageDraw.Draw(out)
    od.ellipse((0, 0, big - 1, big - 1), outline=color, width=ring * 3)
    od.ellipse((ring * 3, ring * 3, big - 1 - ring * 3, big - 1 - ring * 3), outline=(255, 255, 255, 120), width=3)
    return out.resize((D, D), Image.LANCZOS)


# ================================================================== scene
class Scene:
    ROLL = 2.4
    SPIN_END = 2.0
    SHOW = 1.3
    MAXSHOW = 7.5

    def __init__(self, cfg, bank=None):
        self.bank = bank
        self.W = int(cfg["width"])
        self.H = int(cfg["height"])
        self.fps = int(cfg["fps"])
        self.k = self.W / 720.0
        self.ss = 2
        self.events = collections.deque()
        self.cmds = collections.deque()
        self.pending_cfg = None
        self.voice_on = bool(cfg.get("voice_on", True))
        self._tc = {}
        self.cfg = cfg
        self._read_cfg(cfg)
        self.static = self._build_static()
        self.glow = {s: self._build_glow(s) for s in ("red", "blue")}
        self.last_img = None
        self.final_bg = None
        self.clock = 0.0
        self.voice_end = 0.0
        self.vpend = []
        self.want_final = False
        self.final_text = ""
        self.frac = 0.5
        self.last = {"red": random.randint(1, 6), "blue": random.randint(1, 6)}
        self._load_state()
        self.turn = "red"
        self.t = 0.0
        self.winner = None
        if self.final:
            self.phase = "final"
            self.winner = self._winner()
        else:
            self.phase = "roll"
            self._start_roll()

    # ---------------------------------------------------------- helpers
    def S(self, v):
        return int(round(v * self.k))

    def _read_cfg(self, cfg):
        self.cfg = cfg
        self.names = {"red": cfg["red_name"], "blue": cfg["blue_name"]}
        self.say_names = {"red": say_name(cfg, "red"), "blue": say_name(cfg, "blue")}

    def _load_state(self):
        self.score = {"red": 0, "blue": 0}
        self.round = 0
        self.final = False
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                st = json.load(f)
            self.score = {"red": max(0, int(st.get("red", 0))), "blue": max(0, int(st.get("blue", 0)))}
            self.round = max(0, int(st.get("round", 0)))
            self.final = bool(st.get("final", False))
            if self.final:   # pichhla match khatam ho chuka tha: server dobara chale to naya game
                self.score = {"red": 0, "blue": 0}
                self.round = 0
                self.final = False
        except Exception:
            pass

    def _save_state(self):
        try:
            tmp = STATE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"red": self.score["red"], "blue": self.score["blue"],
                           "round": self.round, "final": self.phase == "final"}, f)
            os.replace(tmp, STATE_PATH)
        except Exception:
            pass

    def sfx(self, name, gain=1.0, delay=0.0):
        self.events.append(("sfx", name, gain, delay))

    def put_text(self, img, cx, cy, s, size, fill, stroke=0, sfill=None, alpha=255):
        key = (s, size, fill, stroke, sfill)
        sp = self._tc.get(key)
        if sp is None:
            f = font(size)
            tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
            l, t, r, b = tmp.textbbox((0, 0), s, font=f, stroke_width=stroke)
            sp = Image.new("RGBA", (r - l + 6, b - t + 6), (0, 0, 0, 0))
            ImageDraw.Draw(sp).text((3 - l, 3 - t), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=sfill)
            if len(self._tc) > 500:
                self._tc.clear()
            self._tc[key] = sp
        if alpha < 255:
            a = sp.getchannel("A").point(lambda v: v * alpha // 255)
            sp = sp.copy()
            sp.putalpha(a)
        img.paste(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)), sp)

    def set_quality(self, low):
        self.ss = 1 if low else 2

    # --------------------------------------------------------- liquid glass
    def _rr_mask(self, w, h, r, ss=3):
        m = Image.new("L", (w * ss, h * ss), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
        return m.resize((w, h), Image.LANCZOS)

    def glass(self, base, box, r, tint=(255, 255, 255), ta=0.10, blur=20):
        S = self.S
        x0, y0, x1, y1 = [int(v) for v in box]
        w, h = x1 - x0, y1 - y0
        bl = base.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(max(1, S(blur))))
        bl = ImageEnhance.Color(bl).enhance(1.6)
        bl = ImageEnhance.Brightness(bl).enhance(1.12)
        bl = Image.blend(bl, Image.new("RGB", (w, h), tint), ta)
        mask = self._rr_mask(w, h, r)
        # upar ki chamak (liquid sheen)
        vg = ImageOps.invert(Image.linear_gradient("L")).resize((w, max(1, int(h * 0.5))))
        sheen = Image.new("L", (w, h), 0)
        sheen.paste(vg.point(lambda v: int(v * 0.22)), (0, 0))
        bl.paste((255, 255, 255), (0, 0, w, h), sheen)
        # kinara: upar-left chamakdar, neeche-right halka
        ring = Image.new("L", (w * 3, h * 3), 0)
        ImageDraw.Draw(ring).rounded_rectangle((0, 0, w * 3 - 1, h * 3 - 1), r * 3, outline=255,
                                               width=max(1, S(2)) * 3)
        ring = ring.resize((w, h), Image.LANCZOS)
        vgrad = ImageOps.invert(Image.linear_gradient("L")).resize((w, h))
        hgrad = ImageOps.invert(Image.linear_gradient("L").rotate(90, expand=True)).resize((w, h))
        eg = Image.blend(vgrad, hgrad, 0.5).point(lambda v: 55 + int(v * 0.78))
        edge = ImageChops.multiply(ring, eg).point(lambda v: int(v * 0.85))
        bl.paste((255, 255, 255), (0, 0, w, h), edge)
        base.paste(bl, (x0, y0), mask)

    def _build_glow(self, side):
        S = self.S
        col = GLOW[side]
        m = S(46)
        pw, ph = S(324), S(950)
        w, h = pw + 2 * m, ph + 2 * m
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle((m, m, m + pw, m + ph), S(28), outline=col + (255,), width=S(9))
        layer = layer.filter(ImageFilter.GaussianBlur(S(12)))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle((m, m, m + pw, m + ph), S(28), outline=(255, 255, 255, 235), width=max(2, S(3)))
        d.rounded_rectangle((m, m, m + pw, m + ph), S(28), fill=col + (30,))
        out = []
        for f in (0.55, 0.65, 0.75, 0.85, 0.95, 1.0):
            c = layer.copy()
            c.putalpha(layer.getchannel("A").point(lambda v, f=f: int(v * f)))
            out.append(c)
        return out

    def _build_static(self):
        S, W, H = self.S, self.W, self.H
        grad = Image.linear_gradient("L").resize((W, H))
        img = Image.composite(Image.new("RGB", (W, H), (7, 22, 46)), Image.new("RGB", (W, H), (20, 14, 44)), grad)
        blobs = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blobs)
        for cx, cy, r, col in ((140, 300, 340, (255, 60, 70, 210)), (590, 380, 340, (50, 120, 255, 210)),
                               (360, 860, 400, (140, 70, 255, 150)), (90, 1180, 280, (0, 200, 190, 120)),
                               (640, 1150, 260, (255, 150, 60, 110))):
            bd.ellipse((S(cx - r), S(cy - r), S(cx + r), S(cy + r)), fill=col)
        blobs = blobs.filter(ImageFilter.GaussianBlur(S(95)))
        img.paste(blobs, (0, 0), blobs)
        # title
        self.glass(img, (S(70), S(26), S(650), S(114)), S(44), tint=(255, 255, 255), ta=0.08, blur=14)
        d = ImageDraw.Draw(img, "RGBA")
        title = self.cfg["title"]
        text_c(d, W / 2, S(70), title, fit_size(title, S(500), S(58)), (255, 255, 255),
               stroke=S(2), stroke_fill=(10, 10, 30))
        for side, x0 in (("red", 24), ("blue", 372)):
            col = GLOW[side]
            self.glass(img, (S(x0), S(175), S(x0 + 324), S(1125)), S(34),
                       tint=col, ta=0.10, blur=24)
            self.glass(img, (S(x0 + 34), S(588), S(x0 + 290), S(772)), S(30), tint=col, ta=0.06, blur=10)
            d = ImageDraw.Draw(img, "RGBA")
            cx = S(x0 + 162)
            photo = circle_photo(os.path.join(DATA, side + ".png"), S(240), col, S(9))
            img.paste(photo, (int(cx - photo.width / 2), int(S(330) - photo.height / 2)), photo)
            nm = self.names[side]
            text_c(d, cx, S(515), nm, fit_size(nm, S(280), S(52)), (255, 255, 255),
                   stroke=S(2), stroke_fill=(10, 10, 30))
            text_c(d, cx, S(618), "POINTS", S(24), (225, 230, 245))
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse((W / 2 - S(40), S(330) - S(40), W / 2 + S(40), S(330) + S(40)),
                  fill=(15, 15, 35, 235), outline=GOLD + (255,), width=S(3))
        text_c(d, W / 2, S(330), "VS", S(34), GOLD)
        self.glass(img, (S(24), S(1140), S(696), S(1184)), S(22), ta=0.06, blur=10)
        return img

    # --------------------------------------------------------------- game
    def _winner(self):
        r, b = self.score["red"], self.score["blue"]
        return None if r == b else ("red" if r > b else "blue")

    def _start_roll(self):
        self.result = random.randint(1, 6)
        self.axis1 = (random.uniform(-1, 1), random.uniform(0.3, 1), random.uniform(-1, 1))
        self.axis2 = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.3, 1))
        self.spin1 = random.choice((-1, 1)) * random.uniform(5.5, 7.5) * math.pi
        self.spin2 = random.choice((-1, 1)) * random.uniform(3.0, 5.0) * math.pi
        self.dir = random.choice((-1, 1))
        side = self.turn
        total = self.score[side] + self.result
        extra = ""
        if side == "blue":
            r, b = self.score["red"], total
            if r > b:
                extra = "अब %s %s पॉइंट से आगे है।" % (self.say_names["red"], hn(r - b))
            elif b > r:
                extra = "अब %s %s पॉइंट से आगे है।" % (self.say_names["blue"], hn(b - r))
            else:
                extra = "दोनों बराबर हैं।"
        self.say_text = roll_text(self.say_names[side], self.result, total, extra)
        if self.bank and self.voice_on:
            self.bank.ask(self.say_text)
        # awaaz ka kram: haath mein khadkhadana -> hawa mein whoosh -> table par takrana ->
        # ludhakna -> ruk kar halka 'tok'
        self.sfx("rattle", 0.75, 0.0)
        self.sfx("whoosh", 0.5, 0.05)
        imps = impact_times(self.ROLL - 0.2)
        for tt, st in imps:
            self.sfx("hit", 0.95 * st ** 0.5, tt)
        if len(imps) > 3:
            self.sfx("roll", 0.7, imps[3][0] - 0.05)
        self.sfx("settle", 0.6, self.SPIN_END + 0.05)

    def say(self, text, wait=1.8):
        if self.bank and self.voice_on and text:
            self.bank.ask(text)
            self.vpend.append((text, self.clock + wait))

    def _voice_poll(self):
        for item in list(self.vpend):
            text, deadline = item
            arr = self.bank.get(text) if self.bank else False
            if arr is None:
                if self.clock > deadline:
                    self.vpend.remove(item)
            elif arr is False:
                self.vpend.remove(item)
            else:
                self.events.append(("voice", arr))
                self.voice_end = max(self.voice_end, self.clock + len(arr) / SR + 0.35)
                self.vpend.remove(item)

    def _land(self):
        side = self.turn
        self.phase = "show"
        self.t = 0.0
        self.last[side] = self.result
        prev_lead, prev_total = self._winner(), self.score["red"] + self.score["blue"]
        self.score[side] += self.result
        if side == "blue":
            self.round += 1
        self._save_state()
        n = self.result
        self.sfx("ding%d" % n, 0.9)               # n point = n plink (sikke ginne jaisa)
        if n == 6:
            self.sfx("six", 0.9, 0.05)
            self.sfx("roar", 0.45, 0.45)
        elif n == 1:
            self.sfx("one", 0.8, 0.05)
        lead = self._winner()
        if prev_total > 0 and lead != prev_lead:  # aage nikalna ya barabari
            self.sfx("tie" if lead is None else "lead", 0.85, 0.5)
        self.say(self.say_text)

    def _enter_final(self):
        self.phase = "final"
        self.t = 0.0
        self.want_final = False
        self.winner = self._winner()
        r, b = self.score["red"], self.score["blue"]
        if self.winner:
            sc = self.score[self.winner]
            self.final_text = "%s जीत गया! कुल %s पॉइंट। बधाई हो!" % (self.say_names[self.winner], hn(sc))
        else:
            self.final_text = "मुक़ाबला बराबर रहा! दोनों के %s पॉइंट।" % hn(r)
        self.final_bg = None
        self.sfx("fanfare", 0.95)
        self.sfx("applause", 0.8, 0.4)
        if self.winner:
            self.sfx("roar", 0.8, 0.3)
        for tm in (0.25, 0.65, 1.1, 1.6):
            self.sfx("pop", 0.7, tm)
        self.say(self.final_text, wait=3.0)
        self._save_state()

    def _cmd(self, c):
        if c == "final":
            if self.phase != "final":
                self.want_final = True
        elif c == "new":
            self.score = {"red": 0, "blue": 0}
            self.round = 0
            self.want_final = False
            self.final_bg = None
            self.winner = None
            self.turn = "red"
            self.phase = "roll"
            self.t = 0.0
            self.vpend = []
            self.voice_end = 0.0
            self.frac = 0.5
            self.sfx("ding3", 0.7)
            self._start_roll()
            self._save_state()

    def step(self, dt):
        self.clock += dt
        if self.pending_cfg:
            cfg, self.pending_cfg = self.pending_cfg, None
            self._read_cfg(cfg)
            self.voice_on = bool(cfg.get("voice_on", True))
            self.static = self._build_static()
            self.final_bg = None
        while self.cmds:
            self._cmd(self.cmds.popleft())
        self._voice_poll()
        self.t += dt
        tot = self.score["red"] + self.score["blue"]
        target = 0.5 if tot == 0 else self.score["red"] / float(tot)
        self.frac += (target - self.frac) * min(1.0, dt * 4.0)
        if self.phase == "roll":
            if self.t >= self.ROLL:
                self._land()
        elif self.phase == "show":
            ready = self.t >= self.SHOW and self.clock >= self.voice_end and not self.vpend
            if ready or self.t >= self.MAXSHOW:
                if self.want_final and self.turn == "blue":
                    self._enter_final()
                else:
                    self.turn = "blue" if self.turn == "red" else "red"
                    self.phase = "roll"
                    self.t = 0.0
                    self._start_roll()

    # --------------------------------------------------------------- frame
    def render(self):
        if self.phase == "final":
            img = self._render_final()
        else:
            img = self._render_game()
        self.last_img = img
        return img

    def _render_game(self):
        S, W, H = self.S, self.W, self.H
        img = self.static.copy()
        d = ImageDraw.Draw(img, "RGBA")
        dsize = S(210)
        floor_y = S(930) + int(dsize * 0.52)
        self.put_text(img, W / 2, S(150), "ROUND %d" % (self.round + 1), S(28), GOLD)
        pulse = 0.5 + 0.5 * math.sin(self.clock * 5.0)
        for side, x0 in (("red", 24), ("blue", 372)):
            cx = S(x0 + 162)
            active = self.turn == side
            if active:
                g = self.glow[side][min(5, int(pulse * 6))]
                img.paste(g, (S(x0) - S(46), S(175) - S(46)), g)
            sc = 1.0
            if active and self.phase == "show" and self.t < 0.45:
                sc = 1.0 + 0.28 * math.sin(self.t / 0.45 * math.pi)
            self.put_text(img, cx, S(690), str(self.score[side]), int(S(118) * sc), (255, 255, 255),
                          stroke=S(4), sfill=GLOW[side])
        order = sorted(("red", "blue"), key=lambda s: 1 if (self.turn == s and self.phase == "roll") else 0)
        for side in order:
            x0 = 24 if side == "red" else 372
            cx = S(x0 + 162)
            height = 0.0
            dx = 0.0
            if self.turn == side and self.phase == "roll":
                t = self.t
                height = bounce_h(t) * self.k
                dx = self.dir * S(80) * (1 - min(1.0, t / self.ROLL)) ** 2
                f = max(0.0, 1.0 - t / self.SPIN_END)
                M = mm(R_TILT, mm(rot_axis(self.axis2, self.spin2 * f ** 2.2),
                                   mm(rot_axis(self.axis1, self.spin1 * f ** 2), R_FRONT[self.result])))
                sprite = dice_sprite(M, COLORS[side], dsize, self.ss)
            elif self.turn == side and self.phase == "show":
                sprite = rest_sprite(self.result, side, dsize)
                sc = 1.0 + 0.09 * math.sin(min(1.0, self.t / 0.45) * math.pi)
                if sc > 1.004:
                    sprite = sprite.resize((int(sprite.width * sc), int(sprite.height * sc)), Image.BILINEAR)
            else:
                sprite = rest_sprite(self.last[side], side, dsize)
            sh = 1.0 / (1.0 + height / (140.0 * self.k))
            sw = S(118) * (0.6 + 0.4 * sh)
            d.ellipse((cx + dx - sw, floor_y - S(15) * sh, cx + dx + sw, floor_y + S(15) * sh),
                      fill=(0, 0, 0, int(115 * sh)))
            img.paste(sprite, (int(cx + dx - sprite.width / 2), int(S(930) - height - sprite.height / 2)), sprite)
            if self.turn == side and self.phase == "show":
                a = int(255 * max(0.0, 1 - self.t / 1.6))
                if a > 4:
                    self.put_text(img, cx, S(1090) - self.t * 16 * self.k, "+%d" % self.result, S(70),
                                  GOLD, stroke=S(3), sfill=(0, 0, 0), alpha=a)
        # khichav-patti (tug bar)
        x0, x1, y0, y1 = S(36), S(684), S(1150), S(1174)
        split = x0 + (x1 - x0) * min(0.96, max(0.04, self.frac))
        rr = S(11)
        d.rounded_rectangle((x0, y0, split - 2, y1), rr, fill=GLOW["red"] + (255,))
        d.rounded_rectangle((split + 2, y0, x1, y1), rr, fill=GLOW["blue"] + (255,))
        d.rounded_rectangle((x0 + 3, y0 + 2, split - 5, y0 + S(8)), S(4), fill=(255, 255, 255, 70))
        d.rounded_rectangle((split + 5, y0 + 2, x1 - 3, y0 + S(8)), S(4), fill=(255, 255, 255, 70))
        d.ellipse((split - S(9), (y0 + y1) / 2 - S(9), split + S(9), (y0 + y1) / 2 + S(9)),
                  fill=(255, 255, 255, 255))
        r, b = self.score["red"], self.score["blue"]
        if r == b:
            lead, lc = "BARABARI", (235, 235, 245)
        elif r > b:
            nm = self.names["red"]
            lead, lc = "%s aage hai  +%d" % (nm, r - b), (255, 140, 130)
        else:
            nm = self.names["blue"]
            lead, lc = "%s aage hai  +%d" % (nm, b - r), (130, 180, 255)
        self.put_text(img, W / 2, S(1212), lead, fit_size(lead, S(600), S(32)), lc)
        nm = self.names[self.turn]
        self.put_text(img, W / 2, S(1250), "%s ki baari" % nm, fit_size(nm, S(450), S(30)), (255, 255, 255))
        return img

    def _render_final(self):
        S, W, H = self.S, self.W, self.H
        if self.final_bg is None:
            snap = (self.last_img or self.static).copy()
            snap = ImageEnhance.Brightness(snap).enhance(0.35)
            self.glass(snap, (S(40), S(300), S(680), S(980)), S(46), tint=(255, 255, 255), ta=0.08, blur=26)
            self.final_bg = snap
        img = self.final_bg.copy()
        d = ImageDraw.Draw(img, "RGBA")
        col = GLOW[self.winner] if self.winner else GOLD
        for i in range(60):
            x = (i * 97 + math.sin(self.t * 2 + i) * 18) % W
            y = (i * 53 + self.t * 260 * self.k) % H
            c = GOLD + (255,) if i % 3 == 0 else col + (255,)
            d.rectangle((x, y, x + S(11), y + S(18)), fill=c)
        pop = 1.0 + 0.06 * math.sin(self.t * 4)
        self.put_text(img, W / 2, S(400), "WINNER" if self.winner else "DRAW", int(S(112) * pop), GOLD,
                      stroke=S(4), sfill=(0, 0, 0))
        if self.winner:
            nm = self.names[self.winner]
            self.put_text(img, W / 2, S(560), nm, fit_size(nm, S(560), S(100)), (255, 255, 255),
                          stroke=S(4), sfill=col)
        else:
            self.put_text(img, W / 2, S(560), "BARABARI", S(90), (255, 255, 255), stroke=S(4), sfill=col)
        self.put_text(img, W / 2, S(720), "%d  -  %d" % (self.score["red"], self.score["blue"]), S(86), (255, 255, 255))
        self.put_text(img, W / 2, S(820), "%s  vs  %s" % (self.names["red"], self.names["blue"]),
                      fit_size("%s  vs  %s" % (self.names["red"], self.names["blue"]), S(560), S(34)),
                      (215, 220, 240))
        self.put_text(img, W / 2, S(900), "Total points", S(30), GOLD)
        return img


# ============================================================== streaming
def stream_url(cfg):
    return "rtmp://a.rtmp.youtube.com/live2/" + cfg["stream_key"]


def ffmpeg_cmd(cfg, url, audio_fd):
    fps = int(cfg["fps"])
    br = str(cfg["bitrate"])
    return [
        "ffmpeg", "-loglevel", "warning",
        "-probesize", "32", "-analyzeduration", "0", "-thread_queue_size", "512",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (cfg["width"], cfg["height"]),
        "-r", str(fps), "-i", "-",
        "-probesize", "32", "-analyzeduration", "0", "-thread_queue_size", "512",
        "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:%d" % audio_fd,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", cfg.get("preset", "veryfast"), "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-bf", "0",
        "-g", str(fps * 2), "-keyint_min", str(fps * 2), "-sc_threshold", "0",
        "-b:v", br, "-maxrate", br, "-bufsize", str(int(br[:-1]) * 2) + "k",
        "-c:a", "aac", "-b:a", "128k", "-ar", str(SR), "-ac", "2",
        "-f", "flv", url,
    ]


class _Pump:
    """Awaaz ko alag thread se ffmpeg ke audio pipe mein bharta hai, taki video aur awaaz
    ek doosre ko rok (deadlock) na saken. Pipe bhar jaye to sabse purana tukda hata deta hai."""

    def __init__(self, f):
        self.f = f
        self.q = queue.Queue(maxsize=600)
        self.dead = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def put(self, b):
        if self.dead:
            return
        try:
            self.q.put_nowait(b)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(b)
            except Exception:
                pass

    def _run(self):
        try:
            while True:
                b = self.q.get()
                if b is None:
                    break
                self.f.write(b)
                self.f.flush()
        except Exception:
            pass
        finally:
            self.dead = True
            try:
                self.f.close()
            except Exception:
                pass

    def close(self):
        try:
            self.q.put_nowait(None)
        except Exception:
            pass


class AudioHub:
    """Preview mein admin page ko live awaaz sunane ke liye (browser <audio> /preview.wav se)."""

    def __init__(self):
        self.qs = []
        self.ring = collections.deque(maxlen=24)
        self.lock = threading.Lock()

    @property
    def n(self):
        return len(self.qs)

    def add(self):
        q = queue.Queue(maxsize=300)
        with self.lock:
            for b in list(self.ring):
                q.put_nowait(b)
            self.qs.append(q)
        return q

    def remove(self, q):
        with self.lock:
            if q in self.qs:
                self.qs.remove(q)

    def publish(self, b):
        self.ring.append(b)
        with self.lock:
            for q in self.qs:
                try:
                    q.put_nowait(b)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(b)
                    except Exception:
                        pass


class Streamer:
    def __init__(self):
        self.thread = None
        self.stop_evt = threading.Event()
        self.running = False
        self.live = False
        self.error = ""
        self.preview = None
        self.scene = None
        self.bank = None
        self.music = None
        self.sfx = None
        self.mixer = None
        self.audio = {"music": 0.45, "voice": True, "sfx": 0.85}
        self.amb = None
        self.hub = AudioHub()
        self.perf = {"render_ms": 0.0, "budget_ms": 0.0, "dup": 0, "drop": 0, "reconnects": 0, "low": False}
        self._lock = threading.Lock()

    def _init_audio(self):
        if self.music is None:
            self.music = make_music()
            self.sfx = build_sfx()
            self.amb = build_ambient()
        if self.bank is None:
            self.bank = VoiceBank()

    def start(self, go_live):
        with self._lock:
            if self.running:
                return "Pehle se chal raha hai, pehle STOP karo."
            cfg = load_cfg()
            if go_live:
                if not cfg["stream_key"]:
                    return "Stream key save nahi hai."
                if not shutil.which("ffmpeg"):
                    return "ffmpeg install nahi hai."
            self.error = ""
            self.stop_evt.clear()
            self.running = True
            self.live = go_live
            self.thread = threading.Thread(target=self._run, args=(cfg, go_live), daemon=True)
            self.thread.start()
            return ""

    def stop(self):
        self.stop_evt.set()

    def game(self, action):
        if self.scene is None or not self.running:
            return "Pehle Preview ya GO LIVE chalao."
        self.scene.cmds.append(action)
        return ""

    def set_audio(self, music=None, voice=None, sfx=None):
        if sfx is not None:
            self.audio["sfx"] = max(0.0, min(1.0, float(sfx)))
        if music is not None:
            self.audio["music"] = max(0.0, min(1.0, float(music)))
        if voice is not None:
            self.audio["voice"] = bool(voice)
            if self.scene:
                self.scene.voice_on = bool(voice)

    def apply_cfg(self, cfg):
        if self.scene:
            self.scene.pending_cfg = cfg

    # ---- writer: ffmpeg ko frames+awaaz dena, toote to dobara jodna
    def _writer(self, cfg, go_live, wq):
        proc = None
        aw = None
        log = None
        retry_at = 0.0
        backoff = 3.0
        ok = 0
        fps = int(cfg["fps"])

        def close():
            nonlocal proc, aw, log
            if proc is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if aw is not None:
                aw.close()       # pump thread khud pipe band karega (ffmpeg ke jaane par EPIPE se)
            if log:
                try:
                    log.close()
                except Exception:
                    pass
            proc = aw = log = None

        try:
            while True:
                item = wq.get()
                if item is None:
                    break
                if not go_live:
                    continue
                raw, aud = item
                now = time.monotonic()
                if proc is not None and proc.poll() is not None:
                    close()
                    retry_at = now + backoff
                    backoff = min(15.0, backoff * 1.6)
                    self.error = "Connection toota, dobara jud rahe hain (key/net check karo)"
                if proc is None:
                    if now < retry_at:
                        continue
                    ar, w = os.pipe()
                    log = open(LOG_PATH, "wb")
                    try:
                        proc = subprocess.Popen(ffmpeg_cmd(cfg, stream_url(cfg), ar), stdin=subprocess.PIPE,
                                                stdout=log, stderr=log, pass_fds=(ar,))
                    finally:
                        os.close(ar)
                    aw = _Pump(os.fdopen(w, "wb"))
                    ok = 0
                    self.perf["reconnects"] += 1
                try:
                    aw.put(aud)
                    proc.stdin.write(raw)
                    if aw.dead:
                        raise BrokenPipeError("audio pipe band")
                    ok += 1
                    if ok == fps * 4:
                        self.error = ""
                        backoff = 3.0
                except (BrokenPipeError, OSError, ValueError):
                    close()
                    retry_at = time.monotonic() + backoff
                    backoff = min(15.0, backoff * 1.6)
                    self.error = "Connection toota, dobara jud rahe hain (key/net check karo)"
        finally:
            close()

    # ---- main loop: ek hi ghadi (clock), ek-saman raftaar
    def _run(self, cfg, go_live):
        """Nigrani: koi bhi galti aaye to 3 sec baad apne aap dobara shuru (score state.json mein
        surakshit hai). Sirf admin ke STOP dabane par hi band hota hai."""
        try:
            first = True
            while not self.stop_evt.is_set():
                if not first:
                    cfg = load_cfg()
                    self.error = (self.error or "Error") + " - 3 sec mein dobara shuru"
                    if self.stop_evt.wait(3.0):
                        break
                first = False
                self._run_once(cfg, go_live)
        finally:
            self.running = False
            self.live = False

    def _run_once(self, cfg, go_live):
        wq = queue.Queue(maxsize=10)
        wt = None
        try:
            self._init_audio()
            self.audio["music"] = max(0.0, min(1.0, float(cfg.get("music_vol", 45)) / 100.0))
            self.audio["voice"] = bool(cfg.get("voice_on", True))
            scene = Scene(cfg, self.bank)
            scene.voice_on = self.audio["voice"]
            self.scene = scene
            self.audio["sfx"] = max(0.0, min(1.0, float(cfg.get("sfx_vol", 85)) / 100.0))
            mixer = Mixer(self.music, self.sfx, self.amb)
            self.mixer = mixer
            fps = scene.fps
            fdt = 1.0 / fps
            wt = threading.Thread(target=self._writer, args=(cfg, go_live, wq), daemon=True)
            wt.start()
            self.perf.update(render_ms=0.0, budget_ms=fdt * 1000, dup=0, drop=0, reconnects=0, low=False)
            start = time.monotonic()
            written = 0
            ra = 0.0
            slow_since = None
            last_prev = 0.0
            while not self.stop_evt.is_set():
                target = start + written * fdt
                delay = target - time.monotonic()
                if delay > 0.0015:
                    time.sleep(delay - 0.001)
                    continue
                t0 = time.monotonic()
                n = int((t0 - start) * fps) - written + 1
                if n < 1:
                    n = 1
                if n > fps * 2:           # lamba atak gaya (phone so gaya) - ghadi dobara milao
                    start = t0 - written * fdt
                    n = 1
                if n > 1:
                    self.perf["dup"] += n - 1
                scene.step(n * fdt)
                mixer.music_vol = self.audio["music"] * 1.25
                mixer.amb_gain = self.audio["music"] * 0.5      # peeche ki halki bheed/mahaul
                mixer.sfx_gain = self.audio["sfx"]
                while scene.events:
                    ev = scene.events.popleft()
                    if ev[0] == "sfx":
                        mixer.trigger_sfx(ev[1], ev[2], ev[3])
                    elif ev[0] == "voice" and self.audio["voice"]:
                        mixer.trigger_voice(ev[1])
                img = scene.render()
                raw = img.tobytes()
                for i in range(n):
                    s0 = int(round((written + i) * SR / fps))
                    s1 = int(round((written + i + 1) * SR / fps))
                    aud = mixer.read(s1 - s0)
                    self.hub.publish(aud)
                    try:
                        wq.put((raw, aud), timeout=1.0)
                    except queue.Full:
                        self.perf["drop"] += 1
                written += n
                rt = time.monotonic() - t0
                ra = rt if ra == 0.0 else ra * 0.92 + rt * 0.08
                self.perf["render_ms"] = round(ra * 1000, 1)
                if ra > fdt * 0.85:
                    slow_since = slow_since or t0
                    if t0 - slow_since > 3.0 and not self.perf["low"]:
                        self.perf["low"] = True
                        scene.set_quality(True)
                else:
                    slow_since = None
                if self.error.startswith("Error") and t0 - start > 6.0:
                    self.error = ""      # dobara theek chal raha hai
                if t0 - last_prev > 0.4:
                    last_prev = t0
                    buf = io.BytesIO()
                    img.resize((max(1, img.width // 3), max(1, img.height // 3))).save(buf, "JPEG", quality=65)
                    self.preview = buf.getvalue()
        except Exception as e:  # noqa
            self.error = "Error: %s" % e
        finally:
            try:
                wq.put(None, timeout=2)
            except Exception:
                pass
            if wt is not None:
                wt.join(timeout=8)


streamer = Streamer()

# ===================================================================== web
app = Flask(__name__)


@app.before_request
def _auth():
    pw = load_cfg().get("admin_pass") or ""
    if pw:
        a = request.authorization
        if not a or a.password != pw:
            return Response("Password chahiye", 401, {"WWW-Authenticate": 'Basic realm="Dice Admin"'})


PAGE = """<!doctype html><html lang="hi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dice Battle - Admin</title>
<style>
body{margin:0;background:#0f1020;color:#eee;font-family:system-ui,sans-serif}
.wrap{max-width:560px;margin:auto;padding:16px}
h1{font-size:22px;margin:4px 0 0}
label{display:block;margin:12px 0 4px;font-size:14px;color:#aab}
input[type=text],input[type=password],input[type=number],select{width:100%;box-sizing:border-box;padding:10px;border-radius:10px;border:1px solid #334;background:#1a1c33;color:#fff;font-size:16px}
input[type=file]{color:#aab;font-size:14px;max-width:100%}
input[type=range]{width:100%}
.row{display:flex;gap:10px}.row>div{flex:1}
button{padding:12px 16px;border:0;border-radius:10px;font-size:16px;font-weight:600;color:#fff;cursor:pointer}
.save{background:#5b5bf0;width:100%;margin-top:16px}
.live{background:#e53935}.test{background:#2e7d32}.stop{background:#555}
.fin{background:#f39c12;color:#111}.new{background:#8e44ad}
.card{background:#171932;border-radius:14px;padding:14px;margin-top:16px}
.red{color:#ff6b5e}.blue{color:#5aa2ff}.hint{font-size:12px;color:#889;margin-top:4px}
img.prev{width:100%;max-width:270px;display:none;margin:10px auto;border-radius:10px;background:#000}
img.th{width:48px;height:48px;border-radius:50%;object-fit:cover;vertical-align:middle;margin-right:8px}
.ctl{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.score{font-size:30px;font-weight:800;text-align:center;margin:8px 0 0}
.chk{display:flex;align-items:center;gap:8px;margin:12px 0 0;color:#cdd}
.chk input{width:20px;height:20px}
</style></head><body><div class="wrap">
<h1>🎲 Dice Battle - Admin Panel</h1>
<div class="card">
<div class="ctl" style="margin-top:0">
<button class="test" onclick="act('/api/start','preview')">Preview</button>
<button class="live" onclick="if(confirm('YouTube par LIVE shuru karein?'))act('/api/start','live')">GO LIVE</button>
<button class="stop" onclick="act('/api/stop')">STOP</button>
</div>
<p id="st" style="margin:10px 0 0">...</p>
<div class="score"><span class="red" id="sr">0</span> - <span class="blue" id="sb">0</span></div>
<div class="hint" id="info"></div>
<img id="pv" class="prev" alt="preview">
<div class="ctl">
<button class="fin" onclick="if(confirm('Round poora hote hi FINAL RESULT/Winner dikhaayein?'))act('/api/game','final')">🏆 FINAL RESULT</button>
<button class="new" onclick="if(confirm('Sab points 0 karke naya game shuru karein?'))act('/api/game','new')">🔄 NAYA GAME</button>
</div>
<label>🎵 Music awaaz: <span id="mv">{{cfg.music_vol}}</span>%</label>
<input type="range" min="0" max="100" value="{{cfg.music_vol}}" oninput="mv.textContent=this.value" onchange="aud()" id="mus">
<label>🔔 Sound effects awaaz: <span id="sv">{{cfg.sfx_vol}}</span>%</label>
<input type="range" min="0" max="100" value="{{cfg.sfx_vol}}" oninput="sv.textContent=this.value" onchange="aud()" id="sfx">
<div class="chk"><input type="checkbox" id="vc" {% if cfg.voice_on %}checked{% endif %} onchange="aud()"><span>🗣 Hindi awaaz mein points bolo</span></div>
<div class="ctl"><button class="test" id="pab" onclick="pa()">🔊 Preview awaaz suno</button></div>
<audio id="pa" preload="none"></audio>
<div class="hint">Ye sirf sound-check ke liye hai. Live stream ki awaaz YouTube par apne aap jati hai, phone ke speaker se nahi.</div>
<div class="hint" id="vs"></div>
</div>
<form class="card" method="post" action="/save" enctype="multipart/form-data">
<label>YouTube Stream Key {% if has_key %}(saved ✔ — khali chhodo to wahi rahegi){% endif %}</label>
<input type="password" name="stream_key" placeholder="xxxx-xxxx-xxxx-xxxx-xxxx" autocomplete="off">
<div class="hint">YouTube Studio → Go Live → Stream → Stream key</div>
<label>Title (video ke upar)</label><input type="text" name="title" value="{{cfg.title}}" maxlength="30">
<div class="row">
<div><label class="red">Red ka naam</label><input type="text" name="red_name" value="{{cfg.red_name}}" maxlength="20"></div>
<div><label class="blue">Blue ka naam</label><input type="text" name="blue_name" value="{{cfg.blue_name}}" maxlength="20"></div>
</div>
<div class="row">
<div><label class="red">Red awaaz mein (Hindi)</label><input type="text" name="red_say" value="{{cfg.red_say}}" placeholder="जैसे: रमन" maxlength="30"></div>
<div><label class="blue">Blue awaaz mein (Hindi)</label><input type="text" name="blue_say" value="{{cfg.blue_say}}" placeholder="जैसे: राहुल" maxlength="30"></div>
</div>
<div class="hint">Khali chhodo to naam wahi bola jayega (RED = रेड, BLUE = ब्लू).</div>
<label>Quality (lag ho to Lite lo)</label>
<select name="quality">
<option value="lite" {% if cfg.quality=='lite' %}selected{% endif %}>Lite — 480x854, 24fps (sabse smooth)</option>
<option value="low" {% if cfg.quality=='low' %}selected{% endif %}>Normal — 540x960, 30fps</option>
<option value="hd" {% if cfg.quality=='hd' %}selected{% endif %}>HD — 720x1280, 30fps (tez phone chahiye)</option>
</select>
<label class="red">{% if has_red %}<img class="th" src="/photo/red?{{ts}}">{% endif %}Red photo (left)</label>
<input type="file" name="red_photo" accept="image/*">
<label class="blue">{% if has_blue %}<img class="th" src="/photo/blue?{{ts}}">{% endif %}Blue photo (right)</label>
<input type="file" name="blue_photo" accept="image/*">
<div class="chk"><input type="checkbox" name="auto_live" {% if cfg.auto_live %}checked{% endif %}><span>Server chalu hote hi apne aap LIVE ho jao</span></div>
<label>Admin password (khali = koi password nahi)</label>
<input type="password" name="admin_pass" placeholder="naya password" autocomplete="new-password">
{% if has_pass %}<div class="chk"><input type="checkbox" name="clear_pass"><span>Purana password hata do</span></div>{% endif %}
<button class="save" type="submit">Save settings</button>
<div class="hint">Naam, photo, title turant live video mein badal jate hain. Key/quality ke liye STOP → GO LIVE karo.</div>
</form></div>
<script>
const st=document.getElementById('st'),pv=document.getElementById('pv');
async function poll(){try{const s=await (await fetch('/api/status')).json();
let t=s.running?(s.live?'🔴 LIVE chal raha hai':'👀 Preview chal raha hai'):'⏹ Band hai';
if(s.error)t+=' — '+s.error;st.textContent=t;
document.getElementById('sr').textContent=s.red;document.getElementById('sb').textContent=s.blue;
let i='Round '+s.round;
if(s.phase=='final')i+=' · FINAL result screen par hai';else if(s.want_final)i+=' · Round poora hote hi FINAL aayega';
if(s.running)i+=' · render '+s.perf.render_ms+'ms / '+s.perf.budget_ms.toFixed(0)+'ms'+(s.perf.low?' (auto-halka)':'')+' · dup '+s.perf.dup+' · reconnect '+Math.max(0,s.perf.reconnects-1);
document.getElementById('info').textContent=i;
document.getElementById('vs').textContent='Awaaz engine: '+s.voice;
if(s.running){pv.style.display='block';pv.src='/preview.jpg?'+Date.now();}else{pv.style.display='none';}}catch(e){}}
function act(u,mode){const f=new FormData();if(mode)f.append(u=='/api/game'?'action':'mode',mode);
fetch(u,{method:'POST',body:f}).then(r=>r.json()).then(j=>{if(j.error)alert(j.error);setTimeout(poll,600);});}
function pa(){const a=document.getElementById('pa'),b=document.getElementById('pab');
if(a.paused){a.src='/preview.wav?'+Date.now();a.play().catch(()=>{});b.textContent='⏹ Awaaz band karo';}
else{a.pause();a.removeAttribute('src');a.load();b.textContent='🔊 Preview awaaz suno';}}
setInterval(()=>{if(pv.style.display=='block')pv.src='/preview.jpg?'+Date.now();},450);
function aud(){const f=new FormData();f.append('music',document.getElementById('mus').value);f.append('sfx',document.getElementById('sfx').value);
f.append('voice',document.getElementById('vc').checked?'1':'0');fetch('/api/audio',{method:'POST',body:f});}
setInterval(poll,1500);poll();
</script></body></html>"""


@app.route("/")
@app.route("/admin")
def index():
    cfg = load_cfg()
    return render_template_string(
        PAGE, cfg=cfg, has_key=bool(cfg["stream_key"]), has_pass=bool(cfg["admin_pass"]), ts=int(time.time()),
        has_red=os.path.exists(os.path.join(DATA, "red.png")),
        has_blue=os.path.exists(os.path.join(DATA, "blue.png")))


@app.post("/save")
def save():
    cfg = load_cfg()
    key = request.form.get("stream_key", "").strip()
    if key and re.fullmatch(r"[A-Za-z0-9_.\-]{8,64}", key):
        cfg["stream_key"] = key
    cfg["title"] = request.form.get("title", "").strip()[:30] or DEFAULTS["title"]
    cfg["red_name"] = request.form.get("red_name", "").strip()[:20] or DEFAULTS["red_name"]
    cfg["blue_name"] = request.form.get("blue_name", "").strip()[:20] or DEFAULTS["blue_name"]
    cfg["red_say"] = request.form.get("red_say", "").strip()[:30]
    cfg["blue_say"] = request.form.get("blue_say", "").strip()[:30]
    q = request.form.get("quality", "low")
    cfg["quality"] = q if q in QUALITY else "low"
    cfg["auto_live"] = request.form.get("auto_live") is not None
    pw = request.form.get("admin_pass", "")
    if request.form.get("clear_pass") is not None:
        cfg["admin_pass"] = ""
    elif pw:
        cfg["admin_pass"] = pw[:64]
    for side in ("red", "blue"):
        f = request.files.get(side + "_photo")
        if f and f.filename:
            try:
                im = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
                im.thumbnail((800, 800))
                im.save(os.path.join(DATA, side + ".png"))
            except Exception:
                pass
    save_cfg(cfg)
    streamer.apply_cfg(load_cfg())
    return redirect("/")


@app.post("/api/start")
def api_start():
    return jsonify(error=streamer.start(request.form.get("mode") == "live"))


@app.post("/api/stop")
def api_stop():
    streamer.stop()
    return jsonify(error="")


@app.post("/api/game")
def api_game():
    a = request.form.get("action")
    if a not in ("final", "new"):
        return jsonify(error="galat action")
    return jsonify(error=streamer.game(a))


@app.post("/api/audio")
def api_audio():
    m = request.form.get("music")
    v = request.form.get("voice")
    sx = request.form.get("sfx")
    cfg = load_cfg()
    try:
        if m is not None:
            cfg["music_vol"] = max(0, min(100, int(m)))
        if sx is not None:
            cfg["sfx_vol"] = max(0, min(100, int(sx)))
        if v is not None:
            cfg["voice_on"] = v == "1"
        save_cfg(cfg)
        streamer.set_audio(music=cfg["music_vol"] / 100.0, voice=cfg["voice_on"], sfx=cfg["sfx_vol"] / 100.0)
    except ValueError:
        pass
    return jsonify(error="")


@app.get("/api/status")
def api_status():
    sc = streamer.scene
    bank = streamer.bank
    return jsonify(
        running=streamer.running, live=streamer.live, error=streamer.error,
        red=sc.score["red"] if sc else 0, blue=sc.score["blue"] if sc else 0,
        round=sc.round if sc else 0, phase=sc.phase if sc else "", want_final=bool(sc and sc.want_final),
        perf=streamer.perf, voice=bank.status() if bank else "abhi shuru nahi hua")


@app.get("/preview.wav")
def preview_wav():
    """Preview/Live chalte waqt wahi awaaz browser mein sunne ke liye (sound check)."""
    if not streamer.running:
        abort(404)
    q = streamer.hub.add()

    def gen():
        try:
            yield (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " +
                   struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) + b"data" + struct.pack("<I", 0xFFFFFFFF))
            while streamer.running:
                try:
                    yield q.get(timeout=2)
                except queue.Empty:
                    continue
        finally:
            streamer.hub.remove(q)
    return Response(gen(), mimetype="audio/wav", headers={"Cache-Control": "no-store"})


@app.get("/preview.jpg")
def preview():
    if not streamer.preview:
        abort(404)
    return send_file(io.BytesIO(streamer.preview), mimetype="image/jpeg")


@app.get("/photo/<side>")
def photo(side):
    if side not in ("red", "blue"):
        abort(404)
    p = os.path.join(DATA, side + ".png")
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="image/png")


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    c = load_cfg()
    if c.get("auto_live") and c.get("stream_key"):
        threading.Timer(2.0, lambda: print("auto live:", streamer.start(True) or "ok")).start()
    print("Admin panel: http://%s:%d" % (host, port))
    app.run(host=host, port=port, threaded=True)
