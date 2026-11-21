#!/usr/bin/env python3
"""Çöp Adam Can ile Para Yönetimi - ~13 dk'lık finans eğitim videosu üreticisi.
Kullanım: python3 make_video.py [çıktı.mp4]
Gereksinimler: Python 3, Pillow, numpy, ffmpeg, DejaVu Sans fontu.
"""
import math, os, subprocess, sys, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, S, FPS = 1280, 720, 2, 12
GY = 505  # zemin çizgisi
FD = "/usr/share/fonts/truetype/dejavu/"
INK = (32, 34, 48); WHITE = (255, 255, 255)
BLUE = (52, 120, 246); ORANGE = (255, 150, 20); GREEN = (40, 180, 110)
RED = (230, 64, 70); GOLD = (250, 200, 40); PURPLE = (140, 90, 220)
GRAY = (150, 155, 170); LGRAY = (225, 228, 236); BROWN = (150, 100, 60)
D = None
_fc = {}


def font(sz, b=False):
    k = (sz, b)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FD + ("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf"), int(sz * S))
    return _fc[k]


def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(x): x = clamp(x); return x * x * (3 - 2 * x)
def prog(t, delay, dur): return ease((t - delay) / dur)


def C(x, y, r, f=None, o=None, w=0):
    D.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=f, outline=o, width=int(w * S))


def L(x1, y1, x2, y2, w=4, c=INK):
    D.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=c, width=max(1, int(w * S)))
    if w >= 3:
        C(x1, y1, w / 2, c); C(x2, y2, w / 2, c)


def R(x1, y1, x2, y2, f=None, o=None, w=0, r=0):
    if r:
        D.rounded_rectangle([x1 * S, y1 * S, x2 * S, y2 * S], radius=r * S, fill=f, outline=o, width=int(w * S))
    else:
        D.rectangle([x1 * S, y1 * S, x2 * S, y2 * S], fill=f, outline=o, width=int(w * S))


def POLY(pts, f=None, o=None, w=0):
    D.polygon([(x * S, y * S) for x, y in pts], fill=f)
    if o and w:
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            L(a[0], a[1], b[0], b[1], w, o)


def T(x, y, s, sz=24, c=INK, b=False, a='mm'):
    D.text((x * S, y * S), s, font=font(sz, b), fill=c, anchor=a)


def PIE(cx, cy, r, a0, a1, f, o=None):
    D.pieslice([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], a0, a1, fill=f, outline=o, width=int(2 * S))


def ARC(x1, y1, x2, y2, a0, a1, w=3, c=INK):
    D.arc([x1 * S, y1 * S, x2 * S, y2 * S], a0, a1, fill=c, width=int(w * S))


# ---------------------------------------------------------------- çöp adam
def pose(k, t):
    s = math.sin
    base = dict(la=(8, 6), ra=(-8, -6), ll=(5, 0), rl=(-5, 0), lean=0, bob=0)
    if k == 'idle':
        base.update(la=(10 + 5 * s(t * 2), 8), ra=(-10 - 5 * s(t * 2 + 1), -8), bob=1.5 * abs(s(t * 2)))
    elif k in ('walk', 'run'):
        ph = t * (5 if k == 'walk' else 9); a = 35 if k == 'walk' else 50
        base.update(ll=(a * s(ph), -30 * max(0, -s(ph))), rl=(-a * s(ph), -30 * max(0, s(ph))),
                    la=(-a * 0.8 * s(ph), 18), ra=(a * 0.8 * s(ph), 18),
                    bob=3 * abs(s(ph)), lean=4 if k == 'walk' else 12)
    elif k == 'wave':
        base.update(ra=(150, 25 * s(t * 7)), la=(8, 6), bob=1.2 * abs(s(t * 2)))
    elif k == 'point':
        base.update(la=(80, 4), ra=(-8, -6), bob=1.2 * abs(s(t * 2)))
    elif k == 'cheer':
        base.update(la=(160 + 10 * s(t * 8), 0), ra=(-160 - 10 * s(t * 8 + 1), 0), bob=7 * abs(s(t * 6)))
    elif k == 'sad':
        base.update(la=(4, 8), ra=(-4, -8), lean=-7, bob=0)
    elif k == 'think':
        base.update(la=(35, 115), ra=(-8, -6), bob=1.5 * abs(s(t * 2)))
    elif k == 'hold':
        base.update(la=(55, 40), ra=(40, 50), bob=1.2 * abs(s(t * 2)))
    elif k == 'umb':
        base.update(ra=(100, 75), la=(8, 6), lean=-3)
    elif k == 'push':
        base.update(la=(75, 5), ra=(60, 10), lean=14, ll=(35, 0), rl=(-25, 0))
    elif k == 'shrug':
        base.update(la=(55, -75), ra=(-55, 75), bob=1.5 * abs(s(t * 3)))
    return base


def seg(x, y, a, l, fl):
    return x + math.sin(math.radians(a)) * l * fl, y + math.cos(math.radians(a)) * l


def man(x, gy, pk='idle', t=0.0, col=INK, sc=1.0, mood='smile', flip=1, acc=None, name=None, lw=5):
    p = pose(pk, t)
    hip = (x, gy - 96 * sc - p['bob'] * sc)
    neck = (x + p['lean'] * flip * sc, hip[1] - 68 * sc)
    sh = (neck[0], neck[1] + 8 * sc)
    head = (neck[0] + p['lean'] * 0.4 * flip * sc, neck[1] - 22 * sc)
    w = lw * sc
    out = {}

    def limb(start, spec, l1, l2, key=None):
        k = seg(start[0], start[1], spec[0], l1 * sc, flip)
        e = seg(k[0], k[1], spec[0] + spec[1], l2 * sc, flip)
        L(start[0], start[1], k[0], k[1], w, col); L(k[0], k[1], e[0], e[1], w, col)
        if key: out[key] = e
    limb(hip, p['rl'], 48, 48); limb(sh, p['ra'], 38, 38, 'hr')
    L(hip[0], hip[1], neck[0], neck[1], w, col)
    limb(hip, p['ll'], 48, 48); limb(sh, p['la'], 38, 38, 'hl')
    r = 21 * sc
    if acc == 'tie':
        POLY([(sh[0] - 4 * sc, sh[1] + 2 * sc), (sh[0] + 4 * sc, sh[1] + 2 * sc), (sh[0], sh[1] + 26 * sc)], RED)
    C(head[0], head[1], r, WHITE, col, w * 0.8)
    if acc == 'cap':
        D.pieslice([(head[0] - r) * S, (head[1] - r) * S, (head[0] + r) * S, (head[1] + r) * S], 180, 360, fill=BLUE, outline=col, width=int(3 * S))
        L(head[0], head[1] - 4 * sc, head[0] + flip * (r + 9 * sc), head[1] - 4 * sc, 4 * sc, BLUE)
    if acc == 'glasses':
        for dx in (-7, 7):
            C(head[0] + flip * 3 * sc + dx * sc, head[1] - 3 * sc, 6 * sc, None, col, 2)
        L(head[0] + flip * 3 * sc - sc, head[1] - 3 * sc, head[0] + flip * 3 * sc + sc, head[1] - 3 * sc, 2, col)
    ex = head[0] + flip * 3 * sc
    if mood == 'sleep':
        for dx in (-7, 7):
            L(ex + dx * sc - 3 * sc, head[1] - 3 * sc, ex + dx * sc + 3 * sc, head[1] - 3 * sc, 2, col)
    else:
        for dx in (-7, 7):
            C(ex + dx * sc, head[1] - 3 * sc, 2.4 * sc, col)
    mx, my = ex, head[1] + 8 * sc
    if mood in ('smile', 'sleep'):
        ARC(mx - 8 * sc, my - 7 * sc, mx + 8 * sc, my + 5 * sc, 15, 165, 3, col)
    elif mood == 'sad':
        ARC(mx - 8 * sc, my - 1 * sc, mx + 8 * sc, my + 11 * sc, 195, 345, 3, col)
    elif mood == 'open':
        C(mx, my, 5 * sc, col)
    else:
        L(mx - 6 * sc, my, mx + 6 * sc, my, 3, col)
    if name:
        T(x, head[1] - r - 16 * sc, name, 20, col, True)
    out['head'] = head
    return out


# ---------------------------------------------------------------- nesneler
def coin(x, y, r=15, label='₺'):
    C(x, y, r, GOLD, (200, 150, 0), 2); T(x, y, label, int(r * 1.15), (140, 95, 0), True)


def bag(x, y, s=1.0, label='', col=GREEN):
    C(x, y, 55 * s, col, INK, 3); R(x - 18 * s, y - 70 * s, x + 18 * s, y - 45 * s, col, INK, 3, 6)
    L(x - 22 * s, y - 70 * s, x + 22 * s, y - 70 * s, 4)
    T(x, y + 5 * s, label, int(21 * s), WHITE, True)


def piggy(x, y, s=1.0):
    C(x, y, 50 * s, (255, 170, 190), INK, 3)
    C(x + 48 * s, y + 5 * s, 20 * s, (255, 150, 175), INK, 3)
    POLY([(x + 20 * s, y - 40 * s), (x + 38 * s, y - 62 * s), (x + 44 * s, y - 32 * s)], (255, 150, 175), INK, 3)
    for dx in (-25, 20):
        R(x + dx * s, y + 40 * s, x + (dx + 14) * s, y + 62 * s, (255, 150, 175), INK, 3)
    R(x - 18 * s, y - 50 * s, x + 10 * s, y - 46 * s, INK)
    C(x + 54 * s, y - 8 * s, 3 * s, INK)


def cloud(x, y, s=1.0, c=(190, 200, 215)):
    for dx, dy, r in ((-40, 0, 30), (0, -15, 38), (42, 0, 30), (0, 8, 32)):
        C(x + dx * s, y + dy * s, r * s, c)


def shield(x, y, s=1.0, c=BLUE, label=''):
    POLY([(x - 55 * s, y - 60 * s), (x + 55 * s, y - 60 * s), (x + 55 * s, y + 10 * s), (x, y + 75 * s), (x - 55 * s, y + 10 * s)], c, INK, 4)
    if label: T(x, y - 5 * s, label, int(20 * s), WHITE, True)


def basket(x, y, w=110, label='', eggs=4, col=BROWN, ec=(250, 240, 215), lab_c=INK):
    for i in range(eggs):
        ex = x + (i - (eggs - 1) / 2) * 20 * (w / 110)
        D.ellipse([(ex - 11) * S, (y - 38 - (i % 2) * 8) * S, (ex + 11) * S, (y - 10 - (i % 2) * 8) * S], fill=ec, outline=INK, width=2 * S)
    POLY([(x - w / 2, y - 18), (x + w / 2, y - 18), (x + w / 2 - 14, y + 34), (x - w / 2 + 14, y + 34)], col, INK, 3)
    for k in range(1, 4):
        yy = y - 18 + k * 13
        L(x - w / 2 + 3 + k * 3, yy, x + w / 2 - 3 - k * 3, yy, 2, (110, 70, 35))
    if label:
        T(x, y + 58, label, 17, lab_c, True)


def check(x, y, s=1.0, c=GREEN):
    L(x - 14 * s, y, x - 4 * s, y + 12 * s, 6 * s, c); L(x - 4 * s, y + 12 * s, x + 16 * s, y - 12 * s, 6 * s, c)


def cross(x, y, s=1.0, c=RED):
    L(x - 12 * s, y - 12 * s, x + 12 * s, y + 12 * s, 6 * s, c); L(x - 12 * s, y + 12 * s, x + 12 * s, y - 12 * s, 6 * s, c)


def arrow(x1, y1, x2, y2, c=INK, w=4):
    L(x1, y1, x2, y2, w, c)
    a = math.atan2(y2 - y1, x2 - x1)
    for da in (2.6, -2.6):
        L(x2, y2, x2 + 14 * math.cos(a + da), y2 + 14 * math.sin(a + da), w, c)


def bubble(x, y, w, h, s, sz=22, c=WHITE, tc=INK):
    R(x - w / 2, y - h / 2, x + w / 2, y + h / 2, c, INK, 3, 16)
    T(x, y, s, sz, tc, True)


def card(x, y, w=200, h=125, col=PURPLE, label=''):
    R(x - w / 2, y - h / 2, x + w / 2, y + h / 2, col, INK, 3, 14)
    R(x - w / 2, y - h / 2 + 22, x + w / 2, y - h / 2 + 44, (40, 40, 60))
    R(x - w / 2 + 16, y + 2, x - w / 2 + 54, y + 28, GOLD, INK, 2, 5)
    if label: T(x + 15, y + 40, label, 18, WHITE, True)


def sbar(x, y, w, h, fracs, cols, reveal=1.0, labels=None):
    cx = x; shown = w * reveal; used = 0
    for i, f in enumerate(fracs):
        ww = min(w * f, max(0, shown - used))
        if ww > 0:
            R(cx, y, cx + ww, y + h, cols[i], INK, 2)
            if labels and ww > w * f - 1:
                T(cx + ww / 2, y + h / 2, labels[i], 20, WHITE, True)
        cx += w * f; used += w * f


# ---------------------------------------------------------------- senaryo
SCENES = []


def scene(title, num, bg, caps):
    def deco(fn):
        SCENES.append(dict(title=title, num=num, bg=bg, caps=caps, fn=fn)); return fn
    return deco


# ---------- GİRİŞ
@scene("Giriş", 0, (222, 238, 255), [
    "Merhaba! Ben Can. Bir çöp adamım ama paramla ilgili dev planlarım var.",
    "Bu videoda maaşını nasıl yöneteceğini ve parayı zamanla nasıl büyüteceğini, adım adım ve basit örneklerle göreceğiz.",
    "Maaş günü herkes mutlu. Peki ayın 20'sinde cüzdan neden hep boş?",
    "Sorun çoğu zaman az kazanmak değil; parayı bir plana bağlamamak.",
    "Bugün yedi kural göreceğiz: bütçe, acil durum fonu, bileşik getiri, enflasyon, borç yönetimi, çeşitlendirme ve davranış hataları.",
    "Küçük bir not: Bu video eğitim amaçlıdır. Örnekler varsayımsaldır ve yatırım tavsiyesi değildir.",
])
def s_intro(ci, ct, t):
    for i in range(7):
        coin(120 + i * 170 + 20 * math.sin(t * 1.5 + i), 220 + 12 * math.sin(t * 2 + i * 1.3), 14)
    if ci <= 1:
        T(W / 2, 90, "PARA YÖNETİMİ", 64, BLUE, True)
        T(W / 2, 150, "Çöp Adam Can ile 7 Altın Kural", 30, INK, True)
    if ci == 0:
        x = -80 + 520 * prog(ct, 0, 3.2)
        man(x, GY, 'walk' if ct < 3.4 else 'wave', t, acc='cap', name='CAN', sc=1.25)
    elif ci == 1:
        man(440, GY, 'wave', t, acc='cap', name='CAN', sc=1.25)
        bubble(760, 330, 300, 70, "Hazır mısın?", 28)
    elif ci == 2:
        man(300, GY, 'sad', t, acc='cap', mood='sad', sc=1.25)
        R(560, 260, 860, 450, (140, 100, 70), INK, 4, 18); R(560, 260, 860, 310, (120, 85, 60), INK, 4, 18)
        T(710, 395, "0 ₺", 54, WHITE, True)
        R(930, 250, 1130, 430, WHITE, INK, 4, 10); R(930, 250, 1130, 290, RED, INK, 4, 10)
        T(1030, 270, "AYIN 20'Sİ", 18, WHITE, True); T(1030, 355, "20", 80, INK, True)
    elif ci == 3:
        man(380, GY, 'think', t, acc='cap', mood='open', sc=1.25)
        T(640, 300, "?", int(110 + 12 * math.sin(t * 4)), ORANGE, True)
        bubble(900, 330, 380, 110, "Plan yok = Para yok", 30, (255, 240, 200))
    elif ci == 4:
        man(220, GY, 'point', t, acc='cap', sc=1.25)
        items = ["Bütçe", "Acil durum fonu", "Bileşik getiri", "Enflasyon", "Borç yönetimi", "Çeşitlendirme", "Davranış hataları"]
        n = int(clamp(ct / 8.0, 0, 1) * 7.99)
        for i in range(min(n + 1, 7)):
            R(520, 190 + i * 48 - 18, 600, 190 + i * 48 + 18, ORANGE if i % 2 else BLUE, None, 0, 12)
            T(560, 190 + i * 48, str(i + 1), 22, WHITE, True); T(625, 190 + i * 48, items[i], 26, INK, True, 'lm')
    else:
        man(400, GY, 'shrug', t, acc='cap', sc=1.25)
        R(640, 220, 1150, 400, (255, 245, 205), INK, 4, 18)
        T(895, 275, "EĞİTİM AMAÇLIDIR", 30, RED, True); T(895, 325, "Yatırım tavsiyesi değildir.", 28, INK, True)
        T(895, 365, "Örnekler varsayımsaldır.", 24, INK, False)


# ---------- KURAL 1: BÜTÇE
@scene("Paranın Nereye Gittiğini Bil", 1, (255, 247, 228), [
    "Kural 1: Paranın nereye gittiğini bil. Bilmediğin parayı yönetemezsin.",
    "Can'ın maaşı 20.000 lira. Ama ay sonunda cebinde hiçbir şey kalmıyor. Gizemli bir şekilde!",
    "Gizem şu: kahve, yemek siparişi, abonelikler... Tek tek masum görünür, ama toplamda dev bir rakam yaparlar.",
    "Meşhur bir yöntem var: 50-30-20 kuralı.",
    "Gelirinin yüzde 50'si ihtiyaçlara gitsin: kira, fatura, market, ulaşım. Yani 10.000 lira.",
    "Yüzde 30'u isteklere: dışarıda yemek, eğlence, hobiler. Yani 6.000 lira.",
    "Yüzde 20'si geleceğine: birikim ve yatırım. Yani 4.000 lira.",
    "Altın kural: Önce kendine öde. Maaş yatınca ilk iş birikimini ayır, kalanı harca. Ay sonunda artanı biriktirmeye çalışma.",
    "Bu oranlar herkese uymaz. Kira çok yüksekse 60-20-20 olabilir. Önemli olan, bir sisteminin olması.",
    "Bir ay boyunca her harcamanı yaz. İlk seferde şaşıracaksın. Şaşırmak iyidir; farkındalık buradan başlar.",
])
def s_budget(ci, ct, t):
    pk = {0: 'hold', 1: 'shrug', 2: 'think', 3: 'point', 4: 'point', 5: 'point', 6: 'point', 7: 'cheer', 8: 'idle', 9: 'point'}[ci]
    mood = 'open' if ci in (1, 2) else 'smile'
    man(220 if ci != 9 else 160, GY, pk, t, acc='tie', mood=mood, sc=1.25, name='CAN')
    if ci <= 1:
        bag(420, 400, 1.3, "20.000 ₺")
    if ci == 1:
        for i in range(9):
            q = ((ct * 0.35 + i / 9) % 1)
            coin(470 + q * 700, 380 - 90 * math.sin(q * math.pi), 14)
        T(1000, 200, "?", 120, ORANGE, True)
    if ci == 2:
        items = [(520, "Kahve", "30 × 150 ₺"), (760, "Sipariş", "20 × 400 ₺"), (1000, "Abonelik", "5 × 200 ₺")]
        for i, (x, n, v) in enumerate(items):
            if ct > i * 1.6:
                R(x - 80, 250, x + 80, 420, WHITE, INK, 3, 16)
                if i == 0:
                    R(x - 24, 290, x + 12, 340, WHITE, INK, 4, 6); ARC(x - 6, 300, x + 24, 330, -90, 90, 4)
                    for k in range(3): L(x - 14 + k * 12, 280, x - 10 + k * 12, 262, 3, GRAY)
                elif i == 1:
                    R(x - 20, 270, x + 20, 340, GRAY, INK, 3, 8); C(x, 330, 4, WHITE)
                else:
                    R(x - 34, 285, x + 34, 335, BLUE, INK, 3, 6); L(x - 10, 345, x + 10, 345, 3)
                T(x, 375, n, 22, INK, True); T(x, 402, v, 18, GRAY, True)
        if ct > 5: T(760, 200, "Toplam: ???", 40, RED, True)
    if 3 <= ci <= 7:
        rv = {3: 0, 4: 0.5 * prog(ct, 0, 1.6), 5: 0.5 + 0.3 * prog(ct, 0, 1.6), 6: 0.8 + 0.2 * prog(ct, 0, 1.6)}.get(ci, 1.0)
        cx, cy, r = 700, 285, 150
        C(cx, cy, r, WHITE, INK, 3)
        cum = 0
        for f, c in ((0.5, BLUE), (0.3, ORANGE), (0.2, GREEN)):
            a0 = -90 + 360 * cum; a1 = -90 + 360 * min(cum + f, rv)
            if a1 > a0 + 0.1: PIE(cx, cy, r, a0, a1, c, INK)
            cum += f
        if ci == 3: T(cx, cy, "50 / 30 / 20", 34, INK, True)
        leg = [("İHTİYAÇ  10.000 ₺", BLUE, 4), ("İSTEK  6.000 ₺", ORANGE, 5), ("BİRİKİM  4.000 ₺", GREEN, 6)]
        for i, (s, c, need) in enumerate(leg):
            if ci >= need:
                R(900, 215 + i * 60, 930, 245 + i * 60, c, INK, 2, 5); T(942, 230 + i * 60, s, 22, INK, True, 'lm')
    if ci == 7:
        piggy(470, 440, 1.0)
        q = (ct * 0.7) % 1; coin(480, 330 + q * 70, 14)
        T(470, 540, "Önce kendine öde!", 20, GREEN, True)
    if ci == 8:
        T(740, 180, "Kira yüksekse esnet:", 28, INK, True)
        sbar(420, 220, 740, 62, (0.5, 0.3, 0.2), (BLUE, ORANGE, GREEN), 1, ["%50", "%30", "%20"])
        sbar(420, 320, 740, 62, (0.6, 0.2, 0.2), (BLUE, ORANGE, GREEN), 1, ["%60", "%20", "%20"])
        T(420, 405, "İhtiyaç • İstek • Birikim", 22, GRAY, True, 'lm')
    if ci == 9:
        R(430, 120, 1160, 500, WHITE, INK, 4, 14)
        for k in range(8): L(450, 175 + k * 40, 1140, 175 + k * 40, 1, (190, 200, 230))
        T(795, 148, "HARCAMA DEFTERİ", 24, BLUE, True)
        rows = ["Pzt  Kahve ............ 85 ₺", "Pzt  Öğle yemeği ... 180 ₺", "Sal  Otobüs .......... 40 ₺", "Çar  Sipariş ......... 420 ₺",
                "Per  Market ......... 650 ₺", "Cum  Sinema .......... 230 ₺", "Cmt  Abonelik ....... 199 ₺"]
        n = int(clamp(ct / 5.5, 0, 1) * 7.99)
        for i in range(min(n + 1, 7)):
            T(460, 190 + i * 40 - 2, rows[i], 22, INK, False, 'lm')


# ---------- KURAL 2: ACİL DURUM FONU
@scene("Acil Durum Fonunu Kur", 2, (226, 236, 248), [
    "Kural 2: Acil durum fonu. Çünkü hayat plan yapmaz.",
    "Can'ın arabası bir sabah bozuldu. Tamir 15.000 lira. Birikimi yok. Şimdi ne olacak?",
    "Çoğu insan kredi kartına sarılır. Faiz işlemeye başlar ve küçük bir sorun, büyük bir borca dönüşür.",
    "Acil durum fonu; iş kaybı, hastalık ya da tamir gibi sürprizler için ayrılan güvenlik yastığıdır.",
    "Hedef: zorunlu aylık giderlerinin üç ila altı katı. Gideri 10.000 lira ise hedef 30.000 ile 60.000 lira arasıdır.",
    "Bu para yatırım değil, sigortadır. Amacı kazandırmak değil; ihtiyaç anında hemen elinin altında olmaktır.",
    "Nerede tutulur? Kolay çekilebilen ve değeri aniden oynamayan bir yerde: vadesiz veya likit bir hesap gibi.",
    "Küçük başla. Önce 1.000 lira, sonra bir aylık gider, sonra üç aylık. Her adım seni biraz daha rahat uyutur.",
    "Fon dolunca yatırım yapmak da kolaylaşır; çünkü ilk fırtınada paranı zararına bozdurmak zorunda kalmazsın.",
])
def s_emergency(ci, ct, t):
    if ci == 0:
        cloud(330, 150, 1.4); cloud(700, 120, 1.1)
        for i in range(40):
            x = (i * 53 + t * 60) % 900 + 100; y = (i * 97 + t * 420) % 360 + 190
            L(x, y, x - 6, y + 20, 2, BLUE)
        hr = man(500, GY, 'umb', t, acc='cap', sc=1.25, name='CAN')['hr']
        L(hr[0], hr[1], hr[0], hr[1] - 90, 3)
        D.pieslice([(hr[0] - 90) * S, (hr[1] - 150) * S, (hr[0] + 90) * S, (hr[1] + 30) * S], 180, 360, fill=RED, outline=INK, width=3 * S)
    elif ci == 1:
        cx = 760 + 8 * math.sin(t * 20) * (1 if ct < 1.5 else 0)
        R(cx - 170, 400, cx + 170, 470, RED, INK, 4, 20); POLY([(cx - 100, 400), (cx - 60, 345), (cx + 70, 345), (cx + 120, 400)], RED, INK, 4)
        POLY([(cx - 85, 396), (cx - 55, 355), (cx - 5, 355), (cx - 5, 396)], (200, 230, 255)); POLY([(cx + 5, 396), (cx + 5, 355), (cx + 62, 355), (cx + 100, 396)], (200, 230, 255))
        for dx in (-100, 100): C(cx + dx, 475, 32, INK); C(cx + dx, 475, 14, GRAY)
        for i in range(4): C(cx - 150 - i * 25 - (ct * 10) % 30, 380 - i * 25, 14 + i * 5, (170, 170, 180))
        bubble(cx, 250, 220, 70, "15.000 ₺", 34, (255, 235, 235), RED)
        man(300, GY, 'sad', t, acc='cap', mood='sad', sc=1.25, name='CAN')
    elif ci == 2:
        man(250, GY, 'sad', t, acc='cap', mood='sad', sc=1.25)
        card(620, 340, 260, 160, PURPLE, "KREDİ KARTI")
        v = 15000 * (1 + 0.5 * prog(ct, 0.5, 5))
        T(620, 460, f"Borç: {int(v):,} ₺".replace(",", "."), 30, RED, True)
        arrow(850, 420, 850, 240 - 60 * prog(ct, 0.5, 5), RED, 8)
        T(930, 330, "FAİZ", 34, RED, True)
    elif ci == 3:
        man(220, GY, 'point', t, acc='cap', sc=1.25)
        shield(640, 310, 2.0, BLUE, "")
        T(640, 285, "ACİL", 40, WHITE, True); T(640, 335, "DURUM FONU", 28, WHITE, True)
        for i, (lab, x) in enumerate((("İş kaybı", 930), ("Hastalık", 1040), ("Tamir", 1150))):
            if ct > i * 1.2: bubble(x, 240 + i * 90, 160, 50, lab, 22, (255, 240, 220))
    elif ci == 4:
        man(190, GY, 'point', t, acc='cap', sc=1.25)
        T(740, 130, "Aylık gider: 10.000 ₺", 30, INK, True)
        n = int(clamp(ct / 8.0, 0, 1) * 5.99) + 1
        for i in range(6):
            x = 400 + i * 130
            fill = GREEN if i >= 2 else ORANGE
            R(x, 260, x + 115, 420, WHITE, INK, 3, 10)
            if i < n:
                R(x + 5, 265, x + 110, 415, fill, None, 0, 8)
                T(x + 57, 340, f"{(i + 1) * 10}.000", 22, WHITE, True)
            T(x + 57, 445, f"{i + 1}. ay", 20, GRAY, True)
        if n >= 3: T(740, 215, "Hedef bölge: 3-6 ay", 26, GREEN, True)
    elif ci == 5:
        man(220, GY, 'think', t, acc='cap', sc=1.25)
        for i, (lab, c, x) in enumerate((("YATIRIM", ORANGE, 600), ("SİGORTA", BLUE, 900))):
            R(x - 120, 220, x + 120, 420, WHITE, INK, 4, 20)
            T(x, 270, lab, 34, c, True)
            if i == 0: T(x, 350, "kazandırmak", 24, GRAY, True); cross(x + 90, 235, 1.1)
            else: T(x, 350, "hazır olmak", 24, GRAY, True); check(x + 90, 235, 1.2)
        T(750, 480, "Fonun işi: lazım olduğunda orada olmak", 26, INK, True)
    elif ci == 6:
        man(190, GY, 'point', t, acc='cap', sc=1.25)
        x = 640
        POLY([(x - 170, 270), (x + 170, 270), (x, 170)], BLUE, INK, 4)
        for k in range(5): R(x - 150 + k * 66, 280, x - 126 + k * 66, 430, WHITE, INK, 3)
        R(x - 180, 430, x + 180, 460, GRAY, INK, 3); R(x - 195, 460, x + 195, 480, GRAY, INK, 3)
        T(x, 245, "HESAP", 24, WHITE, True)
        for i, s in enumerate(("Kolay çekilir", "Değeri oynamaz", "Hemen ulaşırsın")):
            if ct > i * 1.4: check(930, 250 + i * 70, 0.8); T(960, 250 + i * 70, s, 24, INK, True, 'lm')
    elif ci == 7:
        man(150, GY, 'cheer', t, acc='cap', sc=1.25)
        for i, (lab, lvl) in enumerate((("1.000 ₺", 0.15), ("1 aylık", 0.4), ("3 aylık", 1.0))):
            x = 460 + i * 270
            R(x - 70, 230, x + 70, 450, WHITE, INK, 4, 20)
            fh = 210 * lvl * prog(ct, 0.8 + i * 2.0, 1.8)
            if fh > 2: R(x - 66, 446 - fh, x + 66, 446, GREEN, None, 0, 16)
            T(x, 490, lab, 26, INK, True)
            T(x, 205, f"{i + 1}. adım", 22, GRAY, True)
    else:
        for i in range(3): T(560 + i * 50, 200 - i * 35 - (ct * 8) % 20, "Z", 36 + i * 10, BLUE, True)
        R(330, 400, 880, 470, (180, 200, 240), INK, 4, 14); R(330, 365, 360, 505, BROWN, INK, 4); R(850, 400, 880, 505, BROWN, INK, 4)
        C(430, 385, 24, WHITE, INK, 4); ARC(420, 380, 440, 395, 15, 165, 3)
        L(454, 385, 700, 395, 6)
        shield(1030, 330, 1.2, BLUE)
        T(1030, 440, "Fonum hazır", 24, BLUE, True)


# ---------- KURAL 3: BİLEŞİK GETİRİ
def fv(monthly, years, rate=0.10):
    r = rate / 12; n = int(years * 12)
    return monthly * (((1 + r) ** n - 1) / r)


def series(start_age, end_age):
    return [(a, fv(2000, a - start_age)) for a in range(start_age, end_age + 1)]


@scene("Zamanı Arkana Al", 3, (233, 246, 236), [
    "Kural 3: Zamanı arkana al. Bileşik getiri, parayı büyüten en güçlü etkilerden biridir.",
    "Fikir basit: kazandığın getiri de zamanla kazanç üretmeye başlar. Tıpkı bir kartopu gibi.",
    "İki arkadaşı tanıyalım: Ali 25 yaşında başladı. Veli ise 35 yaşında.",
    "İkisi de her ay 2.000 lira ayırıyor ve 55 yaşına kadar devam ediyor. Yıllık ortalama yüzde 10 getiri varsayalım.",
    "Ali, 30 yıl boyunca toplam 720.000 lira yatırdı ve sonunda yaklaşık 4,5 milyon liraya ulaştı.",
    "Veli, 20 yıl boyunca 480.000 lira yatırdı ve yaklaşık 1,5 milyon liraya ulaştı.",
    "Ali sadece 240.000 lira daha fazla yatırdı ama sonuç yaklaşık üç kat. Fark paradan değil, zamandan geldi.",
    "Ders: Mükemmel zamanı bekleme. Küçük de olsa bugün başlamak, büyük ama geç başlamaktan çoğu zaman daha güçlüdür.",
    "Uyarı: Getiriler garanti değildir; piyasalar inip çıkar. Bu örnek, mantığı göstermek için basitleştirildi.",
    "Bir de masrafları unutma: yüksek komisyon ve vergiler bileşik etkiyi aşağı çeker. Küçük yüzdeler, uzun vadede büyük fark yaratır.",
])
def s_compound(ci, ct, t):
    if ci <= 1:
        POLY([(0, 330), (1280, 520), (1280, 720), (0, 720)], (200, 225, 200), INK, 3)
        k = (0.1 + ct * 0.1 + ci * 0.45) % 1
        bx = 250 + k * 800
        slope_y = lambda x: 330 + x / 1280 * 190
        r = 20 + k * 80
        C(bx, slope_y(bx) - r, r, (240, 245, 255), INK, 4)
        mx = bx - r - 60
        man(mx, slope_y(mx) + 2, 'push', t, acc='cap', sc=0.9)
        T(640, 110, "Getiri üstüne getiri", 40, GREEN, True)
    elif ci == 2:
        man(330, GY, 'wave', t, acc='cap', name='ALİ (25 yaş)', sc=1.2)
        man(900, GY, 'wave', t + 0.5, acc='glasses', col=PURPLE, name='VELİ (35 yaş)', sc=1.2)
    elif ci == 3:
        man(160, GY, 'idle', t, acc='cap', sc=0.9); man(290, GY, 'idle', t, acc='glasses', col=PURPLE, sc=0.9)
        for i, s in enumerate(("Aylık 2.000 ₺", "Yıllık %10 (varsayım)", "55 yaşına kadar")):
            if ct > i * 2: R(560, 180 + i * 85, 1090, 245 + i * 85, WHITE, INK, 3, 14); T(825, 212 + i * 85, s, 28, INK, True)
    else:
        man(90, GY, 'cheer' if ci in (4, 6) else 'idle', t, acc='cap', sc=0.7, name='ALİ')
        man(210, GY, 'cheer' if ci == 5 else 'idle', t, acc='glasses', col=PURPLE, sc=0.7, name='VELİ')
        ox, oy, cw, chh = 380, 120, 780, 340
        R(ox - 60, oy - 40, ox + cw + 40, oy + chh + 55, WHITE, INK, 3, 12)
        L(ox, oy + chh, ox + cw, oy + chh, 3); L(ox, oy, ox, oy + chh, 3)
        maxv = 4.7e6
        for a in range(25, 56, 5):
            x = ox + (a - 25) / 30 * cw; L(x, oy + chh, x, oy + chh + 8, 2); T(x, oy + chh + 26, str(a), 16, GRAY, True)
        for v in (1, 2, 3, 4):
            y = oy + chh - v * 1e6 / maxv * chh; L(ox, y, ox + cw, y, 1, LGRAY); T(ox - 8, y, f"{v} mn", 15, GRAY, True, 'rm')
        T(ox + cw / 2, oy - 22, "Birikim (₺)  -  yaş", 20, INK, True)
        ali = series(25, 55); vel = series(35, 55)

        def plot(data, col, frac):
            pts = [(ox + (a - 25) / 30 * cw, oy + chh - v / maxv * chh) for a, v in data]
            n = int(len(pts) * frac) if frac > 0 else 0
            for i in range(1, n): L(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], 5, col)
        fa = 1.0 if ci >= 5 else prog(ct, 0.3, 5.5)
        fv_ = 0 if ci < 5 else (1.0 if ci >= 6 else prog(ct, 0.3, 4.5))
        plot(ali, BLUE, fa); plot(vel, PURPLE, fv_)
        if ci >= 4 and fa >= 0.99:
            T(ox + 20, oy + 20, "ALİ ≈ 4,5 milyon ₺", 24, BLUE, True, 'lm'); T(ox + 20, oy + 50, "(yatırdığı: 720.000)", 17, GRAY, True, 'lm')
        if ci >= 5 and fv_ >= 0.99:
            T(ox + 20, oy + 90, "VELİ ≈ 1,5 milyon ₺", 24, PURPLE, True, 'lm'); T(ox + 20, oy + 120, "(yatırdığı: 480.000)", 17, GRAY, True, 'lm')
        if ci == 6: bubble(ox + 330, oy + 200, 220, 58, "≈ 3 KAT", 34, (220, 255, 230), GREEN)
        if ci == 7: bubble(ox + 300, oy + 230, 520, 70, "Bugün başla > mükemmeli bekle", 28, (255, 250, 210))
        if ci == 8:
            POLY([(ox + 300, oy + 180), (ox + 400, oy + 180), (ox + 350, oy + 85)], GOLD, INK, 4); T(ox + 350, oy + 155, "!", 44, INK, True)
            T(ox + 420, oy + 130, "Getiri garanti değildir", 28, RED, True, 'lm')
        if ci == 9:
            for i in range(5):
                q = ((ct * 0.25 + i / 5) % 1)
                coin(ox + 330 + i * 80, oy + 90 + q * 30, 15)
                T(ox + 330 + i * 80, oy + 150 + q * 30, "%", 24, RED, True)
            T(ox + 330, oy + 240, "Komisyon ve vergi küçük görünür,", 22, RED, True, 'lm'); T(ox + 330, oy + 270, "uzun vadede büyür", 22, RED, True, 'lm')


# ---------- KURAL 4: ENFLASYON
@scene("Enflasyonu Hesaba Kat", 4, (255, 238, 232), [
    "Kural 4: Enflasyonu hesaba kat. Yatağın altındaki para sessizce erir.",
    "Geçen yıl 100 liraya aldığın sepet bu yıl 150 lira olduysa, fiyatlar yüzde 50 artmış demektir.",
    "Senin 100 liran ise hâlâ 100 lira. Ama artık sepetin sadece üçte ikisini alabiliyor. Yani alım gücün yaklaşık yüzde 33 azaldı.",
    "Nominal getiri ile reel getiri farkı budur. Hesabına yüzde 40 faiz yatsa bile enflasyon yüzde 50 ise, gerçekte yaklaşık yüzde 7 kaybediyorsun.",
    "Basit formül: reel getiri yaklaşık olarak, getiri eksi enflasyondur.",
    "Bu yüzden sadece biriktirmek yetmez; enflasyonla yarışabilecek araçları da tanımalısın.",
    "Ama dikkat: Enflasyonu yenmeye çalışırken risksiz getiri arama. Daha yüksek potansiyel getiri, genellikle daha yüksek dalgalanma demektir.",
    "Gelirini artırmak da güçlü bir savunmadır: yeni beceriler, yan gelir, maaş pazarlığı. Yatırımların en iyisi kendindir.",
])
def s_inflation(ci, ct, t):
    if ci == 0:
        R(330, 380, 960, 450, (170, 130, 220), INK, 4, 16); R(330, 450, 960, 470, (120, 90, 170), INK, 4)
        rr = 50 * (1 - 0.55 * prog(ct, 0.5, 6))
        coin(560, 365 - rr, rr, '₺'); T(560, 270, "100 ₺", 26, INK, True)
        for i in range(3): T(1060 + i * 40, 330 - i * 30, "Z", 34 + i * 8, BLUE, True)
        man(160, GY, 'think', t, acc='cap', mood='open', sc=1.2)
    elif ci == 1:
        man(180, GY, 'point', t, acc='cap', sc=1.2)
        for i, (x, lab, v0, v1) in enumerate(((560, "Geçen yıl", 100, 100), (900, "Bu yıl", 100, 150))):
            R(x - 90, 330, x + 90, 450, (200, 160, 110), INK, 4, 10); L(x - 90, 345, x + 90, 345, 3)
            for k in range(3): C(x - 50 + k * 50, 322, 22, (240, 90, 80) if k == 0 else (255, 200, 50), INK, 2)
            T(x, 250, lab, 28, GRAY, True)
            v = v0 + (v1 - v0) * prog(ct, 1, 2.5) if i else v0
            R(x - 60, 465, x + 60, 505, WHITE, INK, 3, 10); T(x, 485, f"{int(v)} ₺", 28, RED if i else INK, True)
        arrow(670, 400, 790, 400, INK, 6); T(730, 360, "+%50", 28, RED, True)
    elif ci == 2:
        man(170, GY, 'sad', t, acc='cap', mood='sad', sc=1.2)
        for i, (x, lab, fr) in enumerate(((560, "Geçen yıl: 100 ₺", 1.0), (930, "Bu yıl: 100 ₺", 1 - 0.333 * prog(ct, 1, 3)))):
            R(x - 85, 250, x + 85, 470, WHITE, INK, 4, 12)
            R(x - 81, 466 - 212 * fr, x + 81, 466, GREEN if not i else ORANGE, None, 0, 8)
            T(x, 515, lab, 24, INK, True)
            T(x, 360, f"%{int(fr * 100)}", 44, WHITE, True)
        T(740, 210, "Alım gücü", 28, INK, True); T(930, 225, "≈ %33 azaldı", 24, RED, True)
    elif ci == 3:
        man(170, GY, 'shrug', t, acc='cap', sc=1.2)
        for i, (x, lab, v, c) in enumerate(((470, "Faiz", "%40", GREEN), (730, "Enflasyon", "%50", RED), (990, "Reel getiri", "≈ -%7", PURPLE))):
            if ct > i * 2.2:
                R(x - 110, 230, x + 110, 400, WHITE, c, 5, 18); T(x, 275, lab, 26, GRAY, True); T(x, 345, v, 54, c, True)
        if ct > 2.2: T(600, 315, "−", 50, INK, True)
        if ct > 4.4: T(860, 315, "=", 50, INK, True)
    elif ci == 4:
        man(170, GY, 'point', t, acc='glasses', sc=1.2)
        R(380, 220, 1180, 350, WHITE, INK, 5, 24); T(780, 285, "Reel getiri ≈ Getiri − Enflasyon", 38, BLUE, True)
        T(780, 410, "Örnek: %60 − %50 = yaklaşık +%10", 28, GREEN, True)
    elif ci == 5:
        man(250, GY, 'run', t, acc='cap', sc=1.2)
        px = 700 + ct * 20
        R(px - 70, 330, px + 70, 400, RED, INK, 3, 10); T(px, 365, "FİYATLAR", 24, WHITE, True)
        arrow(px + 80, 365, px + 160, 365, RED, 6)
        T(900, 220, "Fiyatlarla yarış!", 36, INK, True)
    elif ci == 6:
        man(160, GY, 'think', t, acc='cap', sc=1.2)
        ang = -14 * math.sin(clamp(ct / 2.5) * 3.14 / 2)
        cx, cy = 690, 440
        POLY([(cx - 40, 505), (cx + 40, 505), (cx, cy)], GRAY, INK, 3)
        a = math.radians(ang); dx, dy = 280 * math.cos(a), 280 * math.sin(a)
        L(cx - dx, cy - dy, cx + dx, cy + dy, 10, BROWN)
        T(cx - dx, cy - dy - 40, "Daha yüksek getiri", 22, GREEN, True); T(cx + dx, cy + dy - 40, "Daha yüksek risk", 22, RED, True)
        C(cx - dx, cy - dy - 10, 14, GREEN); C(cx + dx, cy + dy - 10, 14, RED)
    else:
        man(260, GY, 'cheer', t, acc='glasses', sc=1.2)
        for i, (lab, x, c) in enumerate((("Yeni beceri", 560, BLUE), ("Yan gelir", 790, GREEN), ("Maaş pazarlığı", 1030, ORANGE))):
            if ct > i * 1.5:
                R(x - 100, 240, x + 100, 340, WHITE, c, 5, 16); T(x, 290, lab, 24, INK, True)
                T(x, 210, "↑", 44, c, True)
        if ct > 5: C(790, 130, 32 + 3 * math.sin(t * 5), GOLD, INK, 3); T(790, 130, "!", 34, INK, True)


# ---------- KURAL 5: BORÇ
@scene("Kötü Borçla Dost Olma", 5, (240, 232, 250), [
    "Kural 5: Kötü borçla dost olma.",
    "Her borç kötü değil. Eğitim ya da değer kazanan bir varlık için, makul koşullarla alınan borç bir araç olabilir.",
    "Kötü borç ise tüketim için alınan, faizi yüksek olan borçtur. En bilinen örneği kredi kartı borcudur.",
    "Diyelim 10.000 lira borcun var ve aylık faiz yüzde 4. Hiç dokunmazsan bir yıl sonra borç yaklaşık 16.000 lira olur.",
    "Asgari ödeme tuzağı: Sadece asgari tutarı ödersen borç çok yavaş azalır, faiz ise çok hızlı işler.",
    "Borç kapatmak için iki meşhur yöntem var: Çığ yöntemi ve kartopu yöntemi.",
    "Çığ yöntemi: En yüksek faizli borca öncelik ver. Matematiksel olarak en ucuz yol budur.",
    "Kartopu yöntemi: En küçük borçtan başla. Küçük zaferler motivasyon verir; bazen disiplin, matematikten daha önemlidir.",
    "Hangisini seçersen seç, önce yeni borç almayı durdur. Delik kovayı doldurmaya çalışma, önce deliği kapat.",
    "Kredi kartını nakit gibi kullan: Ay sonunda ekstreyi tam öde. Ödeyemeyeceğin şeyi kartla alma.",
])
def s_debt(ci, ct, t):
    if ci == 0:
        man(480, GY, 'sad', t, mood='sad', acc='tie', sc=1.3)
        L(480, GY - 5, 650, GY - 25, 6, GRAY); C(690, GY - 38, 42, (80, 80, 95), INK, 4); T(690, GY - 38, "BORÇ", 20, WHITE, True)
    elif ci == 1:
        man(150, GY, 'point', t, acc='tie', sc=1.1)
        R(360, 160, 740, 470, WHITE, GREEN, 5, 20); R(780, 160, 1160, 470, WHITE, RED, 5, 20)
        T(550, 200, "İYİ BORÇ", 32, GREEN, True); T(970, 200, "KÖTÜ BORÇ", 32, RED, True)
        check(550, 270, 1.4)
        POLY([(520, 330), (550, 315), (580, 330), (550, 345)], INK); R(535, 338, 565, 355, INK)
        T(550, 400, "Eğitim • Değer kazanan", 20, INK, True); T(550, 428, "makul koşullar", 20, INK, True)
        cross(970, 270, 1.4); card(970, 355, 150, 90, PURPLE); T(970, 428, "Tüketim • Yüksek faiz", 20, INK, True)
    elif ci == 2:
        man(220, GY, 'sad', t, mood='sad', acc='tie', sc=1.3)
        card(700, 300, 360, 220, PURPLE, "KREDİ KARTI")
        T(700, 460, "Yüksek faiz = pahalı borç", 30, RED, True)
        for i in range(4): T(930 + i * 5, 250 - i * 40 + (ct * 15) % 10, "%", 30 + i * 6, RED, True)
    elif ci == 3:
        man(130, GY, 'shrug', t, acc='tie', sc=1.0)
        ox = 290
        R(ox - 15, 110, ox + 910, 520, WHITE, INK, 3, 12)
        T(ox + 450, 140, "Aylık %4 faizle 10.000 ₺ borç", 28, INK, True)
        shown = int(clamp(ct / 6.5, 0, 1) * 12.99)
        for m in range(13):
            v = 10000 * 1.04 ** m
            h = (v / 16500) * 270
            x = ox + 20 + m * 66
            if m <= shown:
                R(x, 470 - h, x + 52, 470, ORANGE if m < 12 else RED, INK, 2, 3)
                if m in (0, 6, 12): T(x + 26, 455 - h, f"{int(v):,}".replace(",", "."), 17, INK, True)
            T(x + 26, 495, str(m), 16, GRAY, True)
        T(ox + 865, 495, "ay", 18, GRAY, True)
    elif ci == 4:
        man(200, GY, 'sad', t, mood='sad', acc='tie', sc=1.2)
        R(450, 200, 1180, 280, WHITE, INK, 3, 14); R(450, 350, 1180, 430, WHITE, INK, 3, 14)
        T(470, 175, "Asgari ödeme: borç yavaş azalır", 24, GREEN, True, 'lm'); T(470, 325, "Faiz: borç hızla yeniden büyür", 24, RED, True, 'lm')
        R(455, 205, 455 + 710 * (1 - 0.1 * prog(ct, 0.5, 5)), 275, GREEN, None, 0, 10)
        R(455, 355, 455 + 710 * (0.35 + 0.6 * prog(ct, 0.5, 5)), 425, RED, None, 0, 10)
        T(815, 240, "▼ az", 28, WHITE, True); T(815, 390, "▲ çok", 28, WHITE, True)
        T(815, 490, "TUZAK!", 40, RED, True)
    elif ci == 5:
        man(140, GY, 'idle', t, acc='glasses', sc=1.1)
        for x, lab, c in ((520, "ÇIĞ", BLUE), (960, "KARTOPU", ORANGE)):
            R(x - 170, 200, x + 170, 440, WHITE, c, 5, 20); T(x, 400, lab, 36, c, True)
        POLY([(430, 350), (520, 235), (610, 350)], (220, 235, 255), INK, 3)
        for k in range(5): C(540 + k * 18, 290 + k * 20, 8 + k * 2, WHITE, INK, 2)
        C(930, 320, 55, (240, 245, 255), INK, 4); C(1010, 345, 28, (240, 245, 255), INK, 3); arrow(960, 280, 1050, 280, INK, 4)
    elif ci in (6, 7):
        man(130, GY, 'point' if ci == 6 else 'cheer', t, acc='glasses', sc=1.1)
        if ci == 6:
            T(760, 120, "Çığ: en yüksek faiz önce", 30, BLUE, True)
            debts = [("Kart A", "%4", "5.000 ₺"), ("Kart B", "%3", "2.000 ₺"), ("Kredi", "%2", "12.000 ₺")]
            hc, hf = BLUE, (210, 225, 255)
        else:
            T(760, 120, "Kartopu: en küçük borç önce", 30, ORANGE, True)
            debts = [("Kart B", "%3", "2.000 ₺"), ("Kart A", "%4", "5.000 ₺"), ("Kredi", "%2", "12.000 ₺")]
            hc, hf = ORANGE, (255, 235, 205)
        for i, (n, r, a) in enumerate(debts):
            y = 220 + i * 100; hl = i == 0
            R(420, y - 36, 1100, y + 36, hf if hl else WHITE, hc if hl else INK, 4 if hl else 2, 14)
            T(450, y, n, 26, INK, True, 'lm'); T(700, y, "Faiz " + r, 26, RED if ci == 6 else GRAY, True, 'lm'); T(900, y, a, 26, INK, True, 'lm')
            if hl: T(1085, y, "1. ÖNCE", 22, hc, True, 'rm')
    elif ci == 8:
        man(200, GY, 'push', t, acc='tie', sc=1.2)
        x = 640
        POLY([(x - 90, 250), (x + 90, 250), (x + 65, 470), (x - 65, 470)], (150, 160, 180), INK, 4)
        C(x + 20, 400, 11, WHITE, INK, 3)
        for i in range(5):
            q = (ct * 0.6 + i / 5) % 1; coin(x + 20, 420 + q * 80, 11)
        T(x, 160, "Önce deliği kapat!", 34, RED, True)
        for i in range(3): coin(x - 25 + i * 22, 230 - i * 8, 12)
    else:
        man(180, GY, 'cheer', t, acc='tie', sc=1.2)
        R(450, 160, 1000, 480, WHITE, INK, 4, 16)
        T(725, 205, "KART EKSTRESİ", 28, PURPLE, True)
        for k, (n, a) in enumerate((("Market", "650 ₺"), ("Fatura", "480 ₺"), ("Ulaşım", "220 ₺"))):
            T(490, 270 + k * 45, n, 24, INK, False, 'lm'); T(960, 270 + k * 45, a, 24, INK, True, 'rm')
        L(470, 400, 980, 400, 3)
        T(725, 440, "TAM ÖDE ✓", 40, GREEN, True)


# ---------- KURAL 6: ÇEŞİTLENDİRME
@scene("Yumurtaları Tek Sepete Koyma", 6, (255, 249, 225), [
    "Kural 6: Bütün yumurtaları aynı sepete koyma.",
    "Can, bir arkadaşından 'bu hisse uçacak' sözünü duydu ve tüm birikimini tek şirkete yatırdı. Şirket kötü haber alınca değer düştü.",
    "Çeşitlendirme, riski farklı varlıklara dağıtmaktır. Biri düşerken diğerleri ayakta kalabilir.",
    "Sepetler farklı olabilir: nakit ve mevduat, altın, hisse senedi, yatırım fonları, emeklilik planları, gayrimenkul.",
    "Her sepetin bir kişiliği var: Bazısı sakin ama az kazandırır, bazısı hızlı ama sert dalgalanır.",
    "Doğru dağılım yaşına, hedefine ve riske tahammülüne bağlıdır. Yakın zamanda paraya ihtiyacı olan biri, sert dalgalanmaya katlanamayabilir.",
    "Paraya ne zaman ihtiyacın var? Önemli soru bu. Bir yıl içinde lazımsa riskli yere koyma. Yirmi yıl sonra lazımsa dalgalanma sana zaman tanır.",
    "Ve düzenli ol: Her ay sabit tutar yatırmak, 'zirve mi dip mi' stresini azaltır.",
    "Anlamadığın şeye yatırım yapma. Bir ürünü iki cümleyle anlatamıyorsan, henüz hazır değilsin.",
])
def s_diversify(ci, ct, t):
    if ci == 0:
        man(380, GY, 'hold', t, acc='cap', sc=1.25); basket(600, 420, 190, "TEK SEPET", 8)
    elif ci == 1:
        fall = prog(ct, 2.0, 1.2)
        sad = ct > 3
        man(330, GY, 'sad' if sad else 'hold', t, acc='cap', mood='sad' if sad else 'smile', sc=1.25)
        bx = 600 + 20 * fall; by = 420 + 75 * fall
        basket(bx, by, 190, "TEK ŞİRKET", 8 if fall < 0.5 else 0)
        if fall > 0.5:
            for i in range(8):
                ex = bx - 70 + i * 22 + fall * (i - 4) * 12; ey = by + 50 - abs(math.sin(i)) * 10
                D.ellipse([(ex - 12) * S, (ey - 6) * S, (ex + 12) * S, (ey + 8) * S], fill=(255, 220, 90), outline=INK, width=2 * S)
        px = [(880 + i * 40, 230 + (i * 17 % 30) + i * 14) for i in range(8)]
        for i in range(1, 8): L(px[i - 1][0], px[i - 1][1], px[i][0], px[i][1], 4, RED)
        T(1000, 190, "▼ -%60", 40, RED, True)
    elif ci == 2:
        man(150, GY, 'point', t, acc='cap', sc=1.1)
        for i, x in enumerate((420, 640, 860, 1080)):
            fall = prog(ct, 3.5, 1) if i == 1 else 0
            basket(x, 400 + 70 * fall, 150, "", 0 if (i == 1 and fall > 0.5) else 5)
        T(820, 190, "Biri düşse de diğerleri ayakta", 28, GREEN, True)
    elif ci in (3, 4):
        man(100, GY, 'point', t, acc='cap', sc=0.9)
        labs = [("Nakit / Mevduat", (110, 170, 230), "Sakin"), ("Altın", (220, 180, 40), "Orta"), ("Hisse", (230, 100, 90), "Sert"),
                ("Fonlar", (130, 110, 220), "Orta"), ("Emeklilik", (80, 180, 140), "Uzun vade"), ("Gayrimenkul", (200, 130, 80), "Yavaş")]
        for i, (n, c, k) in enumerate(labs):
            x = 330 + (i % 3) * 340; y = 230 + (i // 3) * 200
            if ct > i * 0.9 or ci == 4:
                basket(x, y, 150, n, 4, c)
            if ci == 4:
                col = GREEN if k in ("Sakin", "Yavaş") else (ORANGE if k in ("Orta", "Uzun vade") else RED)
                if ct > i * 1.0: bubble(x, y - 70, 120, 34, k, 18, WHITE, col)
    elif ci == 5:
        man(100, GY, 'think', t, acc='glasses', sc=1.0)
        T(720, 120, "Örnek dağılımlar (varsayımsal)", 28, INK, True)
        sbar(280, 170, 880, 70, (0.7, 0.2, 0.1), (RED, ORANGE, GREEN), prog(ct, 0.3, 2), ["Riskli %70", "Orta %20", "%10"])
        T(280, 262, "Uzun vadeli hedef", 22, INK, True, 'lm')
        sbar(280, 320, 880, 70, (0.15, 0.25, 0.6), (RED, ORANGE, GREEN), prog(ct, 3, 2), ["%15", "Orta %25", "Güvenli %60"])
        T(280, 412, "Yakın hedef (1-2 yıl)", 22, INK, True, 'lm')
    elif ci == 6:
        man(100, GY, 'think', t, acc='glasses', sc=1.0)
        R(280, 140, 1200, 500, WHITE, INK, 3, 14)
        for lab, x in (("1 yıl", 420), ("5 yıl", 700), ("20 yıl", 980)):
            T(x, 470, lab, 26, INK, True)
        pts = []
        for i in range(0, 61):
            x = 320 + i * 14
            amp = 80 * (1 - i / 75)
            pts.append((x, 400 - i * 2.3 + amp * math.sin(i * 0.9)))
        n = int(len(pts) * prog(ct, 0.3, 6))
        for i in range(1, n): L(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], 4, BLUE)
        T(740, 175, "Zaman, dalgalanmayı yumuşatır", 28, GREEN, True)
        T(420, 235, "Sert", 22, RED, True); T(980, 235, "Daha sakin", 22, GREEN, True)
    elif ci == 7:
        man(100, GY, 'cheer', t, acc='cap', sc=1.0)
        T(740, 130, "Her ay aynı tutar", 30, INK, True)
        for i in range(10):
            x = 290 + i * 90
            if ct > i * 0.45:
                h = 60 + (i * 37 % 50)
                R(x, 460 - h, x + 60, 460, BLUE, INK, 2, 4)
                coin(x + 30, 460 - h - 25 - (ct * 20) % 8, 13)
                T(x + 30, 485, f"{i + 1}.ay", 16, GRAY, True)
    else:
        man(260, GY, 'shrug', t, acc='glasses', mood='open', sc=1.25)
        R(520, 190, 1180, 330, (255, 245, 205), INK, 4, 20)
        T(850, 235, "2 cümleyle anlatabiliyor muyum?", 30, INK, True)
        if ct > 4: T(850, 290, "Hayır → Henüz hazır değilim", 30, RED, True)
        T(930, 440, "?", int(120 + 10 * math.sin(t * 4)), ORANGE, True)


# ---------- KURAL 7: PSİKOLOJİ
@scene("En Büyük Rakibin Kendi Beynin", 7, (240, 238, 250), [
    "Kural 7: En büyük rakibin kendi beynin.",
    "İnsanlar kaybetmekten, kazanmaktan daha fazla korkar. Fiyat düşünce panikle satar, fiyat uçunca heyecanla alırlar. Yani pahalıdan alıp ucuzdan satarlar.",
    "Buna FOMO diyoruz: kaçırma korkusu. Herkes kazanıyormuş gibi görünce sen de koşarak sürüye katılırsın.",
    "Sosyal medyada hep kazançlar paylaşılır, kayıplar değil. Gördüğün şey, sadece kazananların vitrini.",
    "Dolandırıcılık alarmı: 'Garantili yüksek kazanç', 'Kaçırma, sadece bugün', 'Kimseye söyleme'. Bu cümleleri duyduysan uzaklaş.",
    "Gerçek bir yatırımda garanti yoktur. Yüksek getiri vaadi çoğu zaman ya yüksek riskin ya da dolandırıcılığın işaretidir.",
    "Yatırım yapmadan önce kurumun lisanslı ve denetlenen bir kuruluş olduğunu resmi kaynaklardan kontrol et. Şifreni ve doğrulama kodlarını asla kimseyle paylaşma.",
    "Planını yaz: Neden aldığını, ne zaman satacağını. Duygular patladığında kâğıttaki plana bak.",
    "Ve kaybetmeyi göze alabileceğin kadarıyla başla. Uykunu kaçıran para, yanlış yerdedir.",
])
def s_psych(ci, ct, t):
    if ci == 0:
        man(640, GY, 'think', t, acc='cap', mood='open', sc=1.25)
        bubble(380, 250, 230, 70, "AL! AL! AL!", 28, (220, 255, 230), GREEN)
        bubble(900, 250, 230, 70, "SAT! SAT!", 28, (255, 225, 225), RED)
        T(640, 130, "Kalp mi, plan mı?", 38, INK, True)
    elif ci == 1:
        man(140, GY, 'sad', t, acc='cap', mood='open', sc=1.0)
        pts = [(330, 400), (430, 340), (530, 300), (630, 230), (720, 170), (800, 260), (880, 340), (980, 420), (1080, 470)]
        n = int(clamp(ct / 6, 0, 1) * 8) + 1
        for i in range(1, min(n + 1, 9)): L(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], 5, BLUE)
        if n >= 5: C(pts[4][0], pts[4][1], 14, GREEN, INK, 3); T(pts[4][0], pts[4][1] - 40, "ALDIM (heyecan)", 20, GREEN, True)
        if n >= 8: C(pts[8][0], pts[8][1], 14, RED, INK, 3); T(pts[8][0] - 40, pts[8][1] + 40, "SATTIM (panik)", 20, RED, True)
        T(700, 120, "Pahalıdan al, ucuzdan sat?!", 32, INK, True)
    elif ci == 2:
        R(940, 190, 1160, 330, (255, 235, 180), INK, 4, 20); T(1050, 260, "%300!", 48, GREEN, True)
        for i in range(7):
            man(160 + i * 105 + min(ct, 8) * 12, GY, 'run', t + i * 0.3, sc=0.65, col=(90 + i * 15, 90, 120))
        T(640, 120, "FOMO: Kaçırma korkusu", 38, ORANGE, True)
    elif ci == 3:
        man(160, GY, 'think', t, acc='cap', sc=1.1)
        R(520, 90, 800, 520, (40, 44, 60), INK, 5, 34); R(540, 120, 780, 490, WHITE, None, 0, 10)
        for i in range(3):
            y = 170 + i * 105
            R(552, y - 38, 768, y + 38, (235, 245, 235), None, 0, 8); T(660, y - 8, f"+%{200 + i * 120}", 28, GREEN, True)
            T(660, y + 20, "Bu ay ben de kazandım!", 14, GRAY, False)
        T(1000, 200, "Kayıplar paylaşılmaz", 28, RED, True); T(1000, 250, "Sadece vitrin!", 28, RED, True)
    elif ci == 4:
        man(170, GY, 'shrug', t, acc='cap', mood='open', sc=1.1)
        for i, s in enumerate(("“Garantili yüksek kazanç!”", "“Kaçırma, sadece bugün!”", "“Kimseye söyleme!”")):
            if ct > i * 2.3:
                R(400, 170 + i * 105, 1180, 250 + i * 105, (255, 235, 235), RED, 4, 18); T(790, 210 + i * 105, s, 32, RED, True)
                POLY([(420, 195 + i * 105), (420, 175 + i * 105), (450, 185 + i * 105)], RED)
                L(418, 175 + i * 105, 418, 240 + i * 105, 4, INK)
    elif ci == 5:
        man(200, GY, 'point', t, acc='glasses', sc=1.2)
        R(480, 190, 900, 400, WHITE, INK, 4, 20); T(690, 260, "GARANTİ", 56, GRAY, True)
        L(520, 380, 860, 210, 10, RED)
        T(940, 300, "Yüksek getiri", 26, INK, True, 'lm'); T(940, 335, "vaadi = yüksek risk", 26, INK, True, 'lm'); T(940, 370, "ya da tuzak", 26, RED, True, 'lm')
    elif ci == 6:
        man(170, GY, 'point', t, acc='glasses', sc=1.1)
        shield(540, 290, 1.7, GREEN); check(540, 290, 2.6, WHITE)
        T(540, 430, "Lisanslı mı?", 28, GREEN, True)
        R(780, 200, 1200, 340, (255, 245, 205), INK, 4, 16); T(990, 250, "Şifre • SMS kodu", 26, INK, True); T(990, 300, "kimseyle paylaşılmaz", 26, RED, True)
        R(940, 400, 1040, 460, GRAY, INK, 4, 8); ARC(960, 360, 1020, 430, 180, 360, 6)
    elif ci == 7:
        man(150, GY, 'point', t, acc='glasses', sc=1.1)
        R(360, 120, 1200, 500, WHITE, INK, 4, 14); T(780, 155, "YATIRIM PLANIM", 28, BLUE, True)
        rows = ["Neden alıyorum?  Uzun vadeli birikim", "Ne kadar?  Maaşın %20'si", "Ne zaman satarım?  Hedef tarihte", "Düşerse?  Panik yok, plana bak"]
        for i, s in enumerate(rows):
            if ct > i * 1.6: T(390, 230 + i * 62, s, 24, INK, False, 'lm'); check(1160, 228 + i * 62, 0.6)
    else:
        for i in range(3): T(580 + i * 45, 190 - i * 32, "Z", 30 + i * 10, GRAY, True)
        R(300, 370, 800, 440, (200, 205, 235), INK, 4, 14); R(300, 340, 328, 500, BROWN, INK, 4); R(772, 380, 800, 500, BROWN, INK, 4)
        C(390, 345, 24, WHITE, INK, 4); ARC(380, 345, 400, 360, 200, 340, 3)
        L(414, 350, 640, 360, 6)
        coin(950, 300, 40); T(950, 380, "Uykunu kaçıran", 24, RED, True); T(950, 410, "para = yanlış yer", 24, RED, True)
        cross(1090, 300, 1.6)


# ---------- ÖZET
@scene("Özet", 8, (222, 248, 236), [
    "Haydi toparlayalım ve yedi kuralı bir arada görelim.",
    "Bir: Paranın nereye gittiğini bil ve önce kendine öde.",
    "İki: Acil durum fonunu kur; üç ila altı aylık gider.",
    "Üç: Zamanı arkana al. Bugün başla, düzenli devam et.",
    "Dört: Enflasyonu unutma; reel getiriye bak.",
    "Beş: Pahalı borçtan kurtul, yenisini alma.",
    "Altı: Riski dağıt ve anlamadığına girme.",
    "Yedi: Duygularına ve garantili kazanç hikâyelerine karşı planlı ol.",
    "Bugünkü ödevin: Bu hafta tüm harcamalarını yaz, bir birikim hedefi belirle ve maaş gününe otomatik birikim talimatı ver.",
    "Unutma: Zengin olmak bir günde olmaz, ama bir günde başlanır. Kendine iyi bak, parana daha iyi bak. Görüşmek üzere!",
    "Bu video eğitim amaçlıdır ve yatırım tavsiyesi değildir. Karar vermeden önce kendi araştırmanı yap veya lisanslı bir uzmana danış.",
])
def s_summary(ci, ct, t):
    items = ["Paranın nereye gittiğini bil", "Acil durum fonunu kur", "Zamanı arkana al", "Enflasyonu hesaba kat",
             "Kötü borçtan kurtul", "Riski dağıt", "Duygularını yönet"]
    man(170, GY, 'cheer' if ci >= 8 else 'wave', t, acc='cap', sc=1.3, name='CAN')
    if ci < 10:
        for i, s in enumerate(items):
            if ci >= i + 1:
                y = 125 + i * 56
                R(380, y - 24, 1180, y + 24, WHITE, GREEN, 3, 14)
                C(415, y, 17, GREEN); T(415, y, str(i + 1), 20, WHITE, True); T(450, y, s, 26, INK, True, 'lm'); check(1140, y, 0.7)
    if ci >= 8:
        for i in range(14):
            q = (ct * 0.3 + i / 14) % 1
            C(100 + i * 85, 60 + q * 380, 6, [RED, BLUE, GOLD, GREEN][i % 4])
    if ci == 10:
        R(380, 110, 1180, 470, (255, 247, 215), INK, 4, 20)
        T(780, 190, "EĞİTİM AMAÇLIDIR", 40, RED, True); T(780, 255, "Yatırım tavsiyesi değildir.", 32, INK, True)
        T(780, 330, "İzlediğin için teşekkürler!", 34, BLUE, True); T(780, 390, "Beğen • Paylaş • Planını yap", 26, GRAY, True)


# ---------------------------------------------------------------- zaman çizelgesi
def cap_dur(s):
    return max(4.5, len(s.split()) / 2.0 + 1.6)


TITLE_DUR = 4.0
timeline = []  # (start, end, scene_idx, kind, ci)
tcur = 0.0
for si, sc in enumerate(SCENES):
    if sc['num'] > 0:
        timeline.append((tcur, tcur + TITLE_DUR, si, 'title', 0)); tcur += TITLE_DUR
    for ci, c in enumerate(sc['caps']):
        d = cap_dur(c); timeline.append((tcur, tcur + d, si, 'cap', ci)); tcur += d
TOTAL = tcur
NFR = int(TOTAL * FPS)


def wrap(s, f, maxw):
    words = s.split(); lines = []; cur = ""
    for w_ in words:
        tr = (cur + " " + w_).strip()
        if f.getlength(tr) <= maxw * S: cur = tr
        else: lines.append(cur); cur = w_
    lines.append(cur)
    return lines


def render(fi):
    global D
    t = fi / FPS
    ent = timeline[-1]
    for e in timeline:
        if e[0] <= t < e[1]: ent = e; break
    st, en, si, kind, ci = ent
    sc = SCENES[si]
    img = Image.new("RGB", (W * S, H * S), sc['bg']); D = ImageDraw.Draw(img)
    L(0, GY + 12, W, GY + 12, 3, (120, 125, 140))
    ct = t - st
    if kind == 'title':
        a = prog(ct, 0, 0.6)
        T(W / 2, 95, f"KURAL {sc['num']}" if sc['num'] < 8 else "SONUÇ", 40, GRAY, True)
        T(W / 2, 165, sc['title'], 48, BLUE, True)
        man(W / 2, GY, 'wave', ct, acc='cap', sc=1.15 * (0.6 + 0.4 * a), name='CAN')
        for i in range(8): coin(200 + i * 120, 400 + 14 * math.sin(ct * 3 + i), 16)
    else:
        sc['fn'](ci, ct, t)
        if sc['num'] > 0:
            tag = (f"KURAL {sc['num']} • " if sc['num'] < 8 else "") + sc['title']
            R(30, 24, 70 + len(tag) * 11.5, 62, (255, 255, 255), None, 0, 14)
            T(48, 43, tag, 18, BLUE, True, 'lm')
        R(0, 578, W, H, (24, 28, 44)); f = font(29, True)
        lines = wrap(sc['caps'][ci], f, 1130)
        y0 = 647 - (len(lines) - 1) * 21
        for k, ln in enumerate(lines):
            T(W / 2, y0 + k * 42, ln, 29, WHITE, True)
    R(0, H - 5, W * t / TOTAL, H, ORANGE)
    out = img.reduce(S)
    if ct < 0.35:
        out = Image.blend(Image.new("RGB", (W, H), WHITE), out, clamp(ct / 0.35) * 0.9 + 0.1)
    return out


def worker(args):
    wi, f0, f1, outp = args
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", outp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(render(fi).tobytes())
    p.stdin.close(); p.wait()
    return outp


# ---------------------------------------------------------------- müzik
def make_music(path, dur, sr=44100):
    import wave
    n = int(dur * sr); out = np.zeros(n, dtype=np.float32)
    chords = [(261.63, 329.63, 392.0), (220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66)]
    bass = [65.41, 55.0, 43.65, 49.0]
    bar = 2.0
    t_all = np.arange(int(bar * sr)) / sr
    for b in range(int(dur / bar) + 1):
        ci = b % 4; s0 = int(b * bar * sr)
        if s0 >= n: break
        seg_ = np.zeros(len(t_all), dtype=np.float32)
        env = np.minimum(1, t_all / 0.4) * np.minimum(1, (bar - t_all) / 0.4)
        for f in chords[ci]:
            seg_ += 0.05 * (np.sin(2 * np.pi * f * t_all) + 0.5 * np.sin(2 * np.pi * f * 1.004 * t_all)) * env
        seg_ += 0.10 * np.sin(2 * np.pi * bass[ci] * 2 * t_all) * env
        for k in range(8):
            f = chords[ci][k % 3] * (2 if k % 4 == 3 else 1) * 2
            st = int(k * 0.25 * sr); ln = int(0.5 * sr)
            tt = np.arange(ln) / sr
            note = 0.07 * np.sin(2 * np.pi * f * tt) * np.exp(-5 * tt)
            e = min(len(seg_), st + ln)
            seg_[st:e] += note[:e - st]
        e = min(n, s0 + len(seg_)); out[s0:e] += seg_[:e - s0]
    idx = np.arange(n)
    out *= np.minimum(1, idx / (sr * 3)) * np.minimum(1, (n - idx) / (sr * 4))
    out = out / (np.abs(out).max() + 1e-9) * 0.5
    pcm = (np.stack([out, out], 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())


def main():
    from multiprocessing import Pool
    outfile = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "finans_stickman.mp4")
    tmp = os.path.join(os.path.dirname(outfile), "_tmp_render")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    print(f"Toplam süre: {TOTAL:.1f} sn ({TOTAL / 60:.1f} dk), kare: {NFR}", flush=True)
    nw = os.cpu_count() or 4; step = math.ceil(NFR / (nw * 4))
    jobs = [(i, f0, min(NFR, f0 + step), os.path.join(tmp, f"seg{i:03d}.mp4")) for i, f0 in enumerate(range(0, NFR, step))]
    with Pool(nw) as pool:
        for k, r in enumerate(pool.imap(worker, jobs)):
            print(f"  parça {k + 1}/{len(jobs)} tamam", flush=True)
    with open(os.path.join(tmp, "list.txt"), "w") as f:
        for j in jobs: f.write(f"file '{j[3]}'\n")
    wav = os.path.join(tmp, "music.wav"); make_music(wav, TOTAL + 1)
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"), "-i", wav,
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-t", f"{TOTAL:.2f}", "-movflags", "+faststart", outfile])
    shutil.rmtree(tmp, ignore_errors=True)
    print("Hazır:", outfile)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--frames":
        os.makedirs(sys.argv[2], exist_ok=True)
        for s in sys.argv[3:]:
            render(int(float(s) * FPS)).save(os.path.join(sys.argv[2], f"f_{float(s):07.1f}.png"))
        print("toplam", TOTAL)
    else:
        main()
