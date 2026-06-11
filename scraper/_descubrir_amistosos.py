"""
Descubre el unique-tournament id de amistosos internacionales y su season 2026
sondeando los partidos programados en las ventanas FIFA de 2026.
"""
import sys, os, json
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scraper.scraper_premier import SofaScoreSession

# Ventanas FIFA 2026 (amistosos pre-Mundial) + algunas fechas de marzo
FECHAS = [
    '2026-03-23', '2026-03-25', '2026-03-27', '2026-03-31',
    '2026-06-01', '2026-06-03', '2026-06-05', '2026-06-06',
    '2026-06-07', '2026-06-08', '2026-06-09', '2026-06-10',
]

sess = SofaScoreSession()
torneos = defaultdict(lambda: {'nombre': '', 'category': '', 'seasons': set(), 'n': 0, 'ejemplos': []})
try:
    for fecha in FECHAS:
        data = sess.api(f'/sport/football/scheduled-events/{fecha}')
        if not data:
            continue
        evs = data.get('events', [])
        print(f'{fecha}: {len(evs)} eventos')
        for ev in evs:
            ut = ev.get('tournament', {}).get('uniqueTournament', {}) or {}
            cat = ev.get('tournament', {}).get('category', {}) or {}
            utid = ut.get('id')
            utname = ut.get('name', '')
            catname = cat.get('name', '')
            if not utid:
                continue
            # Interesa amistosos / International
            if 'friendl' in utname.lower() or 'amistos' in utname.lower() or catname.lower() in ('international', 'world'):
                t = torneos[utid]
                t['nombre'] = utname
                t['category'] = catname
                t['n'] += 1
                sid = ev.get('season', {}).get('id')
                if sid:
                    t['seasons'].add(sid)
                if len(t['ejemplos']) < 3:
                    h = ev.get('homeTeam', {}).get('name', '')
                    a = ev.get('awayTeam', {}).get('name', '')
                    st = ev.get('status', {}).get('type', '')
                    t['ejemplos'].append(f'{h} vs {a} [{st}]')
finally:
    sess.close()

print('\n=== TORNEOS CANDIDATOS (amistosos / international) ===')
for utid, t in sorted(torneos.items(), key=lambda x: -x[1]['n']):
    print(f"\nid={utid}  {t['nombre']}  (cat: {t['category']})  partidos={t['n']}")
    print(f"  seasons: {sorted(t['seasons'])}")
    for e in t['ejemplos']:
        print(f"  - {e}")
