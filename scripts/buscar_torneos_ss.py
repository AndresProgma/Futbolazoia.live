"""Busca IDs de torneos de selecciones en SofaScore."""
import sys
sys.path.insert(0, '.')
from scraper.scraper_premier import SofaScoreSession

queries = ['nations+league', 'world+cup', 'copa+america', 'african+cup+of+nations', 'asian+cup', 'gold+cup', 'euro+championship']

sess = SofaScoreSession()
try:
    for q in queries:
        r = sess.api('/search/all?q=' + q)
        if not r:
            print(f'\n--- {q}: sin respuesta ---')
            continue
        torns = r.get('uniqueTournaments', {}).get('results', [])[:6]
        print(f'\n--- {q} ---')
        for t in torns:
            cat = t.get('category', {}).get('name', '')
            print(f'  id={t["id"]:6}  {t["name"][:45]:45}  cat={cat}')
finally:
    sess.close()
