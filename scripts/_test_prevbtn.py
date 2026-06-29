import time
from playwright.sync_api import sync_playwright

SEASON_ID = 58210
TOURNAMENT_ID = 16
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

pw = sync_playwright().start()
br = pw.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
ctx = br.new_context(user_agent=UA, locale='es-ES')
pg = ctx.new_page()
pg.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=45000)
time.sleep(3)

captured = []
def on_resp(r):
    if '/events/last/' in r.url or '/events/round/' in r.url:
        try: captured.append((r.url, r.json()))
        except Exception: captured.append((r.url, None))
pg.on('response', on_resp)

pg.goto(f"https://www.sofascore.com/tournament/football/world/world-championship/{TOURNAMENT_ID}#id:{SEASON_ID},tab:matches",
        wait_until='domcontentloaded', timeout=45000)
time.sleep(6)

# Buscar botones de navegacion de pagina (flechas prev/next) en la seccion matches
btns = pg.query_selector_all('button')
print('total botones:', len(btns))
clickable = []
for b in btns:
    try:
        al = (b.get_attribute('aria-label') or '').lower()
        txt = (b.inner_text() or '').strip().lower()
        if any(k in al for k in ['previous','anterior','prev']) or any(k in txt for k in ['previous','anterior','mostrar','show']):
            clickable.append((al, txt, b))
    except Exception: pass
print('candidatos prev:', [(a,t) for a,t,_ in clickable])

# clickear el primero varias veces
for a,t,b in clickable[:1]:
    for i in range(6):
        try:
            b.scroll_into_view_if_needed(); b.click(); time.sleep(2.5)
        except Exception as e:
            print('click err', e); break

ids=set()
for u,j in captured:
    if j and 'events' in j:
        for e in j['events']: ids.add(e['id'])
    print('cap', u.split('/api/v1')[-1], '->', len(j.get('events',[])) if j else j)
print('IDs unicos:', len(ids))
br.close(); pw.stop()
