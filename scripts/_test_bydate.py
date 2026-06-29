import time
from playwright.sync_api import sync_playwright

TOURNAMENT_ID = 16
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

pw = sync_playwright().start()
br = pw.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
ctx = br.new_context(user_agent=UA, locale='es-ES')
pg = ctx.new_page()
pg.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=45000)
time.sleep(3)

def fetch(path):
    js = ("async () => { const r = await fetch("
          f"'https://www.sofascore.com/api/v1{path}',"
          "{headers:{'Accept':'application/json'}}); return r.ok ? r.json() : {_e:r.status}; }")
    return pg.evaluate(js)

total = {}
for d in range(11, 27):
    fecha = f'2026-06-{d:02d}'
    data = fetch(f'/sport/football/scheduled-events/{fecha}')
    if not data or '_e' in data:
        print(fecha, 'ERROR', data)
        continue
    evs = [e for e in data.get('events', [])
           if e.get('tournament',{}).get('uniqueTournament',{}).get('id') == TOURNAMENT_ID]
    fin = [e for e in evs if e.get('status',{}).get('type')=='finished']
    for e in fin:
        total[e['id']] = e
    print(fecha, 'WC eventos:', len(evs), 'finished:', len(fin))
    time.sleep(0.5)
print('TOTAL finished unicos:', len(total))
br.close(); pw.stop()
