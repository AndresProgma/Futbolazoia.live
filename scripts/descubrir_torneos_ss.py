"""
Navega Sofascore para encontrar IDs reales de torneos internacionales.
Intercepta las llamadas a la API mientras navega el sitio.
"""
import sys, json, time
sys.path.insert(0, '.')
from playwright.sync_api import sync_playwright

URLS_TORNEOS = [
    ('UEFA Nations League', 'https://www.sofascore.com/tournament/football/europe/uefa-nations-league-a/1304'),
    ('UEFA Nations League', 'https://www.sofascore.com/football/international'),
    ('World Cup',           'https://www.sofascore.com/tournament/football/world/world-cup/1'),
    ('Euro',                'https://www.sofascore.com/tournament/football/europe/euro/1'),
]

pw   = sync_playwright().start()
br   = pw.chromium.launch(headless=True)
ctx  = br.new_context(
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36'
)
page = ctx.new_page()

captured = []

def on_response(resp):
    url = resp.url
    if 'unique-tournament' in url and 'season' in url:
        captured.append(url)

page.on('response', on_response)

# Ir a la página de torneos internacionales de Sofascore
print('Navegando a Sofascore internacional...')
page.goto('https://www.sofascore.com/football/international', wait_until='networkidle', timeout=30000)
time.sleep(3)

# Intentar encontrar links de torneos en la página
links = page.evaluate("""() => {
    const anchors = [...document.querySelectorAll('a[href*="tournament"]')];
    return anchors.slice(0, 30).map(a => ({href: a.href, text: a.textContent.trim().slice(0,50)}));
}""")

print('\n=== Links de torneos encontrados en /football/international ===')
for l in links:
    print(f'  {l["text"]:40} {l["href"]}')

# También interceptar llamadas API que ya se hicieron
print(f'\n=== URLs de API capturadas ({len(captured)}) ===')
for u in captured[:20]:
    print(' ', u)

# Buscar en el JS de la página algún ID de Nations League
content = page.content()
import re
ids_encontrados = re.findall(r'unique-tournament[/:](\d+)', content)
print(f'\n=== IDs de unique-tournament en HTML ===')
print(set(ids_encontrados))

br.close()
pw.stop()
