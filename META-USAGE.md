## Використання ШІ.
## Інструменти
Claude (Anthropic), claude.ai.
## Використовувала для пошуку джерел основної вибірки.
## Допомога у вирішенні проблем з кодом, частичне виправлення.
## Також використовувала для написання тестів test_methods.py, зокрема методи CLAHE, Оцу і Sauvola, test_block_stats_by_hand(), 
У test_pipeline.py. Мала запити для пояснення основної суті пайплайнів, а також я брала такий код тестів за основу до своїх: import requests
from typing import Any, Dict, List
import logging
import sys

def get_exchange_data() -> List[Dict[str, Any]]:
    url = "https://api.coincap.io/v2/exchanges" # Note this API is not longer active
    try:
        r = requests.get(url)
    except requests.ConnectionError as ce:
        logging.error(f"There was an error with the request, {ce}")
        sys.exit(1)
    return r.json().get("data", [])
##Згенеровані методи для розбору roi.py, sauvola.py, otsu.py, pipelines.py(частково)

## Першу структуру з файлами без коду згенерував ШІ, далі для цих файлів я писала код.