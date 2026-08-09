import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for theme in ['Theme-1', 'Theme-3', 'Theme-5', 'Theme-7']:
    json_file = Path('outputs') / theme / f'{theme}_extracted.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    blocks = data.get('text_blocks', [])
    empty = [b for b in blocks if b.get('type') != 'blank_page' and not b.get('text','').strip()]
    print(f'{theme}: {len(empty)} empty non-blank_page blocks')
    for b in empty[:5]:
        btype = b.get('type', 'N/A')
        page = b.get('page_num', '?')
        text_repr = repr(b.get('text', ''))
        bp = b.get('is_boilerplate', False)
        print(f'  type={btype}  page={page}  text={text_repr}  boilerplate={bp}')
    print()
