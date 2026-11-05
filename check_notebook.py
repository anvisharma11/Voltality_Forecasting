import json

nb = json.load(open('main.ipynb', encoding='utf-8'))
cells = nb['cells']
print('Total cells:', len(cells))
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    print(f"Cell {i}: [{c['cell_type']}] {src[:100].strip().replace(chr(10), ' ')}")
