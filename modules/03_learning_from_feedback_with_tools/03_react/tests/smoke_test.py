"""Run with `python tests/smoke_test.py`. No API key or internet required."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat

shop_dir = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


shop = module('test_webshop_helpers', shop_dir / 'helpers.py')

products, goals = shop.load_fixtures(shop_dir / 'data')
assert len(products) == 9 and len(goals) == 6
assert len(shop.catalog_table(products)) == 9
assert 'target_id' not in shop.goal_table(goals)
assert shop.parse_action('Thought: inspect\nAction: click[D100]') == 'click[D100]'
assert shop.parse_action('buy anything') is None

env = shop.MiniWebShop(products)
env.reset(goals[0])
for action in ['search[citrus sensitive skin deodorant]', 'click[D100]', 'click[bright citrus]', 'click[3 oz]']:
    obs, reward, done, info = env.step(action)
    assert info['valid']
obs, reward, done, _ = env.step('click[Buy Now]')
assert done and reward == 1.0

bad = shop.MiniWebShop(products)
bad.reset(goals[0])
for action in ['search[citrus sensitive skin deodorant]', 'click[D100]', 'click[calming lavender]', 'click[3 oz]']:
    bad.step(action)
assert bad.step('click[Buy Now]')[1] == 0.0

baseline_df = shop.evaluate(goals, products, cfg={'results_limit': 5})
assert len(baseline_df) == len(goals) and shop.summarize(baseline_df)['n'].sum() == len(goals)

original = shop.ask_model
actions = iter(['Action: search[citrus sensitive skin deodorant]', 'Action: click[D100]',
               'Action: click[bright citrus]', 'Action: click[3 oz]', 'Action: click[Buy Now]'])
shop.ask_model = lambda client, system, prompt, cfg: (next(actions), 10, 0.01)
try:
    r = shop.model_episode(goals[0], 'ReAct', None,
                           {'max_steps': 7, 'results_limit': 5, 'observation_chars': 1300}, products)
finally:
    shop.ask_model = original
assert r['reward'] == 1.0 and r['steps'] == 5

notebook_path = shop_dir / '03_react_webshop.ipynb'
nb = nbformat.read(notebook_path, as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type == 'code' and not cell.source.startswith('%pip'):
        compile(cell.source, str(notebook_path), 'exec')

print('WebShop notebook and offline helper smoke checks passed.')
