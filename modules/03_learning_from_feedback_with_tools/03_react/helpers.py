"""ReAct / WebShop-style teaching helpers.

The main experiment uses a synthetic catalog and a *mini-simulator*, NOT the
original WebShop benchmark. This module never makes any real purchases.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd

_ACTION = re.compile(r'(?im)^\s*Action(?:\s+\d+)?\s*:\s*(search|click)\[(.*?)\]\s*$')
_SEARCH_STOP = set('a an the i me with for in on and buy find size want to'.split())


def load_fixtures(directory='data'):
    """Load visible catalog and hidden evaluation targets from JSON files."""
    folder = Path(directory)
    with (folder / 'mini_products.json').open(encoding='utf8') as file:
        products = json.load(file)
    with (folder / 'mini_goals.json').open(encoding='utf8') as file:
        goals = json.load(file)
    assert all({'id', 'title', 'price', 'options'} <= p.keys() for p in products)
    assert all({'id', 'instruction', 'target_id', 'required'} <= g.keys() for g in goals)
    return products, goals


def catalog_table(products):
    return pd.DataFrame([{'id': p['id'], 'title': p['title'],
                          'price': p['price'], 'options': str(p['options'])}
                         for p in products])


def goal_table(goals):
    """Agent-visible portion only; no target IDs or gold attribute labels."""
    return pd.DataFrame([{'id': goal['id'], 'instruction': goal['instruction']}
                         for goal in goals])


def words(text):
    return re.findall(r'[a-z0-9]+', str(text).lower())


class MiniWebShop:
    """Observable state-machine: home → results → item → simulated purchase."""
    def __init__(self, products, results_limit=5):
        self.products = {p['id']: p for p in products}
        self.results_limit = results_limit
        self.done = True

    def reset(self, goal):
        self.goal = goal
        self.page, self.current, self.results = 'home', None, []
        self.selected, self.history = {}, []
        self.done, self.steps, self.invalid = False, 0, 0
        return 'Shopping home. Instruction: ' + goal['instruction'] + '\nAvailable: search[query]'

    def observation(self):
        if self.page == 'home':
            return 'Shopping home. Available: search[query]'
        if self.page == 'results':
            lines = ['Search results. Click a bracketed product ID:']
            lines += [f'[{pid}] {self.products[pid]["title"]} | ${self.products[pid]["price"]:.2f}'
                      for pid in self.results]
            return '\n'.join(lines + ['[Back to Search]'])
        if self.page == 'item':
            product = self.products[self.current]
            options = '\n'.join(f'{name}: ' + ' '.join(f'[{value}]' for value in values)
                                for name, values in product['options'].items())
            return (f'Item [{product["id"]}] {product["title"]} | ${product["price"]:.2f}\n'
                    f'{product["description"]}\n{options}\nSelected: {self.selected}\n'
                    '[Description] [Buy Now] [Back to Search]')
        return 'Episode completed.'

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode ended. Call reset before another task.')
        self.steps += 1
        match = re.fullmatch(r'\s*(search|click)\[(.*?)\]\s*', str(action), flags=re.I | re.S)
        reward, done, valid = 0.0, False, True
        if not match:
            message, valid = 'Invalid syntax. Use search[...] or click[...].', False
        else:
            verb, label = match.group(1).lower(), match.group(2).strip()
            if verb == 'search':
                if self.page not in {'home', 'results'}:
                    message, valid = 'Search unavailable. Click Back to Search first.', False
                else:
                    query_words = set(words(label))
                    ranked = sorted(self.products.values(),
                                    key=lambda p: (-len(query_words & set(words(p['title'] + ' ' + p['description']))), p['id']))
                    self.results = [p['id'] for p in ranked
                                    if query_words & set(words(p['title'] + ' ' + p['description']))][:self.results_limit]
                    self.page = 'results'
                    message = self.observation() if self.results else 'No matching products. Try a new search.'
            elif label.lower() == 'back to search' and self.page in {'results', 'item'}:
                self.page, self.current, self.results, self.selected = 'home', None, [], {}
                message = self.observation()
            elif self.page == 'results' and label in self.results:
                self.page, self.current, self.selected = 'item', label, {}
                message = self.observation()
            elif self.page == 'item' and label.lower() == 'description':
                message = self.observation()
            elif self.page == 'item' and label.lower() == 'buy now':
                product = self.products[self.current]
                reward = float(product['id'] == self.goal['target_id'] and all(
                    self.selected.get(k) == v for k, v in self.goal['required'].items()))
                self.done, self.page, done = True, 'done', True
                message = f'Purchase simulated; episode finished. Reward: {reward:.1f}'
            elif self.page == 'item':
                product = self.products[self.current]
                option = next(((k, v) for k, vals in product['options'].items() for v in vals
                               if v.casefold() == label.casefold()), None)
                if option:
                    self.selected[option[0]] = option[1]
                    message = 'Selected ' + option[0] + ': ' + option[1] + '\n' + self.observation()
                else:
                    message, valid = 'Not a clickable option on this page.', False
            else:
                message, valid = 'Action not allowed in current state.', False
        if not valid:
            self.invalid += 1
        self.history.append({'action': action, 'observation': message,
                             'valid': valid, 'reward': reward})
        return message, reward, done, {'valid': valid, 'page': self.page}


def query_from_instruction(instruction):
    return ' '.join(token for token in words(instruction) if token not in _SEARCH_STOP)


def greedy_episode(goal, products, results_limit=5):
    """Weak baseline: first search hit + first option values. NO target-label access."""
    env = MiniWebShop(products, results_limit)
    env.reset(goal)
    started = time.perf_counter()
    actions = [f'search[{query_from_instruction(goal["instruction"])}]']
    env.step(actions[0])
    if env.results:
        product_id = env.results[0]
        actions.append(f'click[{product_id}]')
        actions += [f'click[{values[0]}]' for values in env.products[product_id]['options'].values()]
        actions.append('click[Buy Now]')
        for action in actions[1:]:
            env.step(action)
            if env.done:
                break
    return {'reward': env.history[-1]['reward'] if env.done else 0.0,
            'status': 'finished' if env.done else 'not_finished',
            'steps': env.steps, 'invalid_actions': env.invalid,
            'purchased_id': env.current, 'selection': dict(env.selected),
            'latency_s': time.perf_counter() - started,
            'tokens': None, 'trace': env.history}


def parse_action(raw):
    match = _ACTION.search(str(raw))
    return f'{match.group(1).lower()}[{match.group(2).strip()}]' if match else None


def _api_key(provider):
    name = {'groq': 'GROQ_API_KEY', 'openai': 'OPENAI_API_KEY'}.get(provider)
    if not name:
        raise ValueError('provider must be groq or openai')
    key = os.getenv(name)
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get(name)
        except Exception:
            key = None
    if not key:
        raise RuntimeError(f'Provide {name} via Colab Secrets or environment variables')
    return key


def build_client(provider='groq'):
    from openai import OpenAI
    key = _api_key(provider)
    if provider == 'groq':
        return OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1', timeout=45.0)
    return OpenAI(api_key=key, timeout=45.0)


def ask_model(client, system, prompt, cfg):
    start, last_error = time.perf_counter(), None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=cfg['model'], temperature=cfg['temperature'],
                max_tokens=cfg['max_output_tokens'],
                messages=[{'role': 'system', 'content': system},
                          {'role': 'user', 'content': prompt}],
            )
            usage = getattr(response, 'usage', None)
            used = getattr(usage, 'total_tokens', None) if usage else None
            return (response.choices[0].message.content or '').strip(), used, time.perf_counter() - start
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'Hosted-model request failed after retries: {last_error}')


def _agent_system(mode):
    common = ('You are an agent operating a small SIMULATED shopping store. '
              'Only observations describe available products and clickable options. '
              'Use exactly one Action per turn: search[words] or click[visible label]. '
              'Search on home/results pages, click listed product IDs, choose requested options, '
              'and click[Buy Now] only when sure. Use click[Back to Search] to return. '
              'Never claim a click succeeded without an observation. No real purchase occurs. ')
    if mode == 'ReAct':
        return common + 'You may include one short Thought: line stating your next plan before Action.'
    if mode == 'Act':
        return common + 'Output one Action line only; no Thought or explanation.'
    raise ValueError(f'Unknown method: {mode}')


def model_episode(goal, mode, client, cfg, products):
    env = MiniWebShop(products, cfg['results_limit'])
    observation = env.reset(goal)
    history = 'Instruction: ' + goal['instruction'] + '\nObservation: ' + observation
    total_tokens, usage_seen, seconds, status = 0, True, 0.0, 'step_limit'
    trace = []
    for step in range(1, cfg['max_steps'] + 1):
        raw, used, latency = ask_model(client, _agent_system(mode), history + '\nNext action:', cfg)
        seconds += latency
        if used is None:
            usage_seen = False
        else:
            total_tokens += used
        action = parse_action(raw)
        if action:
            obs, reward, done, info = env.step(action)
            valid = info['valid']
        else:
            # Invalid formatting consumes the action budget, just like a bad click.
            obs, reward, done, valid = 'Invalid syntax. Specify one Action: search[...] or click[...].', 0.0, False, False
            env.invalid += 1
            env.steps += 1
        obs = obs[:cfg['observation_chars']]
        trace.append({'step': step, 'model_output': raw, 'action': action,
                      'observation': obs, 'valid': valid})
        history += '\n' + raw + '\nObservation: ' + obs
        if done:
            status = 'finished'
            break
    return {'reward': env.history[-1]['reward'] if env.done else 0.0,
            'status': status, 'steps': env.steps, 'invalid_actions': env.invalid,
            'purchased_id': env.current, 'selection': dict(env.selected),
            'latency_s': seconds, 'tokens': total_tokens if usage_seen else None,
            'trace': trace}


def evaluate(goals, products, client=None, cfg=None):
    """Always run Greedy; add Act/ReAct only when client was explicitly supplied."""
    rows = []
    for goal in goals:
        row = greedy_episode(goal, products, cfg['results_limit'] if cfg else 5)
        rows.append({'id': goal['id'], 'instruction': goal['instruction'],
                     'method': 'Greedy', 'success': row['reward'], **{
                         k: v for k, v in row.items() if k != 'reward'}})
    if client is None:
        return pd.DataFrame(rows)
    for goal in goals:
        for mode in ('Act', 'ReAct'):
            print(f'{goal["id"]}: {mode}')
            try:
                result = model_episode(goal, mode, client, cfg, products)
                rows.append({'id': goal['id'], 'instruction': goal['instruction'],
                             'method': mode, 'success': result['reward'], **{
                                 k: v for k, v in result.items() if k != 'reward'}})
            except Exception as error:
                print('  MODEL ERROR:', error)
                rows.append({'id': goal['id'], 'instruction': goal['instruction'],
                             'method': mode, 'status': 'api_error', 'error': str(error),
                             'success': None})
    return pd.DataFrame(rows)



def model_results(goals, products, methods, client, cfg):
    """Evaluate only requested model policies; baseline is run separately."""
    rows = []
    for goal in goals:
        for method in methods:
            print(f'{goal["id"]}: {method}')
            try:
                run = model_episode(goal, method, client, cfg, products)
                rows.append({'id': goal['id'], 'instruction': goal['instruction'],
                             'method': method, 'success': run['reward'], **{
                                 k: v for k, v in run.items() if k != 'reward'}})
            except Exception as error:
                print('  MODEL ERROR:', error)
                rows.append({'id': goal['id'], 'instruction': goal['instruction'],
                             'method': method, 'status': 'api_error', 'error': str(error),
                             'success': None})
    return pd.DataFrame(rows)


def summarize(results):
    valid = results[results['status'] != 'api_error']
    return valid.groupby('method', as_index=False).agg(
        n=('success', 'count'), success_rate=('success', 'mean'),
        avg_steps=('steps', 'mean'), avg_invalid_actions=('invalid_actions', 'mean'),
        avg_tokens=('tokens', 'mean'), avg_latency_s=('latency_s', 'mean'),
    )


def save_results(results, folder='react_webshop_results'):
    output = Path(folder)
    output.mkdir(exist_ok=True, parents=True)
    results.drop(columns=['trace'], errors='ignore').to_csv(output / 'scores.csv', index=False)
    with (output / 'trajectories.jsonl').open('w', encoding='utf8') as stream:
        for item in results.to_dict('records'):
            stream.write(json.dumps(item, default=str, ensure_ascii=False) + '\n')
    return output


def sample_real_webshop_products():
    """Optional exploration of a third-party mirror of WebShop's 1k preview."""
    from huggingface_hub import hf_hub_download
    source = hf_hub_download(repo_id='YWZBrandon/webshop-data', repo_type='dataset',
                             filename='items_shuffle_1000.json')
    return json.loads(Path(source).read_text(encoding='utf8'))
