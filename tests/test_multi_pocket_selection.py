"""Run with python -m unittest discover -s tests; no chemistry installation needed."""
import ast
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest


def load_functions(path):
    # Load the real selection/checkpoint functions without importing Schrodinger.
    tree = ast.parse(Path(path).read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(ast=ast, json=json, math=math, os=os, random=random)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


RUNNER = Path(__file__).resolve().parents[1] / 'src/frag_adder/run_multi_pk.py'
FUNCTIONS = load_functions(RUNNER)


class MultiPocketSelectionTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(10)

    def test_open_bonds_ignore_negative_weights_and_deduplicate(self):
        payload = {
            'pk0': [dict(atom_from=1, atom_h=2, score=0), dict(atom_from=3, atom_h=4, score=0)],
            'pk1': [dict(atom_from=1, atom_h=2, score=100), dict(atom_from=3, atom_h=4, score=-100)],
        }
        rows = FUNCTIONS['select_open_bond_across_pockets'](payload, 1, [1, -1], 100)['open_bonds']
        self.assertEqual([row['atom_from'] for row in rows], [3, 1])
        self.assertEqual([row['probability'] for row in rows], [0.5, 0.5])

    def test_fragments_use_best_dihedral_and_require_every_pocket(self):
        payload = {
            'pk0': [dict(fragname='A', score=3, idx=0), dict(fragname='A', score=1, idx=1),
                    dict(fragname='B', score=-100, idx=2)],
            'pk1': [dict(fragname='A', score=4, idx=7)],
        }
        rows = FUNCTIONS['select_fragment_across_pockets'](payload, 1, [2, -1], 20)['fragments']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['score'], -2)
        self.assertEqual(rows[0]['indices_by_pocket'], {'pk0': 1, 'pk1': 7})

    def test_prefilter_uses_positive_weights_before_signed_sampling(self):
        payload = {
            'pk0': [dict(fragname='A', score=0, idx=0), dict(fragname='B', score=10, idx=1)],
            'pk1': [dict(fragname='A', score=0, idx=0), dict(fragname='B', score=100, idx=1)],
        }
        select = FUNCTIONS['select_fragment_across_pockets']
        self.assertEqual(select(payload, 1, [1, -1])['fragments'][0]['fragname'], 'A')
        self.assertEqual(select(payload, 1, [1, -1], cutoff_ratio=1)['fragments'][0]['fragname'], 'B')

    def test_exhausted_branches_are_distinct_from_invalid_arguments(self):
        for name in ['select_open_bond_across_pockets', 'select_fragment_across_pockets']:
            select = FUNCTIONS[name]
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    select({}, 1)
                with self.assertRaises(ValueError):
                    select({'pk0': []}, 1, [1, 2])
                with self.assertRaises(ValueError):
                    select({'pk0': []}, 1, None, 0)
                with self.assertRaises(FUNCTIONS['BranchExhaustedError']):
                    select({'pk0': []}, 1)

    def test_checkpoint_pointer_resolves_beside_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'checkpoint_d2.json'
            manifest.write_text(json.dumps(dict(depth=2, branches=[])))
            pointer = Path(directory) / 'checkpoint_latest.json'
            pointer.write_text(json.dumps(dict(checkpoint=manifest.name, depth=2)))
            resolve = FUNCTIONS['resolve_checkpoint_manifest']
            self.assertEqual(resolve(str(pointer)), resolve(str(manifest)))
            self.assertEqual(resolve(str(pointer))[0], str(manifest))

    def test_random_state_restores_next_draw(self):
        saved = repr(random.getstate())
        expected = random.random()
        random.seed(999)
        FUNCTIONS['load_random_state'](saved)
        self.assertEqual(random.random(), expected)


if __name__ == '__main__':
    unittest.main()
