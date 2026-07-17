"""Unit tests for ligand-node bookkeeping without a Schrödinger install."""

import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch


def _load_ligand_node():
    schrodinger = types.ModuleType("schrodinger")
    structutils = types.ModuleType("schrodinger.structutils")
    structutils.interactions = types.SimpleNamespace()
    structutils.measure = types.SimpleNamespace()
    structutils.analyze = types.SimpleNamespace()
    schrodinger.structutils = structutils

    molecule_formats = types.ModuleType("src.utils.molecule_formats")
    molecule_formats.struc_to_smiles = lambda structure: ""
    struc_tools = types.ModuleType("src.utils.struc_tools")
    struc_tools.get_bonded_indices = lambda structure, atom_idx: structure.bonds[atom_idx]
    struc_tools.write_mae = lambda *args, **kwargs: None
    struc_tools.num_heavy_atoms = lambda structure: 0

    modules = {
        "schrodinger": schrodinger,
        "schrodinger.structutils": structutils,
        "src.utils.molecule_formats": molecule_formats,
        "src.utils.struc_tools": struc_tools,
    }
    with patch.dict(sys.modules, modules):
        sys.modules.pop("src.frag_adder.ligand_node", None)
        return importlib.import_module("src.frag_adder.ligand_node")


class LigandNodeBranchpointTests(unittest.TestCase):
    def test_branchpoint_uses_fragment_index_not_bond_list_order(self):
        ligand_node = _load_ligand_node()
        ligand = types.SimpleNamespace(
            atom_total=30,
            total_weight=0,
            # The core neighbor appears last, as can happen after attachment.
            bonds={23: [30, 22]},
        )
        node = ligand_node.LigandNode(
            protein=None, ligand=ligand, parent_ligand_size=30,
            attachment_atom_idx=23,
        )

        self.assertEqual(node.get_branchpoint_atom(), 30)


class CustomFragmentAttachmentTests(unittest.TestCase):
    def test_custom_fragment_attachment_uses_indices_from_copied_ligand(self):
        schrodinger = types.ModuleType("schrodinger")
        structutils = types.ModuleType("schrodinger.structutils")
        structutils.build = types.SimpleNamespace(attach_structure=Mock(return_value={24: 23, 23: None}))
        structutils.interactions = types.SimpleNamespace()
        schrodinger.structutils = structutils

        fragment_config = types.ModuleType("src.utils.fragment_info.simple_fragment_config")
        fragment_config.fragment_info = {}
        ligand_node = types.ModuleType("src.frag_adder.ligand_node")
        ligand_node.LigandNode = Mock()
        struc_tools = types.ModuleType("src.utils.struc_tools")
        struc_tools.get_bonded_indices = lambda *args: []
        build_molecules = types.ModuleType("src.utils.build_molecules")
        build_molecules.Fragment_Attachment_Set = object

        modules = {
            "schrodinger": schrodinger,
            "schrodinger.structutils": structutils,
            "src.utils.fragment_info.simple_fragment_config": fragment_config,
            "src.frag_adder.ligand_node": ligand_node,
            "src.utils.struc_tools": struc_tools,
            "src.utils.build_molecules": build_molecules,
        }
        with patch.dict(sys.modules, modules):
            sys.modules.pop("src.frag_adder.base_adder", None)
            base_adder = importlib.import_module("src.frag_adder.base_adder")

            class TestAdder(base_adder.FragmentAdder):
                def run(self):
                    return None

                def open_bond_scorefxn(self, current, goal):
                    return 0

                def fragment_scorefxn(self, current, goal):
                    return 0

            adder = TestAdder({"fragname_list": [], "debug": False})
            fragment_atom = types.SimpleNamespace(
                bonded_atoms=[types.SimpleNamespace(index=2)])
            fragment = types.SimpleNamespace(atom={7: fragment_atom})
            adder.frag_set = types.SimpleNamespace(
                get_fragment=lambda name: fragment,
                get_attachment_id=lambda name: 7,
                is_tricky=lambda name, h_id: False,
            )
            ligand = types.SimpleNamespace(copy=lambda: "copied ligand")
            open_bond = (
                types.SimpleNamespace(index=24, atomic_number=6),
                types.SimpleNamespace(index=23, atomic_number=1),
            )

            adder.add_fragment(ligand, open_bond, "R0:7")

        structutils.build.attach_structure.assert_called_once_with(
            "copied ligand", 24, 23, fragment, 2, 7)


if __name__ == "__main__":
    unittest.main()
