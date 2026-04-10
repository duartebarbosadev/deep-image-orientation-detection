import ast
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PREDICTION_SCRIPTS = [
    ROOT_DIR / "predict.py",
    ROOT_DIR / "predict_batch.py",
    ROOT_DIR / "predict_onnx.py",
    ROOT_DIR / "predict_onnx_batch.py",
]


def collect_imported_modules(script_path: Path) -> set[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    return modules


class PredictionScriptStandaloneTests(unittest.TestCase):
    def test_prediction_scripts_do_not_import_training_modules(self):
        for script_path in PREDICTION_SCRIPTS:
            imported_modules = collect_imported_modules(script_path)

            disallowed = {
                module
                for module in imported_modules
                if module == "train"
                or module.startswith("train.")
                or module == "src"
                or module.startswith("src.")
            }

            self.assertFalse(
                disallowed,
                f"{script_path.name} imports internal training modules: {sorted(disallowed)}",
            )


if __name__ == "__main__":
    unittest.main()
