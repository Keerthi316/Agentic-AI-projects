"""Verification script for evaluation framework Step 1."""
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["RECRUITMENT_DEMO_MODE"] = "true"

print("=" * 55)
print("  EVALUATION FRAMEWORK — STEP 1 VERIFICATION")
print("=" * 55)

# 1. Raw JSON parse
json_path = os.path.join("evaluation", "datasets", "recruitment_eval_dataset.json")
with open(json_path, encoding="utf-8") as f:
    raw = json.load(f)
task_count = len(raw["tasks"])
print(f"\n[1] JSON parse")
print(f"    tasks found : {task_count}")
assert task_count >= 10, f"Expected >=10 tasks, got {task_count}"
print(f"    [PASS]")

# 2. Pydantic schema validation
from evaluation.datasets.schema import EvalDataset, TaskCategory
dataset = EvalDataset.model_validate(raw)
print(f"\n[2] Schema validation")
print(f"    version     : {dataset.version}")
print(f"    tasks       : {len(dataset.tasks)}")
s = dataset.summary()
for cat, count in s["by_category"].items():
    print(f"      {cat:<25} : {count}")
print(f"    critical    : {s['critical_count']}")
print(f"    [PASS]")

# 3. Loader functions
from evaluation.datasets.loader import load_dataset, load_tasks_by_category, load_task_by_id
ds = load_dataset()
print(f"\n[3] Loader")
print(f"    load_dataset()            : {len(ds.tasks)} tasks")
t = load_task_by_id("task_007")
print(f"    load_task_by_id(task_007) : {t.name}")
inj = load_tasks_by_category("injection_attack")
print(f"    injection_attack tasks    : {len(inj)}")
border = load_tasks_by_category(TaskCategory.BORDERLINE)
print(f"    borderline tasks          : {len(border)}")
print(f"    [PASS]")

# 4. All __init__ imports
import evaluation
import evaluation.datasets
import evaluation.traces
import evaluation.metrics
import evaluation.tests
import evaluation.reports
import evaluation.deepeval_suite
import evaluation.giskard_suite
import evaluation.promptfoo_suite
print(f"\n[4] Package imports")
print(f"    All 9 sub-packages imported successfully")
print(f"    [PASS]")

# 5. conftest path setup (just verify the file exists and is valid Python)
conftest_path = os.path.join("evaluation", "tests", "conftest.py")
assert os.path.exists(conftest_path), "conftest.py missing"
import py_compile
py_compile.compile(conftest_path, doraise=True)
print(f"\n[5] conftest.py")
print(f"    Syntax check   : PASS")
print(f"    File exists    : PASS")
print(f"    [PASS]")

print(f"\n{'=' * 55}")
print(f"  ALL CHECKS PASSED ({task_count} tasks validated)")
print(f"{'=' * 55}\n")
