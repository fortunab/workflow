from orchestrate_pipeline import run_orchestration
import json

result = run_orchestration(use_demo=True)
print(json.dumps(result, indent=2, ensure_ascii=False))
