import argparse
import json
from datetime import datetime
from pathlib import Path

import utils.chains_lcel as chains
import utils.llm_models as llms


EVAL_CASES = [
    {
        "response_mode": "Direct answer",
        "query": "What is overfitting in regression? Keep it practical.",
        "must_include": ["**Answer**", "Check yourself"],
    },
    {
        "response_mode": "Teach me step-by-step",
        "query": "I need help interpreting model output.",
        "must_include": ["**Step 1**"],
    },
]


def run_eval(output_path: Path):
    chain = chains.class_chain(llms.openai_gpt56_luna)
    results = []

    for idx, case in enumerate(EVAL_CASES, start=1):
        payload = {
            "query": case["query"],
            "chat_history": "Student: I am learning regression.\nAssistant: Great, let's build intuition.",
            "response_mode": case["response_mode"],
            "learning_objective": "Regression modeling and interpretation",
            "learner_level": "novice",
            "attempt_check": "no",
        }
        response_text = chain.invoke(payload)
        missing = [marker for marker in case["must_include"] if marker not in response_text]
        results.append(
            {
                "case_id": idx,
                "response_mode": case["response_mode"],
                "query": case["query"],
                "passed": len(missing) == 0,
                "missing_markers": missing,
                "response_preview": response_text[:350],
            }
        )

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_cases": len(results),
        "passed_cases": sum(1 for item in results if item["passed"]),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote prompt style eval report to {output_path}")
    print(f"Passed {report['passed_cases']}/{report['total_cases']} cases")


def main():
    parser = argparse.ArgumentParser(description="Evaluate response-style prompt behavior.")
    parser.add_argument("--output", default="analytics/prompt_style_eval.json")
    args = parser.parse_args()
    run_eval(Path(args.output))


if __name__ == "__main__":
    main()
