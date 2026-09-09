"""The curriculum taxonomy is derived from course_data/concepts.csv: pills,
topic inference and the `learning_objective` analytics label all read it.
These tests run against the real CSV, so they double as a check that the
file still yields a usable pill tree."""

import pytest

from utils import concept_taxonomy as tax
from utils.chains_lcel import (
    curriculum_topics,
    format_topic_focus,
    get_subtopics,
    infer_curriculum_topic,
    infer_learning_objective,
    infer_topic_from_history,
    is_new_question,
)


# ---- pills -----------------------------------------------------------------
def test_outline_is_modules_in_teaching_order_with_written_topics():
    out = tax.outline()
    assert [m["id"] for m in out][:4] == [
        "describing-one-variable", "describing-two-variables", "hypothesis-testing", "simple-regression",
    ]
    by_id = {m["id"]: m for m in out}
    assert [t["label"] for t in by_id["simple-regression"]["topics"]] == [
        "p-values", "Slope", "R-squared and model fit",
    ]
    # Three central-tendency concepts collapse into one pill.
    central = next(t for t in by_id["describing-one-variable"]["topics"] if t["label"] == "Central tendency")
    assert central["n_concepts"] == 3


def test_curriculum_topics_are_module_labels():
    topics = curriculum_topics()
    assert topics[0] == "Describing one variable"
    assert "Simple regression" in topics and "Sensitivity analysis" in topics
    # Nothing the index cannot answer from.
    assert "Probability" not in topics and "SAS JMP / Excel workflows" not in topics


def test_subtopics_resolve_by_label_or_id():
    assert get_subtopics("Simple regression") == ["p-values", "Slope", "R-squared and model fit"]
    assert tax.subtopics("simple-regression") == get_subtopics("Simple regression")
    assert get_subtopics("Hypothesis testing") == ["Hypothesis testing"]
    assert get_subtopics("Probability") == []


def test_topic_labels_read_as_labels():
    # The lint in vat-research enforces this at build time; here it guards the
    # checked-in file. A hyphenated id-looking topic becomes an ugly pill.
    for module in tax.outline():
        for topic in module["topics"]:
            label = topic["label"]
            assert " " in label or "-" not in label or label[1] == "-", label
            assert label[0].isupper() or label.startswith(("p-", "z-", "t-")), label


def test_split_focus_recovers_the_module_from_a_pill_focus():
    assert format_topic_focus("Simple regression", "Slope") == "Simple regression: Slope"
    assert tax.split_focus("Simple regression: Slope") == ("simple-regression", "Slope")
    assert tax.split_focus("Simple regression") == ("simple-regression", "Simple regression")
    assert tax.split_focus("Interpreting R-squared") == ("", "Interpreting R-squared")
    assert tax.split_focus("Regression") == ("", "Regression")


def test_taxonomy_reads_follow_the_file(tmp_path):
    path = tmp_path / "concepts.csv"
    path.write_text(
        "id,title,topic,module,body,status\n"
        "a,Alpha thing,Alpha,mod-one,Body text here,core\n"
        "b,Beta thing,Beta,mod-one,,planned\n"
        "c,Gamma thing,Gamma,mod-two,,planned\n",
        encoding="utf-8",
    )
    assert tax.curriculum_topics(path) == ["Mod one"]          # mod-two has nothing written
    assert tax.subtopics("Mod one", path) == ["Alpha"]         # planned rows are not pills
    assert "mod-two — NOT WRITTEN YET" in tax.format_modules_for_prompt(path)


# ---- inference --------------------------------------------------------------
@pytest.mark.parametrize(
    "query, expected",
    [
        ("What does an R-squared of 0.62 mean?", "Simple regression"),
        ("what does this coefficient mean", "Multiple regression"),
        ("How do I interpret the mean and median of salaries?", "Describing one variable"),
        ("Is multicollinearity a problem here?", "Multiple regression"),
        ("How do I run a regression in JMP?", "Simple regression"),
        ("What is a p-value?", "Simple regression"),
        ("is 0.03 significant", "Hypothesis testing"),
        ("what is a z-score", "Describing one variable"),
        ("how do I fold back a decision tree", "Decision basics"),
        ("what is the value of information", "Sensitivity analysis"),
        ("is there an association between gender and churn", "Describing two variables"),
        ("when should I use logistic regression", "Logistic regression"),
        ("When is the next assignment due?", ""),
        ("what is bayes theorem", ""),
        ("hello", ""),
    ],
)
def test_infer_curriculum_topic(query, expected):
    assert infer_curriculum_topic(query) == expected


def test_learning_objective_buckets_software_and_logistics():
    assert infer_learning_objective("how do I install the treeplan add-in") == "JMP / Excel workflows"
    assert infer_learning_objective("when is the deadline") == "Course schedule and due date logistics"
    assert infer_learning_objective("hello there") == "General data and decision analytics reasoning"
    # A concept question that mentions the tool is still about the concept.
    assert infer_learning_objective("what does the R-squared in JMP output mean") == "Simple regression"


def test_topic_from_history_prefers_latest_student_turn(chat_history):
    assert infer_topic_from_history(chat_history) == "Simple regression"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("multicollinearity", False),
        ("type I error", False),
        ("when is A3 due?", True),
        ("what is the midterm about", True),
        # Deliberately literal: only a leading question word counts.
        ("actually what is the midterm about", False),
        ("", False),
    ],
)
def test_is_new_question(text, expected):
    assert is_new_question(text) is expected


def test_index_drift_reports_a_csv_newer_than_the_build(tmp_path):
    csv_path = tmp_path / "concepts.csv"
    csv_path.write_text("id,title,topic,module,body,status\na,T,Topic,mod,Body,core\n", encoding="utf-8")
    prov = tmp_path / "provenance.json"
    assert "no build stamp" in tax.index_drift(csv_path, prov)
    import hashlib, json
    prov.write_text(json.dumps({"sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(), "built_at": "t"}))
    assert tax.index_drift(csv_path, prov) == ""
    csv_path.write_text("id,title,topic,module,body,status\na,T,Topic,renamed,Body,core\n", encoding="utf-8")
    assert "has changed since" in tax.index_drift(csv_path, prov)
