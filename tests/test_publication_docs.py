from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DOCS = (
    ROOT / "README.md",
    ROOT / "MODEL_CARD.md",
    ROOT / "FINAL_REPORT.md",
    ROOT / "RESEARCH_PLAN.md",
    ROOT / "LICENSE_AUDIT.md",
    ROOT / "HUMAN_EVALUATION_GUIDE.md",
    ROOT / "LOCAL_USE.md",
    ROOT / "reports" / "final" / "README.md",
)
LOCAL_LINK = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)")


def test_publication_docs_have_no_broken_local_links() -> None:
    missing: list[str] = []
    for document in PUBLICATION_DOCS:
        assert document.is_file(), document
        for raw_target in LOCAL_LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip("<>").split("#", maxsplit=1)[0]
            if target and not (document.parent / target).exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "\n".join(missing)


def test_model_card_declares_the_approved_weight_license() -> None:
    model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert model_card.startswith("---\n")
    assert re.search(r"(?m)^license: apache-2\.0$", model_card)
    assert "Photonyx" in model_card and "CC-BY" in model_card
    assert "LICENSE_AUDIT.md" in model_card


def test_release_package_includes_publication_evidence() -> None:
    package_script = (
        ROOT / "scripts" / "cloud" / "package_final_delivery.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "README.md",
        "FINAL_REPORT.md",
        "RESEARCH_PLAN.md",
        "MODEL_CARD.md",
        "LICENSE_AUDIT.md",
        "HUMAN_EVALUATION_GUIDE.md",
        "LOCAL_USE.md",
        "LICENSE",
    ):
        assert re.search(rf"(?m)^\s+{re.escape(required)}$", package_script)
