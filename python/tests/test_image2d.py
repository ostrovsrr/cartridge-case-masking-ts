import json
from pathlib import Path

import cv2

from image2d import main


def test_brief_image_cli_writes_labels_and_preview(tmp_path):
    source = Path(__file__).resolve().parents[1] / "samples" / "brief" / "brief_p1_before.png"
    main([str(source), "-o", str(tmp_path)])
    result = tmp_path / source.stem
    summary = json.loads((result / "masks.json").read_text(encoding="utf-8"))
    assert summary["needs_review"]
    labels = cv2.imread(str(result / "labels.png"), cv2.IMREAD_UNCHANGED)
    original = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    assert labels.shape == original.shape[:2]
    assert int(labels.max()) > 0
    assert (tmp_path / f"{source.stem}__side_by_side_2d.png").stat().st_size > 0
