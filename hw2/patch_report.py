"""Patch report_111550132.docx in-place:
1. Figure numbers: all +1 (0->1, 1a->2a, 2->3, 3a->4a, 4->5, 4a->6a, 5a->7a, 6a->8a)
2. 90% recovery calculation: make explicit
3. Replace all images with larger-font versions
"""

import os
import re
from docx import Document
from docx.shared import Inches
from docx.oxml.ns import qn

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "report_111550132.docx")
DST = os.path.join(BASE, "report_111550132.docx")  # overwrite

doc = Document(SRC)

# =========================================================================
# 1 & 2: Fix text in paragraphs (figure numbers + 90% calculation)
# =========================================================================

# Figure renaming map — order matters (longer patterns first to avoid partial matches)
FIG_MAP = [
    ("Figure 6b", "Figure 8b"),
    ("Figure 6a", "Figure 8a"),
    ("Figure 5b", "Figure 7b"),
    ("Figure 5a", "Figure 7a"),
    ("Figure 4b", "Figure 6b"),
    ("Figure 4a", "Figure 6a"),
    ("Figure 4", "Figure 5"),
    ("Fig. 1b", "Fig. 2b"),
    ("Fig. 1a", "Fig. 2a"),
    ("Figure 3b", "Figure 4b"),
    ("Figure 3a", "Figure 4a"),
    ("Figure 2", "Figure 3"),
    ("Figure 1b", "Figure 2b"),
    ("Figure 1a", "Figure 2a"),
    ("Figure 0", "Figure 1"),
]

OLD_90 = (
    "SSL recovers approximately 90% of the improvement that supervised learning "
    "achieves over random features (46.31 / 51.22"
)
NEW_90 = (
    "To quantify: the improvement of SSL over random is "
    "86.60 \u2212 40.29 = 46.31%, while the improvement of supervised over random is "
    "91.51 \u2212 40.29 = 51.22%. Thus, SSL recovers 46.31 / 51.22"
)


def patch_runs(paragraph):
    """Apply text replacements across all runs in a paragraph, preserving formatting."""
    full = paragraph.text
    changed = False

    # Check figure renaming
    for old, new in FIG_MAP:
        if old in full:
            full = full.replace(old, new)
            changed = True

    # Check 90% calc
    if OLD_90 in full:
        full = full.replace(OLD_90, NEW_90)
        changed = True

    if not changed:
        return

    # Rewrite: keep first run's formatting, put all text into it, remove the rest
    if not paragraph.runs:
        return
    first_run = paragraph.runs[0]
    first_run.text = full
    for run in paragraph.runs[1:]:
        run.text = ""


# Also handle table cells
def patch_all_paragraphs(paragraphs):
    for p in paragraphs:
        patch_runs(p)


# Patch main body
patch_all_paragraphs(doc.paragraphs)

# Patch tables (figure captions inside tables)
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            patch_all_paragraphs(cell.paragraphs)

print("Text patching done.")

# =========================================================================
# 3: Replace all images with larger-font versions
# =========================================================================

# Build a map: original image path -> new image path
# We need to find all images (blip elements) in the docx and replace them

LOGS = os.path.join(BASE, "logs")

# Map of image files we regenerated (relative paths from LOGS)
IMAGE_FILES = [
    "baseline/loss_curve.png",
    "baseline/knn_curve.png",
    "ssl_probe/linear_probe_acc.png",
    "supervised/loss_curve.png",
    "supervised/accuracy_curve.png",
    "temp01/loss_curve.png",
    "temp01/knn_curve.png",
    "temp50/loss_curve.png",
    "temp50/knn_curve.png",
    "no_proj/loss_curve.png",
    "no_proj/knn_curve.png",
    "no_proj_probe/linear_probe_acc.png",
    "temp01_probe/linear_probe_acc.png",
    "temp50_probe/linear_probe_acc.png",
    "random_probe/linear_probe_acc.png",
    "ssl_cifar100/linear_probe_acc.png",
    "sl_cifar100/linear_probe_acc.png",
    "random_cifar100/linear_probe_acc.png",
    "visuals/augmentation_examples.png",
    "visuals/knn_retrieval_examples.png",
]

# Collect all image rIds and their relationships
rels = doc.part.rels
image_rels = {}
for rel_id, rel in rels.items():
    if "image" in rel.reltype:
        # rel.target_ref is like 'media/image1.png'
        image_rels[rel_id] = rel

# Find all blip elements (embedded images) in the document
blips = doc.element.findall('.//' + qn('a:blip'))
print(f"Found {len(blips)} embedded images in document.")

# We'll replace images by their order of appearance (matching the order in generate_report.py)
# The order in the docx should match the order we added them:
# 1. augmentation_examples.png  (Figure 1)
# 2. baseline/loss_curve.png    (Figure 2a - in table)
# 3. baseline/knn_curve.png     (Figure 2b - in table)
# 4. ssl_probe/linear_probe_acc.png (Figure 3)
# 5. supervised/loss_curve.png  (Figure 4a)
# 6. supervised/accuracy_curve.png (Figure 4b)
# 7. knn_retrieval_examples.png (Figure 5)
# 8. temp01/loss_curve.png      (Figure 6a)
# 9. temp50/loss_curve.png      (Figure 6b)
# 10. temp01/knn_curve.png      (Figure 7a)
# 11. temp50/knn_curve.png      (Figure 7b)
# 12. no_proj/loss_curve.png    (Figure 8a)
# 13. no_proj/knn_curve.png     (Figure 8b)

IMAGE_ORDER = [
    "visuals/augmentation_examples.png",
    "baseline/loss_curve.png",
    "baseline/knn_curve.png",
    "ssl_probe/linear_probe_acc.png",
    "supervised/loss_curve.png",
    "supervised/accuracy_curve.png",
    "visuals/knn_retrieval_examples.png",
    "temp01/loss_curve.png",
    "temp50/loss_curve.png",
    "temp01/knn_curve.png",
    "temp50/knn_curve.png",
    "no_proj/loss_curve.png",
    "no_proj/knn_curve.png",
]

replaced = 0
for idx, blip in enumerate(blips):
    r_embed = blip.get(qn('r:embed'))
    if r_embed and r_embed in image_rels and idx < len(IMAGE_ORDER):
        new_img_path = os.path.join(LOGS, IMAGE_ORDER[idx])
        if os.path.exists(new_img_path):
            rel = image_rels[r_embed]
            # Replace the image data in the relationship
            with open(new_img_path, 'rb') as f:
                rel.target_part._blob = f.read()
            replaced += 1
            print(f"  Replaced image {idx}: {IMAGE_ORDER[idx]}")

print(f"Replaced {replaced} images.")

# Save
doc.save(DST)
print(f"Saved to: {DST}")
