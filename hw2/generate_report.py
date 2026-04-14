"""Generate report.docx from experimental results."""

import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn

BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs")


def img(relpath):
    return os.path.join(LOGS, relpath)


def set_cell_shading(cell, color):
    """Set cell background color."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shading = tcPr.makeelement(qn("w:shd"), {
        qn("w:fill"): color,
        qn("w:val"): "clear",
    })
    tcPr.append(shading)


def add_table(doc, headers, rows, col_widths=None):
    """Add a formatted table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(10)
        run.font.name = "Times New Roman"
        set_cell_shading(cell, "D9E2F3")

    # Data rows
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            run.font.size = Pt(10)
            run.font.name = "Times New Roman"
            # Bold the first column
            if c_idx == 0:
                run.bold = True

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    doc.add_paragraph()  # spacing
    return table


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h


def add_para(doc, text, bold=False, size=12, align=None, space_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    run.bold = bold
    if align:
        p.alignment = align
    fmt = p.paragraph_format
    fmt.space_after = Pt(space_after)
    fmt.space_before = Pt(0)
    return p


def add_figure(doc, path, caption, width=Inches(4.5)):
    """Add a centered image with caption."""
    if not os.path.exists(path):
        add_para(doc, f"[Image not found: {path}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(path, width=width)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.font.size = Pt(10)
    run.font.name = "Times New Roman"
    run.italic = True
    cap.paragraph_format.space_after = Pt(6)


def add_side_by_side_figures(doc, path1, cap1, path2, cap2, width=Inches(3.0)):
    """Add two images side by side using a 2-column table."""
    table = doc.add_table(rows=2, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for path, col in [(path1, 0), (path2, 1)]:
        if os.path.exists(path):
            cell = table.rows[0].cells[col]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run()
            run.add_picture(path, width=width)

    for cap, col in [(cap1, 0), (cap2, 1)]:
        cell = table.rows[1].cells[col]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(cap)
        run.font.size = Pt(9)
        run.italic = True

    # Remove borders
    for row in table.rows:
        for cell in row.cells:
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            borders = tcPr.makeelement(qn("w:tcBorders"), {})
            for edge in ["top", "left", "bottom", "right"]:
                el = borders.makeelement(qn(f"w:{edge}"), {
                    qn("w:val"): "none", qn("w:sz"): "0",
                    qn("w:space"): "0", qn("w:color"): "auto"
                })
                borders.append(el)
            tcPr.append(borders)

    doc.add_paragraph()


def add_page_break(doc):
    doc.add_page_break()


def build_report():
    doc = Document()

    # --- Page setup: A4 ---
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

    # --- Default font ---
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0)

    # =====================================================================
    # TITLE
    # =====================================================================
    add_para(doc, "Artificial Intelligence — Project #2", bold=True, size=16,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_para(doc, "SimCLR Self-Supervised Learning on CIFAR-10",
             bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_para(doc, "NYCU Spring 2026", size=12,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

    # =====================================================================
    # 1. RESEARCH QUESTION & MOTIVATION
    # =====================================================================
    add_heading(doc, "1. Research Question and Motivation", level=1)

    add_para(doc,
        "Self-supervised learning (SSL) has emerged as a powerful paradigm for learning visual "
        "representations without human-annotated labels. SimCLR, a contrastive learning framework, "
        "trains a neural network to distinguish between different images by maximizing agreement "
        "between augmented views of the same image. This project investigates the following questions:"
    )
    add_para(doc,
        "(1) How effectively can SimCLR learn representations on CIFAR-10 compared to supervised "
        "learning trained from scratch with the same architecture?"
    )
    add_para(doc,
        "(2) How do key hyperparameters — temperature and the projector head — affect the quality "
        "of learned representations?"
    )
    add_para(doc,
        "(3) Do SSL representations generalize better than supervised ones when transferred to a "
        "different dataset (CIFAR-100)?"
    )

    # =====================================================================
    # 2. METHODS
    # =====================================================================
    add_heading(doc, "2. Methods", level=1)

    add_heading(doc, "2.1 Model Architecture", level=2)
    add_para(doc,
        "We use a modified ResNet-18 as the backbone encoder. To accommodate 32×32 CIFAR-10 images, "
        "the first convolutional layer is changed from 7×7 (stride=2) to 3×3 (stride=1, padding=1), "
        "and the subsequent max-pooling layer is replaced with an identity operation. The backbone "
        "outputs a 512-dimensional feature vector after global average pooling. "
        "A two-layer MLP projector head (512→512→128 with ReLU) maps features to a 128-dim space "
        "for contrastive loss computation. All models are trained from scratch (no pretrained weights)."
    )

    add_heading(doc, "2.2 SimCLR Training", level=2)
    add_para(doc,
        "For each batch of N images, two independent augmented views are generated, producing 2N "
        "samples. Augmentations include: RandomResizedCrop (scale 0.2–1.0), RandomHorizontalFlip (p=0.5), "
        "ColorJitter (brightness/contrast/saturation=0.4, hue=0.1, applied with p=0.8), "
        "RandomGrayscale (p=0.2), and normalization with CIFAR-10 statistics. "
        "The NT-Xent loss is computed on L2-normalized projector outputs: for each view, "
        "the loss is the negative log-softmax probability of its mate among 2N−1 candidates, "
        "scaled by temperature τ. Training uses Adam optimizer (lr=3e-4, weight decay=1e-6) "
        "for 200 epochs with batch size 256."
    )

    add_figure(doc, img("visuals/augmentation_examples.png"),
               "Figure 0: SimCLR augmentation examples — original image (left) and two "
               "independently augmented views (middle, right) for each CIFAR-10 class.",
               width=Inches(3.8))
    add_para(doc,
        "Figure 0 shows augmentation examples for all 10 CIFAR-10 classes. Each row displays "
        "the original image and two independently sampled views. The diversity of crops, color "
        "shifts, and grayscale conversions forces the model to learn content-invariant features "
        "rather than relying on color or spatial cues."
    )

    add_heading(doc, "2.3 Evaluation Methods", level=2)
    add_para(doc,
        "kNN Monitor: Every 5 epochs, we compute k-nearest-neighbor classification (k=20) on the "
        "test set using L2-normalized backbone features of the training set as the memory bank. "
        "This provides a training-free measure of representation quality."
    )
    add_para(doc,
        "Linear Probing: After SSL training, we freeze the backbone and train a single linear "
        "layer (512→num_classes) with Adam (lr=1e-3, weight decay=1e-6) for 100 epochs. "
        "This evaluates the linear separability of learned representations."
    )
    add_para(doc,
        "Supervised Baseline: The same backbone architecture with a classification head (512→10) "
        "is trained end-to-end with cross-entropy loss, Adam (lr=3e-4), and standard augmentation "
        "(RandomCrop with padding=4, RandomHorizontalFlip) for 200 epochs."
    )

    add_heading(doc, "2.4 Tools and Libraries", level=2)
    add_para(doc,
        "Implementation uses PyTorch and torchvision. The modified ResNet-18 is based on "
        "torchvision.models.resnet18 with surgical modifications. Code structure is modularized "
        "into config.py, dataset.py, model.py, utils.py, train_simclr.py, train_supervised.py, "
        "and linear_eval.py. Claude Code (AI assistant) was used to assist with code implementation."
    )

    # =====================================================================
    # 3. EXPERIMENTS AND RESULTS
    # =====================================================================
    add_heading(doc, "3. Experiments and Results", level=1)

    # --- 3.1 SimCLR Baseline ---
    add_heading(doc, "3.1 SimCLR Baseline", level=2)

    add_side_by_side_figures(doc,
        img("baseline/loss_curve.png"), "Figure 1a: NT-Xent Loss",
        img("baseline/knn_curve.png"), "Figure 1b: kNN Accuracy (k=20)",
    )

    add_para(doc,
        "The NT-Xent loss decreases steadily from 5.265 to 4.502 over 200 epochs (Fig. 1a). "
        "The kNN accuracy rises rapidly in early training — reaching 63.3% by epoch 10 — "
        "and continues improving to 84.55% at epoch 200, though gains slow after epoch 100 (Fig. 1b)."
    )

    add_table(doc,
        ["Epoch", "kNN Accuracy"],
        [["1", "44.31%"], ["10", "63.33%"], ["50", "76.33%"],
         ["100", "81.16%"], ["150", "83.15%"], ["200", "84.55%"]],
    )

    # --- 3.2 Linear Probing ---
    add_heading(doc, "3.2 Linear Probing on SSL Model", level=2)
    add_figure(doc, img("ssl_probe/linear_probe_acc.png"),
               "Figure 2: SSL Linear Probing Accuracy on CIFAR-10", width=Inches(4.0))
    add_para(doc,
        "Freezing the SimCLR backbone and training a linear classifier for 100 epochs yields "
        "a final test accuracy of 86.60%, confirming that the learned representations are "
        "linearly separable across CIFAR-10 classes."
    )

    # --- 3.3 Supervised ---
    add_heading(doc, "3.3 Supervised Baseline", level=2)
    add_side_by_side_figures(doc,
        img("supervised/loss_curve.png"), "Figure 3a: Supervised Loss",
        img("supervised/accuracy_curve.png"), "Figure 3b: Supervised Accuracy",
    )
    add_para(doc,
        "The supervised model reaches 99.65% train accuracy but 91.51% test accuracy "
        "(best: 92.16% at epoch 189), showing a ~8% generalization gap indicative of overfitting."
    )

    # --- 3.4 Comparison ---
    add_heading(doc, "3.4 SSL vs. Supervised vs. Random Baseline", level=2)

    add_table(doc,
        ["Method", "CIFAR-10 Test Acc", "Note"],
        [["Supervised Learning", "91.51%", "End-to-end with labels"],
         ["SimCLR + Linear Probe", "86.60%", "No labels for backbone"],
         ["SimCLR kNN (Epoch 200)", "84.55%", "Training-free evaluation"],
         ["Random Backbone + LP", "40.29%", "Lower bound"]],
    )

    add_para(doc,
        "SimCLR linear probing (86.60%) trails supervised learning (91.51%) by only 4.91 "
        "percentage points — despite using zero labels for backbone training. The random baseline "
        "(40.29%) confirms that the SSL training provides substantial representation quality: "
        "SSL recovers approximately 90% of the improvement that supervised learning achieves "
        "over random features (46.31 / 51.22 ≈ 90.4%)."
    )

    add_figure(doc, img("visuals/knn_retrieval_examples.png"),
               "Figure 4: kNN retrieval examples — for each query image (blue border, left), "
               "the 5 nearest neighbors from the training set are shown. Green borders indicate "
               "correct class matches; red borders indicate mismatches. Cosine similarity scores "
               "are displayed above each neighbor.",
               width=Inches(5.2))
    add_para(doc,
        "Figure 4 provides qualitative evidence of representation quality. For most classes, "
        "all 5 nearest neighbors share the same class as the query, and similarity scores are "
        "high (>0.85). Mismatches tend to occur between visually similar classes (e.g., automobile "
        "and truck), revealing interpretable failure modes that reflect genuine visual similarity "
        "rather than random errors."
    )

    # =====================================================================
    # 3.5 Temperature Ablation
    # =====================================================================
    add_heading(doc, "3.5 Temperature Ablation", level=2)

    add_para(doc,
        "We train SimCLR with three temperature values: τ=0.1 (low), τ=0.5 (baseline), "
        "and τ=5.0 (high). All other settings are identical."
    )

    # Loss curves side by side
    add_side_by_side_figures(doc,
        img("temp01/loss_curve.png"), "Figure 4a: Loss (τ=0.1)",
        img("temp50/loss_curve.png"), "Figure 4b: Loss (τ=5.0)",
    )

    add_table(doc,
        ["Temperature", "Initial Loss", "Final Loss", "Loss Drop"],
        [["0.1", "4.036", "0.433", "3.603"],
         ["0.5 (baseline)", "5.265", "4.502", "0.763"],
         ["5.0", "6.111", "6.054", "0.057"]],
    )

    add_para(doc,
        "The loss magnitude is not comparable across temperatures — τ directly scales the logits, "
        "so τ=0.1 produces the largest drop while τ=5.0 barely moves. This demonstrates that "
        "the loss value alone is a poor indicator of representation quality in SSL."
    )

    # kNN curves
    add_side_by_side_figures(doc,
        img("temp01/knn_curve.png"), "Figure 5a: kNN (τ=0.1)",
        img("temp50/knn_curve.png"), "Figure 5b: kNN (τ=5.0)",
    )

    add_table(doc,
        ["Epoch", "τ=0.1", "τ=0.5 (baseline)", "τ=5.0"],
        [["1", "50.59%", "44.31%", "38.36%"],
         ["50", "75.65%", "76.33%", "64.57%"],
         ["100", "79.09%", "81.16%", "70.46%"],
         ["200", "82.26%", "84.55%", "77.80%"]],
    )

    add_table(doc,
        ["Temperature", "kNN (Epoch 200)", "Linear Probing"],
        [["0.1", "82.26%", "84.43%"],
         ["0.5 (baseline)", "84.55%", "86.60%"],
         ["5.0", "77.80%", "82.03%"]],
    )

    add_para(doc,
        "τ=0.1 learns faster initially (50.59% vs 44.31% at epoch 1) but plateaus lower, "
        "because the overly peaked softmax causes the model to focus on the hardest negatives, "
        "producing less globally structured representations. "
        "τ=5.0 learns very slowly, as the flattened softmax provides weak gradient signals. "
        "τ=0.5 offers the best balance, achieving the highest kNN (84.55%) and linear probing (86.60%)."
    )

    # =====================================================================
    # 3.6 Projector Head Ablation
    # =====================================================================
    add_heading(doc, "3.6 Projector Head Ablation", level=2)

    add_side_by_side_figures(doc,
        img("no_proj/loss_curve.png"), "Figure 6a: No Projector — Loss",
        img("no_proj/knn_curve.png"), "Figure 6b: No Projector — kNN",
    )

    add_table(doc,
        ["Model", "kNN (Epoch 200)", "Linear Probing"],
        [["With Projector (baseline)", "84.55%", "86.60%"],
         ["Without Projector", "83.15%", "83.96%"],
         ["Difference", "−1.40%", "−2.64%"]],
    )

    add_para(doc,
        "Removing the projector head decreases linear probing accuracy by 2.64%. "
        "The projector acts as an information bottleneck that absorbs task-specific distortions "
        "from the contrastive loss, allowing the backbone to retain more general-purpose features. "
        "Without it, the NT-Xent loss directly shapes the backbone's 512-dim output, "
        "over-specializing it for the contrastive task at the expense of downstream linear separability."
    )

    # =====================================================================
    # 3.7 Transfer Learning
    # =====================================================================
    add_heading(doc, "3.7 Transfer Learning to CIFAR-100", level=2)

    add_para(doc,
        "We freeze backbones trained on CIFAR-10 (SSL, Supervised, Random) and perform linear "
        "probing on CIFAR-100 (100 classes, 500 images/class). Images are resized to 32×32 "
        "and normalized with CIFAR-100 statistics. Linear probing settings are identical to CIFAR-10."
    )

    add_table(doc,
        ["Backbone", "CIFAR-10 LP", "CIFAR-100 LP"],
        [["SimCLR (SSL)", "86.60%", "50.50%"],
         ["Supervised (SL)", "91.51%", "49.48%"],
         ["Random", "40.29%", "19.16%"]],
    )

    add_para(doc,
        "A striking result: SSL (50.50%) surpasses Supervised (49.48%) on CIFAR-100 transfer, "
        "despite trailing by 4.91% on CIFAR-10. This reversal demonstrates that SSL's instance "
        "discrimination objective learns more transferable features than supervised learning, "
        "which overfits to the specific 10-class label space of CIFAR-10. "
        "The relative accuracy drop from CIFAR-10 to CIFAR-100 is also smaller for SSL "
        "(41.7%) than for Supervised (45.9%), further confirming the generalization advantage."
    )

    # =====================================================================
    # 4. DISCUSSION
    # =====================================================================
    add_page_break(doc)
    add_heading(doc, "4. Discussion", level=1)

    add_heading(doc, "4.1 Are the Results Expected?", level=2)
    add_para(doc,
        "The ~5% gap between SSL linear probing and supervised learning on CIFAR-10 is consistent "
        "with published SimCLR results on small-scale benchmarks. The original SimCLR paper reports "
        "similar gaps when using ResNet-18 with moderate batch sizes. The kNN accuracy of 84.55% "
        "also falls within the expected range for 200 epochs of training with batch size 256."
    )
    add_para(doc,
        "The temperature ablation results match theoretical expectations: τ=0.5 is near the commonly "
        "recommended range (0.1–0.5), and extreme values degrade performance for the reasons discussed. "
        "The projector head ablation confirms findings from the original SimCLR paper."
    )
    add_para(doc,
        "The most interesting finding is the SSL transfer advantage on CIFAR-100. While this has "
        "been reported in the literature for larger models and datasets, observing it with a simple "
        "ResNet-18 on CIFAR-10/100 is encouraging and validates the core thesis of SSL."
    )

    add_heading(doc, "4.2 Key Factors Affecting Results", level=2)
    add_para(doc,
        "Batch size: SimCLR relies on in-batch negatives. With batch size 256, the model sees "
        "511 negative pairs per sample. Larger batches would likely improve performance. "
        "Augmentation: The combination of crop, color jitter, and grayscale is critical. "
        "Crop forces spatial invariance; color jitter forces color invariance. "
        "Training duration: The kNN curve at epoch 200 has not fully saturated, suggesting "
        "longer training (e.g., 500–1000 epochs) could further improve results. "
        "Dataset scale: CIFAR-10 has only 50,000 32×32 images. SimCLR benefits more from "
        "larger datasets (e.g., ImageNet) where supervised labeling is expensive."
    )

    add_heading(doc, "4.3 Future Experiments", level=2)
    add_para(doc,
        "Given more time, the following experiments would be valuable: "
        "(1) Batch size ablation (32–1024) to quantify the effect of negative sample diversity. "
        "(2) Augmentation ablation to identify the most critical transforms. "
        "(3) Longer training (500+ epochs) to determine if the SSL-SL gap continues to narrow. "
        "(4) Using projector outputs (128-dim) as representations during evaluation. "
        "(5) Training on a larger dataset (e.g., STL-10 unlabeled split) and evaluating on CIFAR-10. "
        "(6) Comparing with other SSL methods such as BYOL or SimSiam."
    )

    add_heading(doc, "4.4 Lessons Learned", level=2)
    add_para(doc,
        "The most important takeaway is that the contrastive loss value is not a reliable "
        "training signal — kNN monitoring is essential for tracking actual representation quality. "
        "The temperature ablation vividly illustrates this: τ=0.1 produces the largest loss drop "
        "but not the best representations. "
        "Second, the projector head is not just an architectural detail but a principled design "
        "choice that protects the backbone's representation space. "
        "Third, the SSL transfer advantage on CIFAR-100 demonstrates why self-supervised learning "
        "is valuable as a foundation model approach: the representations are not tied to a specific "
        "label set and generalize better to new tasks."
    )

    add_heading(doc, "4.5 Remaining Questions", level=2)
    add_para(doc,
        "(1) Would the SSL transfer advantage over supervised learning persist on more distant "
        "domains (e.g., medical images, satellite imagery) rather than the closely related "
        "CIFAR-10 → CIFAR-100 setting? The two datasets share similar low-resolution natural "
        "image statistics, so the gap might widen or narrow on more diverse transfers."
    )
    add_para(doc,
        "(2) The kNN accuracy at epoch 200 (84.55%) has not fully saturated. How much further "
        "could it improve with 500 or 1000 epochs? Is there a point of diminishing returns, "
        "or could the SSL-SL gap be closed entirely with sufficient training?"
    )
    add_para(doc,
        "(3) Our batch size of 256 provides 511 negatives per sample. The original SimCLR paper "
        "used batch sizes up to 8192. How much of the remaining 5% gap between SSL and supervised "
        "learning is attributable to insufficient batch size versus a fundamental limitation of "
        "the linear probing protocol?"
    )
    add_para(doc,
        "(4) The temperature ablation showed τ=0.5 is optimal among {0.1, 0.5, 5.0}, but the "
        "search is coarse. Is the optimal temperature closer to 0.3 or 0.7? How does it interact "
        "with batch size — should larger batches use lower temperatures?"
    )
    add_para(doc,
        "(5) The projector head ablation showed a 2.64% drop without it. But does a deeper or "
        "wider projector (e.g., 3 layers, or 2048 hidden units) yield further gains, or is the "
        "current 2-layer MLP already near optimal for this scale?"
    )

    # =====================================================================
    # 5. REFERENCES
    # =====================================================================
    add_heading(doc, "5. References", level=1)

    refs = [
        "[1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, \"A Simple Framework for "
        "Contrastive Learning of Visual Representations,\" ICML 2020. arXiv:2002.05709.",
        "[2] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image Recognition,\" "
        "CVPR 2016.",
        "[3] A. Krizhevsky, \"Learning Multiple Layers of Features from Tiny Images,\" "
        "Technical Report, University of Toronto, 2009.",
        "[4] PyTorch: https://pytorch.org/",
        "[5] torchvision: https://pytorch.org/vision/",
        "[6] Claude Code (Anthropic) was used as an AI coding assistant for implementation.",
    ]
    for ref in refs:
        add_para(doc, ref, size=11, space_after=3)

    # =====================================================================
    # APPENDIX: CODE LISTING
    # =====================================================================
    add_page_break(doc)
    add_heading(doc, "Appendix: Program Code", level=1)
    add_para(doc, "(Not counting toward the 10-page limit)", size=10)

    code_files = [
        "config.py", "dataset.py", "model.py", "utils.py",
        "train_simclr.py", "train_supervised.py", "linear_eval.py",
    ]
    for fname in code_files:
        fpath = os.path.join(BASE, fname)
        if not os.path.exists(fpath):
            continue
        add_heading(doc, fname, level=2)
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()
        p = doc.add_paragraph()
        run = p.add_run(code)
        run.font.size = Pt(8)
        run.font.name = "Consolas"
        fmt = p.paragraph_format
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(6)

    # Save
    out = os.path.join(BASE, "report.docx")
    doc.save(out)
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    build_report()
