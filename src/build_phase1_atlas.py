"""Build a PDF atlas with all published Tables 1-7 and local Figure 1-20 analogues."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "outputs" / "source_code").is_dir()),
    SCRIPT_PATH.parents[1],
)
DEFAULT_FIGURES = PROJECT_ROOT / "outputs" / "phase1_reconstruction" / "figures"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "Phase1_Full_Paper_Reconstruction_Atlas.pdf"
PAPER_URL = "https://doi.org/10.1016/j.eswa.2023.121751"
NAVY = colors.HexColor("#14324A")
BLUE = colors.HexColor("#247BA0")
PALE = colors.HexColor("#EAF2F6")
LINE = colors.HexColor("#CBD5DC")
INK = colors.HexColor("#25323B")
MUTED = colors.HexColor("#5B6B75")
WHITE = colors.white

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="AtlasTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20,
                          leading=24, textColor=NAVY, alignment=TA_LEFT, spaceAfter=10))
styles.add(ParagraphStyle(name="AtlasSection", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                          leading=18, textColor=NAVY, spaceBefore=3, spaceAfter=7))
styles.add(ParagraphStyle(name="AtlasSub", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
                          leading=13, textColor=BLUE, spaceBefore=6, spaceAfter=4))
styles.add(ParagraphStyle(name="AtlasBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9,
                          leading=12.5, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="AtlasSmall", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.6,
                          leading=9.7, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(name="AtlasCell", parent=styles["BodyText"], fontName="Helvetica", fontSize=6.7,
                          leading=8.2, textColor=INK, spaceAfter=0))
styles.add(ParagraphStyle(name="AtlasCellHead", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=6.7,
                          leading=8.1, textColor=WHITE, spaceAfter=0))


def P(text: str, style: str = "AtlasBody"):
    return Paragraph(text, styles[style])


def make_table(rows, widths, font=None):
    converted = []
    for i, row in enumerate(rows):
        converted.append([cell if isinstance(cell, Paragraph) else P(escape(str(cell)), "AtlasCellHead" if i == 0 else "AtlasCell") for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), .35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ]
    for i in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, i), (-1, i), PALE))
    result.setStyle(TableStyle(commands))
    return result


def on_page(canvas, doc):
    canvas.saveState()
    w, h = landscape(A4)
    canvas.setStrokeColor(LINE)
    canvas.line(15 * mm, 12 * mm, w - 15 * mm, 12 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(15 * mm, 7.5 * mm, "Local figures are reconstructions; tables transcribed from Sharma et al. (2024).")
    canvas.drawRightString(w - 15 * mm, 7.5 * mm, str(doc.page))
    canvas.restoreState()


def paper_tables():
    nsl_dos = "apache2, land, pod, smurf, udpstorm, worm, back, mailbomb, teardrop, neptune, processtable"
    nsl_probe = "ipsweep, mscan, portsweep, satan, nmap, saint"
    nsl_r2l = "ftp_write, httptunnel, imap, named, phf, sendmail, Snmpgetattack, snmpguess, warezclient, warezmaster, xlock, guess_passwd, multihop, spy, xsnoop"
    nsl_u2r = "buffer_overflow, perl, ps, sqlattack, xterm, loadmodule, rootkit"
    table1 = [
        ["Attack class", "Attack subtype(s)", "Records"],
        ["Denial of Service (DoS)", nsl_dos, "45,927"], ["Probe", nsl_probe, "11,656"],
        ["Root to local (R2L)", nsl_r2l, "995"], ["User to Root (U2R)", nsl_u2r, "52"],
        ["Normal", "", "67,342"], ["Sum of records", "", "125,972"],
    ]
    nsl_features = [
        "protocol_type", "src_bytes", "rerror_rate", "is_guest_login", "srv_serror_rate", "diff_srv_rate", "service", "num_failed_logins", "dst_host_count", "num_compromised", "dst_host_same_srv_rate", "su_attempted", "Flag", "dst_host_rerror_rate",
        "num_shells", "num_access_files", "serror_rate", "dst_host_serror_rate", "duration", "count", "srv_count", "wrong_fragment", "dst_bytes", "land", "hot", "urgent", "srv_diff_host_rate", "logged_in",
        "srv_rerror_rate", "root_shell", "dst_host_diff_srv_rate", "num_root", "dst_host_same_src_port_rate", "is_host_login", "dst_host_srv_serror_rate", "num_file_creations", "dst_host_srv_rerror_rate", "num_outbound_cmds", "dst_host_srv_count", "same_srv_rate", "dst_host_srv_diff_host_rate", "class",
    ]
    table2 = [["No.", "Feature name", "No.", "Feature name", "No.", "Feature name"]]
    for i in range(14):
        row = []
        for col in range(3):
            idx = i + col * 14
            row.extend([str(idx + 1), nsl_features[idx]] if idx < len(nsl_features) else ["", ""])
        table2.append(row)

    unsw_features = [
        "Dur", "dpkts", "spkts", "dbytes", "sbytes", "rate", "Sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "ct_dst_src_ltm", "ct_ftp_cmd", "ct_src_ltm", "is_sm_ips_ports", "swin", "attack_cat", "Stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "is_ftp_login", "ct_dst_ltm", "ct_dst_sport_ltm", "ct_src_dport_ltm", "ct_state_ttl", "ct_flw_http_mthd", "ct_srv_dst", "proto", "service", "state", "label",
    ]
    table3 = [["No.", "Feature name", "No.", "Feature name", "No.", "Feature name"]]
    for i in range(15):
        row = []
        for col in range(3):
            idx = i + col * 15
            row.extend([str(idx + 1), unsw_features[idx]] if idx < len(unsw_features) else ["", ""])
        table3.append(row)

    table4 = [
        ["Attack category", "Records", "Attack category", "Records"],
        ["Generic", "58,871", "Exploits", "44,525"], ["DoS", "16,353", "Fuzzers", "24,246"],
        ["Analysis", "2,677", "Reconnaissance", "13,987"], ["Backdoor", "2,329", "Worms", "174"],
        ["Shellcode", "1,511", "Total attack", "164,673"], ["Normal", "93,000", "", ""], ["Total records", "257,673", "", ""],
    ]
    table5 = [["Metric", "Formula"], ["Accuracy", "A = (TP + TN) / (TP + TN + FP + FN)"],
              ["Precision", "P = TP / (TP + FP)"], ["Recall", "R = TP / (TP + FN)"], ["F1 score", "F = 2PR / (P + R)"]]
    table6 = [["Attack class", "DNN P", "DNN R", "DNN F1", "1D-CNN P", "1D-CNN R", "1D-CNN F1", "2D-CNN P", "2D-CNN R", "2D-CNN F1"],
              ["DoS", "1.00", "0.99", "1.00", "1.00", "0.99", "0.99", "1.00", "1.00", "1.00"],
              ["Normal", "0.99", "1.00", "0.99", "0.98", "0.99", "0.99", "0.99", "1.00", "0.99"],
              ["Probe", "0.99", "0.99", "0.99", "0.98", "0.99", "0.98", "0.98", "0.99", "0.99"],
              ["R2L", "0.80", "0.54", "0.06", "0.85", "0.70", "0.77", "0.93", "0.51", "0.66"],
              ["U2R", "0.83", "0.38", "0.53", "0.58", "0.54", "0.56", "0.60", "0.23", "0.33"],
              ["Accuracy", "", "0.99", "", "", "0.99", "", "", "0.99", ""]]
    table7 = [["Attack class", "DNN P", "DNN R", "DNN F1", "1D-CNN P", "1D-CNN R", "1D-CNN F1", "2D-CNN P", "2D-CNN R", "2D-CNN F1"],
              ["DoS", "0.49", "0.03", "0.06", "0.48", "0.06", "0.10", "0.57", "0.04", "0.07"],
              ["Exploits", "0.67", "0.90", "0.77", "0.64", "0.94", "0.76", "0.69", "0.91", "0.78"],
              ["Fuzzers", "0.59", "0.81", "0.68", "0.65", "0.67", "0.66", "0.60", "0.80", "0.69"],
              ["Generic", "1.00", "0.97", "0.98", "0.99", "0.97", "0.98", "1.00", "0.97", "0.99"],
              ["Normal", "0.93", "0.79", "0.85", "0.91", "0.81", "0.85", "0.91", "0.81", "0.86"],
              ["Accuracy", "", "0.80", "", "", "0.80", "", "", "0.81", ""]]
    return [table1, table2, table3, table4, table5, table6, table7]


def build(figures: Path, output: Path):
    manifest_path = figures / "figure_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run phase1_visuals.py first; missing {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        records = {int(row["figure"]): row for row in csv.DictReader(stream)}
    missing = sorted(set(range(1, 21)) - set(records))
    if missing:
        raise ValueError(f"Figure manifest is incomplete; missing {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    page = landscape(A4)
    doc = SimpleDocTemplate(str(output), pagesize=page, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=17 * mm,
                            title="Phase 1 Full Paper Reconstruction Atlas - Sharma et al. (2024)",
                            author="Replication project")
    story = [Spacer(1, 25 * mm), P("Phase 1: Full Paper Reconstruction Atlas", "AtlasTitle"),
             P("Sharma et al. (2024) - Explainable Artificial Intelligence for Intrusion Detection in IoT Networks", "AtlasSub"),
             Spacer(1, 5 * mm),
             P("This atlas includes all seven tables transcribed from the published paper and a local counterpart for each of its 20 figures. Every generated plot uses this project's saved datasets, splits, predictions, model histories, or DNN explanations. The original paper images and its original model outputs are not represented as local results.", "AtlasBody"),
             P("The local figures are visual counterparts, not pixel-for-pixel copies. The paper does not provide code, split indices, a random seed, complete explanation settings, or the underlying output arrays. Published tables below remain labeled as paper-reported values. Local figures are generated from the saved Phase 1 models and data; LIME and SHAP choices are documented on their pages.", "AtlasBody"),
             P(f"Source article: Sharma, B., Sharma, L., Lal, C., & Roy, S. (2024), Expert Systems with Applications 238, 121751. {escape(PAPER_URL)}", "AtlasSmall"),
             P("Figure and table inventory", "AtlasSub"),
             P("Figures 1-20: workflow; two correlation matrices; two model architectures; confusion-matrix key; six model confusion heatmaps; per-class metrics; training curves; accuracy comparison; four local LIME cases; two local SHAP plots; two global SHAP summaries; and two per-class mean-absolute-SHAP plots.", "AtlasBody"),
             P("Tables 1-7: published NSL-KDD class counts, NSL-KDD and UNSW feature lists, UNSW class counts, metric definitions, and both published class-level classification reports.", "AtlasBody"), PageBreak()]

    table_data = paper_tables()
    table_titles = [
        "Table 1. NSL-KDD CUP dataset attack statistics (paper, p. 5)",
        "Table 2. NSL-KDD CUP feature list (paper, p. 5)",
        "Table 3. UNSW-NB15 feature list (paper, p. 6)",
        "Table 4. UNSW-NB15 attack statistics (paper, p. 6)",
        "Table 5. Model performance evaluation metrics (paper, p. 8)",
        "Table 6. Published classification report for NSL-KDDnew (paper, p. 10)",
        "Table 7. Published classification report for UNSW-NBnew (paper, p. 10)",
    ]
    for i, (title, rows) in enumerate(zip(table_titles, table_data), start=1):
        story += [P(title, "AtlasSection"),
                  P("Published source table, transcribed for comparison. The values below are not measurements from this reproduction.", "AtlasSmall")]
        if i == 1:
            widths = [35 * mm, 204 * mm, 24 * mm]
        elif i in (2, 3):
            widths = [12 * mm, 66 * mm, 12 * mm, 66 * mm, 12 * mm, 66 * mm]
        elif i == 4:
            widths = [70 * mm, 48 * mm, 70 * mm, 48 * mm]
        elif i == 5:
            widths = [48 * mm, 219 * mm]
        else:
            widths = [33 * mm] + [26 * mm] * 9
        story += [make_table(rows, widths)]
        if i == 6:
            story += [Spacer(1, 3 * mm), P("Published Table 6 lists DNN R2L precision 0.80 and recall 0.54 with F1 0.06. The standard harmonic-mean calculation gives approximately 0.64, so the source row is internally inconsistent; it is preserved as printed.", "AtlasSmall")]
        if i == 7:
            story += [Spacer(1, 3 * mm), P("The article uses 'UNSW-NBnew' for the processed benchmark in its captions. This reproduction uses the standard UNSW-NB15 split files and reports its preprocessing details separately.", "AtlasSmall")]
        story.append(PageBreak())

    for number in range(1, 21):
        record = records[number]
        story += [P(f"Figure {number}. {escape(record['title'])} - local counterpart", "AtlasSection"),
                  P(f"Paper location: {escape(record['paper_page'])}. Local generation: {escape(record['local_method'])}", "AtlasSmall")]
        img_path = figures / record["file"]
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        img = Image(str(img_path), width=page[0] - 32 * mm, height=page[1] - 63 * mm, kind="proportional")
        story += [Spacer(1, 2 * mm), img]
        story.append(PageBreak())

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.figures_dir, args.output)


if __name__ == "__main__":
    main()
