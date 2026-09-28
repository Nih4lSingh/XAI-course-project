from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

output = "report/replication_summary.pdf"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=colors.black, spaceAfter=18))
styles.add(ParagraphStyle(name="HeadingCustom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=8))
doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
story = []
story.append(Paragraph("Replication Summary Template", styles["TitleCustom"]))
story.append(Paragraph("Sharma et al. 2024 - Explainable artificial intelligence for intrusion detection in IoT networks", styles["BodyCustom"]))
story.append(Spacer(1, 8))
story.append(Paragraph("Status: template only. No experimental values are reported in this document.", styles["BodyCustom"]))
for heading, body in [
    ("Scope", "This report records a planned or completed replication. It does not claim results until the configured code has been run on documented NSL-KDD and UNSW-NB15 files."),
    ("Source and Protocol", "Paper DOI: 10.1016/j.eswa.2023.121861<br/>Full-paper table/page consulted: ____________________<br/>Dataset versions and acquisition dates: ____________________<br/>Task definition and labels: ____________________<br/>Encoding, scaling, feature selection, correlation threshold, and selected feature count: ____________________"),
    ("Experiment Record", "Transcribe paper-specific values before comparing an experiment to the source paper."),
]:
    story.append(Paragraph(heading, styles["HeadingCustom"])); story.append(Paragraph(body, styles["BodyCustom"]))
table = Table([["Model", "Datasets", "Paper settings", "Replication settings", "Status"], ["DNN", "Both", "Not transcribed", "Not run", "Pending"], ["1D-CNN", "Both", "Not transcribed", "Not run", "Pending"], ["2D-CNN", "Both", "Not transcribed", "Not run", "Pending"]], colWidths=[0.75*inch, 0.75*inch, 1.75*inch, 1.75*inch, 0.7*inch])
table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F4E79")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME", (0,1), (-1,-1), "Helvetica"), ("FONTSIZE", (0,0), (-1,-1), 8.5), ("LEADING", (0,0), (-1,-1), 11), ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#C9D3DD")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7), ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#F6F9FC"))]))
story.append(table)
story.append(Paragraph("Both refers to NSL-KDD and UNSW-NB15; create one row per completed model-dataset run in the result ledger.", styles["BodyCustom"]))
story.append(Paragraph("Results", styles["HeadingCustom"]))
story.append(Paragraph("No replication measurements have been generated. After execution, carry metrics from results folders into the comparison ledger and retain run metadata, figures, and environment details.", styles["BodyCustom"]))
story.append(Paragraph("Explainability and Deviations", styles["HeadingCustom"]))
story.append(Paragraph("Explainability protocol, target model, background sample, and interpretation: ____________________<br/><br/>Record every departure from the paper, software versions, hardware, random seed, split, runtime, and unresolved ambiguity here.", styles["BodyCustom"]))
doc.build(story)
