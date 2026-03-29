from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table
from reportlab.lib.units import inch
import os
from datetime import datetime


def generate_report(data, image_path, heatmap_path):
    pdf_path = os.path.join("static", "outputs", "AI_Skin_Report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    # -----------------------------
    # Title Style
    # -----------------------------
    title_style = ParagraphStyle(
        name='Title',
        fontSize=18,
        textColor=colors.HexColor("#6a00ff"),
        spaceAfter=15
    )

    elements.append(Paragraph("AI Skin Cancer Analysis Report", title_style))
    elements.append(Spacer(1, 12))

    # -----------------------------
    # Patient & Prediction Info Table
    # -----------------------------
    now = datetime.now().strftime("%d-%m-%Y %H:%M")
    info = [
        ["Patient Name", data.get("name", "N/A")],
        ["Age",          data.get("age", "N/A")],
        ["Gender",       data.get("gender", "N/A")],
        ["Date",         now],
        ["Predicted Class", data.get("class")],
        ["AI Confidence",   str(data.get("confidence")) + "%"],
        ["Risk Level",      data.get("risk")]
    ]

    table = Table(info, colWidths=[2.5 * inch, 3 * inch])
    elements.append(table)
    elements.append(Spacer(1, 20))

    # -----------------------------
    # Uploaded Lesion Image
    # -----------------------------
    if os.path.exists(image_path):
        elements.append(Paragraph("Uploaded Lesion Image", title_style))
        elements.append(Spacer(1, 10))
        elements.append(Image(image_path, width=4 * inch, height=4 * inch))

    elements.append(PageBreak())

    # -----------------------------
    # Grad-CAM Heatmap
    # -----------------------------
    if os.path.exists(heatmap_path):
        elements.append(Paragraph("AI Heatmap Analysis", title_style))
        elements.append(Spacer(1, 10))
        elements.append(Image(heatmap_path, width=4 * inch, height=4 * inch))

    # -----------------------------
    # Build PDF
    # -----------------------------
    doc.build(elements)
    return pdf_path
