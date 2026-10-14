"""Create the fictional sample handbook used by the demo and tests."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "sample_data" / "reporting_handbook.pdf"

PAGES = [
    (
        "Reporting schedule",
        [
            "The monthly reporting deadline is the fifth business day of each month. "
            "The report covers the previous calendar month.",
            "The data analyst prepares the monthly dashboard. The BI team lead reviews "
            "the dashboard before publication. The approved dashboard is shared with "
            "the operations manager.",
            "Each report must include net revenue, completed orders, and return rate. "
            "The analyst records the reporting period and the last data refresh date.",
        ],
    ),
    (
        "Metric definitions",
        [
            "Net revenue is calculated as gross sales minus discounts minus refunds. "
            "For example, gross sales of 10,000 JOD, discounts of 500 JOD, and refunds "
            "of 300 JOD produce net revenue of 9,200 JOD.",
            "The return rate is returned orders divided by completed orders, multiplied "
            "by 100. For example, 25 returned orders out of 500 completed orders give "
            "a return rate of 5 percent.",
            "Completed orders exclude cancelled orders. All amounts in the sample "
            "dashboard are reported in Jordanian dinars (JOD).",
        ],
    ),
    (
        "Data quality checks",
        [
            "Remove duplicate records using order_id before calculating metrics. "
            "Keep the latest record when the same order_id appears more than once.",
            "Rows with a missing order_id must be moved to a review file. Do not "
            "include these rows in the published metrics until the identifier is fixed.",
            "Check missing values, invalid dates, and unexpected negative amounts. "
            "Compare total sales with the source report and document any difference "
            "before publishing the dashboard.",
        ],
    ),
]


def create_pdf():
    OUTPUT.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Body",
            fontName="Helvetica",
            fontSize=12,
            leading=19,
            textColor=colors.HexColor("#243247"),
            spaceAfter=16,
        )
    )
    story = []
    for number, (title, paragraphs) in enumerate(PAGES, start=1):
        if number > 1:
            story.append(PageBreak())
        story.append(Paragraph("Monthly Reporting Handbook", styles["Title"]))
        story.append(
            Paragraph("Fictional sample data for the PDF QA bot", styles["Normal"])
        )
        story.append(Spacer(1, 30))
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 12))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["Body"]))

    def footer(canvas, document):
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(50, 35, "Demo document - not a real company policy")
        canvas.drawRightString(A4[0] - 50, 35, f"Page {document.page}")

    SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=60,
        title="Monthly Reporting Handbook",
        author="PDF QA Bot Demo",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print("Created sample_data/reporting_handbook.pdf")


if __name__ == "__main__":
    create_pdf()
