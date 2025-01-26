from fpdf import FPDF

def generate_pdf_report(data, results, bankruptcy_status):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Bankruptcy Prediction Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Bankruptcy Status: {bankruptcy_status}", ln=True)

    pdf.cell(200, 10, txt="Model Results:", ln=True)

    # Loop through the results and print each model's information
    for result in results:
        # Ensure 'accuracy' (or 'test_accuracy') exists in the result
        accuracy_key = 'test_accuracy'  # Adjust if you are using a different key
        if accuracy_key in result:
            pdf.cell(200, 10, txt=f"{result['name']}: {result[accuracy_key]:.2f}", ln=True)
        else:
            pdf.cell(200, 10, txt=f"{result['name']}: Accuracy Not Available", ln=True)

    pdf.cell(200, 10, txt="Summary Visualizations Included.", ln=True)

    # Save the PDF
    pdf.output("bankruptcy_report.pdf")

