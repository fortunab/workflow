from shiny import App, ui, render, reactive
import requests

API_URL = "http://localhost:8000/predict"

app_ui = ui.page_fluid(
    ui.h2("Workflow-Centric Medical AI Demo"),
    ui.p("Upload a medical/endoscopic image. The app calls the FastAPI pipeline and returns structured tokens and a clinical-style report."),
    ui.input_file("image", "Upload image", accept=[".jpg", ".jpeg", ".png"]),
    ui.h4("Structured tokens"),
    ui.output_text_verbatim("tokens"),
    ui.h4("Generated report"),
    ui.output_text_verbatim("report")
)

def server(input, output, session):
    @reactive.Calc
    def api_result():
        file_info = input.image()
        if not file_info:
            return None

        datapath = file_info[0]["datapath"]

        with open(datapath, "rb") as f:
            response = requests.post(API_URL, files={"file": f}, timeout=120)

        response.raise_for_status()
        return response.json()

    @output
    @render.text
    def tokens():
        result = api_result()
        if result is None:
            return "Upload an image to run the pipeline."
        return "\n".join(result.get("tokens", []))

    @output
    @render.text
    def report():
        result = api_result()
        if result is None:
            return ""
        return result.get("report", "")

app = App(app_ui, server)
