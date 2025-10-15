import gradio as gr
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

from ML_TeamProject import CompactGoodreadsAnalyzer

PROJECT_DESCRIPTION = """
# Goodreads Review Popularity Classifier
Predict whether a Goodreads review is *popular* or not using textual, semantic, and non-textual features.
"""

analyzer = None


def run_analysis(json_path, sample_size=5000):
    global analyzer
    analyzer = CompactGoodreadsAnalyzer(json_path, sample_size)
    analyzer.load_data()
    analyzer.define_popularity()
    analyzer.engineer_features()
    analyzer.prepare_data()
    analyzer.train_models()
    analyzer.print_results()
    analyzer.plot_results()

    summary = "\n".join([
        f"{model}** → Accuracy: {res['accuracy']:.3f}, F1: {res['f1']:.3f}, ROC-AUC: {res['roc_auc']:.3f}"
        for model, res in analyzer.results.items()
    ])
    return summary, "results.png"


def predict_review(review_text, rating=3, n_comments=0):
    if not analyzer or not analyzer.models:
        return "⚠ Please run the analysis first."

    row = pd.Series({"rating": rating, "n_comments": n_comments, "read_at": None, "started_at": None})
    text = analyzer.clean_text(review_text)
    features = analyzer.extract_features(text, row)
    feat_df = pd.DataFrame([features]).fillna(0)

    # safe alignment
    expected = analyzer.feature_df.shape[1]
    current = feat_df.shape[1]
    if current < expected:
        for i in range(expected - current):
            feat_df[f"dummy_{i}"] = 0
    X = feat_df.iloc[:, :expected]

    best_model = max(analyzer.results.items(), key=lambda x: x[1]["accuracy"])[0]
    model = analyzer.models[best_model]

    try:
        pred_proba = model.predict_proba(X)[0, 1] if best_model != "NN" else model.predict(X, verbose=0)[0][0]
    except Exception as e:
        return f"⚠ Error during prediction: {e}"

    popularity = "Popular" if pred_proba > 0.95 else "Not Popular"
    return f"*Prediction:* {popularity}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(PROJECT_DESCRIPTION)

    with gr.Tab("Run Analysis"):
        file_input = gr.File(label="Upload Goodreads JSON File")
        sample_slider = gr.Slider(1000, 100000, value=5000, step=1000, label="Sample Size")
        run_button = gr.Button("Run Model Training")
        output_summary = gr.Markdown()
        output_plot = gr.Image()
        run_button.click(run_analysis, inputs=[file_input, sample_slider], outputs=[output_summary, output_plot])

    with gr.Tab("Predict Review Popularity"):
        review_text = gr.Textbox(label="Enter Review Text", lines=5)
        rating_input = gr.Slider(1, 5, step=1, value=3, label="Rating")
        comments_input = gr.Slider(0, 50, step=1, value=0, label="Number of Comments")
        predict_button = gr.Button("Predict Popularity")
        pred_output = gr.Markdown()
        predict_button.click(predict_review, inputs=[review_text, rating_input, comments_input], outputs=[pred_output])

demo.launch()