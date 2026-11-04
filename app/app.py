"""Upload a waste photo and get a suggested category."""

import argparse

import gradio as gr

from waste_classifier.model import load_checkpoint
from waste_classifier.predict import predict_image
from waste_classifier.utils import get_device

DESCRIPTION = (
    "Upload a photo of a waste item to get a suggested category. "
    "This is a suggestion only: check your local disposal rules before you decide."
)


def build_app(checkpoint: str) -> gr.Interface:
    device = get_device()
    model, meta = load_checkpoint(checkpoint, device)

    def classify(image):
        return predict_image(model, meta, image, device)

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Waste photo"),
        outputs=gr.Label(num_top_classes=len(meta["classes"]), label="Suggested category"),
        title="Waste Classifier",
        description=DESCRIPTION,
        flagging_mode="never",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/transfer_large.pt")
    args = parser.parse_args()
    build_app(args.checkpoint).launch()
