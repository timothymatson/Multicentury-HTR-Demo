import gradio as gr
import numpy as np
import time
import os
from PIL import Image
import torch

from plotting_functions import PlotHTR
from segment_image import SegmentImage
from text_recognition import TextRecognition

LINE_MODEL_PATH = "Kansallisarkisto/multicentury-textline-detection"
REGION_MODEL_PATH = "Kansallisarkisto/court-records-region-detection"
TROCR_MODEL_PATH = "Kansallisarkisto/multicentury-htr-model"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_segmenter():
    """Initialize segmentation class."""
    try:
        segmenter = SegmentImage(line_model_path=LINE_MODEL_PATH, 
                            device=device, 
                            line_iou=0.3,
                            region_iou=0.5,
                            line_overlap=0.5,
                            line_nms_iou=0.7,  
                            region_nms_iou=0.3,
                            line_conf_threshold=0.25, 
                            region_conf_threshold=0.5,  
                            region_model_path=REGION_MODEL_PATH, 
                            order_regions=True, 
                            region_half_precision=False, 
                            line_half_precision=False)
        return segmenter
    except Exception as e:
        print('Failed to initialize SegmentImage class: %s' % e)

def get_recognizer():
    """Initialize text recognition class."""
    try:
        recognizer = TextRecognition(
                        model_path = TROCR_MODEL_PATH, 
                        device = device, 
                        half_precision = True,
                        line_threshold = 10
                    )
        return recognizer
    except Exception as e:
        print('Failed to initialize TextRecognition class: %s' % e)

segmenter = get_segmenter()
recognizer = get_recognizer()
plotter = PlotHTR()

color_codes = """**Text region type:** <br>
        Paragraph ![#EE1289](https://placehold.co/15x15/EE1289/EE1289.png) 
        Marginalia ![#00C957](https://placehold.co/15x15/00C957/00C957.png) 
        Page number ![#0000FF](https://placehold.co/15x15/0000FF/0000FF.png)"""

def merge_lines(segment_predictions):
    img_lines = []
    for region in segment_predictions:
        img_lines += region['lines']
    return img_lines

def get_text_predictions(image, segment_predictions, recognizer):
    """Collects text prediction data into dicts based on detected text regions."""
    img_lines = merge_lines(segment_predictions)
    height, width = segment_predictions[0]['img_shape']
    # Process all lines of an image
    texts = recognizer.process_lines(img_lines, image, height, width)
    return texts

# Run demo code
with gr.Blocks(theme=gr.themes.Monochrome(), title="Multicentury HTR Demo") as demo:
    gr.Markdown("# Multicentury HTR Demo")
    gr.Markdown("""The HTR pipeline contains three components: text region detection, textline detection and handwritten text recognition.
    The components run machine learning models that have been trained at the National Archives of Finland using mostly handwritten documents
    from 17th, 18th, 19th and 20th centuries. 
    """)
    
    with gr.Tab("Text content"):
        with gr.Row():
            input_imgs = gr.File(label="Input images", file_count="multiple", file_types=["image"])
            textboxes = gr.Textbox(label="Predicted text content", lines=10, interactive=False, show_label=False)
        button = gr.Button("Process images")
        processing_time = gr.Markdown()
    with gr.Tab("Text regions"):
        region_imgs = gr.Gallery(label="Predicted text regions")
        gr.Markdown(color_codes)
    with gr.Tab("Text lines"):
        line_imgs = gr.Gallery(label="Predicted text lines")
        gr.Markdown(color_codes)


    def run_pipeline(files):
        images = []
        for file in files:
            try:
                img = Image.open(file.name)
                images.append((os.path.basename(file.name), img))
            except Exception as e:
                print(f"Error opening image {file.name}: {e}")
                continue

        results = []
        start = time.time()
        for file_name, image in images:
            segment_predictions = segmenter.get_segmentation(image)
            if segment_predictions:
                region_plot = plotter.plot_regions(segment_predictions, image)
                line_plot = plotter.plot_lines(segment_predictions, image)
                text_predictions = get_text_predictions(np.array(image), segment_predictions, recognizer)
                text = f"{file_name}:\n" + "\n".join(text_predictions)
                results.append((region_plot, line_plot, text))
        end = time.time()
        proc_time = end - start
        proc_time_str = f"Processing time: {proc_time:.4f}s"
        region_plots, line_plots, texts = zip(*results) if results else ([], [], [])
        return {
            region_imgs: list(region_plots),
            line_imgs: list(line_plots),
            textboxes: "\n\n".join(texts),
            processing_time: proc_time_str
        }

    button.click(fn=run_pipeline, inputs=input_imgs, outputs=[region_imgs, line_imgs, textboxes, processing_time])

if __name__ == "__main__":
    demo.queue()
    demo.launch(show_error=True)
