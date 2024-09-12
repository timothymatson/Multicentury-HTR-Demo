from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import TrOCRProcessor
from huggingface_hub import login
import gradio as gr
import numpy as np
import onnxruntime
import torch
import time
import os

from plotting_functions import PlotHTR
from segment_image import SegmentImage
from onnx_text_recognition import TextRecognition


LINE_MODEL_PATH = "Kansallisarkisto/multicentury-textline-detection"
REGION_MODEL_PATH = "Kansallisarkisto/court-records-region-detection"
TROCR_PROCESSOR_PATH = "Kansallisarkisto/multicentury-htr-model-onnx"
TROCR_MODEL_PATH = "Kansallisarkisto/multicentury-htr-model-onnx"

login(token=os.getenv("HF_TOKEN"), add_to_git_credential=True)

print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

def get_segmenter():
    """Initialize segmentation class."""
    try:
        segmenter = SegmentImage(line_model_path=LINE_MODEL_PATH, 
                            device='cuda:0', 
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
                        processor_path = TROCR_PROCESSOR_PATH, 
                        model_path = TROCR_MODEL_PATH, 
                        device = '0', 
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
    with gr.Tab("Text content"):
        with gr.Row():
            input_img = gr.Image(label="Input image", type="pil")
            textbox = gr.Textbox(label="Predicted text content", lines=10)
        button = gr.Button("Process image")
        processing_time = gr.Markdown()
    with gr.Tab("Text regions"):
        region_img = gr.Image(label="Predicted text regions", type="numpy")
        gr.Markdown(color_codes)
    with gr.Tab("Text lines"):
        line_img = gr.Image(label="Predicted text lines", type="numpy")
        gr.Markdown(color_codes)
        
    def run_pipeline(image):
        # Predict region and line segments
        start = time.time()
        segment_predictions = segmenter.get_segmentation(image)
        print('segmentation ok')
        if segment_predictions:
            region_plot = plotter.plot_regions(segment_predictions, image)
            line_plot = plotter.plot_lines(segment_predictions, image)
            text_predictions = get_text_predictions(np.array(image), segment_predictions, recognizer)
            print('text pred ok')
            text = "\n".join(text_predictions)
            end = time.time()
            proc_time = end - start
            proc_time_str = f"Processing time: {proc_time:.4f}s"
            return {
                region_img: region_plot, 
                line_img: line_plot, 
                textbox: text,
                processing_time: proc_time_str
                }
        else:
            end = time.time()
            proc_time = end - start
            proc_time_str = f"Processing time: {proc_time:.4f}s"
            return {
                region_img: None, 
                line_img: None, 
                textbox: None,
                processing_time: proc_time_str
                }


    button.click(fn=run_pipeline, 
                 inputs=input_img, 
                 outputs=[region_img, line_img, textbox, processing_time])

if __name__ == "__main__":
    demo.queue()
    demo.launch(show_error=True)
