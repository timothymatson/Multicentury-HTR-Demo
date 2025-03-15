from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import math
import cv2

class TextRecognition:
    def __init__(self, 
                model_path, 
                device = 'cuda:0', 
                half_precision = False,
                line_threshold = 10):
        self.device = device
        self.half_precision = half_precision
        self.line_threshold = line_threshold
        self.model_path = model_path
        self.processor = self.init_processor()
        print(self.processor)
        self.recognition_model = self.init_recognition_model()
        
    def init_processor(self):
        """Function for initializing the processor."""
        try:
            return TrOCRProcessor.from_pretrained(self.model_path, subfolder="processor")
        except Exception as e:
            print('Failed to initialize processor: %s' % e)
    
    def init_recognition_model(self):
        """Function for initializing the text detection model."""
        try:
            recognition_model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            return recognition_model
        except Exception as e:
            print('Failed to load the text recognition model: %s' % e)

    def crop_line(self, image, polygon, height, width):
        """Crops predicted text line based on the polygon coordinates
        and returns binarised text line image."""
        polygon = np.array([[int(lst[0]), int(lst[1])] for lst in polygon], dtype=np.int32)
        rect = cv2.boundingRect(polygon)
        cropped_image = image[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        mask = np.zeros([cropped_image.shape[0], cropped_image.shape[1]], dtype=np.uint8)
        cv2.drawContours(mask, [polygon- np.array([[rect[0],rect[1]]])], -1, (255, 255, 255), -1, cv2.LINE_AA)
        res = cv2.bitwise_and(cropped_image, cropped_image, mask = mask)
        wbg = np.ones_like(cropped_image, np.uint8)*255
        cv2.bitwise_not(wbg,wbg, mask=mask)
        # Overlap the resulted cropped image on the white background
        dst = wbg+res
        return dst

    def crop_lines(self, polygons, image, height, width):
        """Returns a list of line images cropped following the detected polygon coordinates."""
        cropped_lines = []
        for i, polygon in enumerate(polygons):
            cropped_line = self.crop_line(image, polygon, height, width)
            cropped_lines.append(cropped_line)
        return cropped_lines
    
    def get_scores(self, lgscores):
        """Get exponent of log scores."""
        scores = []
        for lgscore in lgscores:
            score = math.exp(lgscore)
            scores.append(score)
        return scores

    def predict_text(self, cropped_lines):
        """Functions for predicting text content from the cropped line images."""
        pixel_values = self.processor(cropped_lines, return_tensors="pt").pixel_values
        generated_dict = self.recognition_model.generate(pixel_values.to(self.device), max_new_tokens=128, return_dict_in_generate=True, output_scores=True)
        generated_ids, lgscores = generated_dict['sequences'], generated_dict['sequences_scores']
        scores = self.get_scores(lgscores.tolist())
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return scores, generated_text

    def get_text_lines(self, cropped_lines):
        scores, generated_text = [], []
        if len(cropped_lines) <= self.line_threshold:
            scores, generated_text = self.predict_text(cropped_lines)
        else:
            n = math.ceil(len(cropped_lines) / self.line_threshold)
            for i in range(n):
                print(i)
                start = int(i * self.line_threshold)
                end = int(min(start + self.line_threshold, len(cropped_lines)))
                sc, gt = self.predict_text(cropped_lines[start:end])
                scores += sc
                print(gt)
                generated_text += gt
        return scores, generated_text
            
    def get_res_dict(self, polygons, generated_text, height, width, image_name, line_confs, scores):
        """Combines the results in a dictionary form."""
        line_dicts = []
        for i in range(len(generated_text)):
            line_dict = {'polygon': polygons[i], 'text': generated_text[i], 'conf': line_confs[i], 'text_conf':scores[i]}
            line_dicts.append(line_dict)
        lines_dict = {'img_name': image_name, 'height': height, 'width': width, 'text_lines': line_dicts}
        return lines_dict

    def process_lines(self, polygons, image, height, width):
        # Crop line images
        print('starting text generation')
        cropped_lines = self.crop_lines(polygons, image, height, width)
        print('cropped lines')
        # Get text predictions
        scores, generated_text = self.get_text_lines(cropped_lines)
        return generated_text
