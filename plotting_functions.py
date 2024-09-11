import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class PlotHTR:
    def __init__(self, region_alpha = 0.3, line_alpha = 0.3):
        self.region_colors = {'marginalia': (0,201,87), 'page-number': (0,0,255), 'paragraph': (238,18,137), 'case-start': (238,18,137)}
        self.region_alpha = region_alpha 
        self.line_alpha = line_alpha 

    def plot_regions(self, region_data, image):
        """Plots the detected text regions."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for region in region_data:
            region_type = region['region_name']
            img_h, img_w = region['img_shape']
            region_polygon = list(map(tuple, region['region_coords']))
            
            color = self.region_colors[region_type]
            
            draw.polygon(region_polygon, fill = color, outline='black')
            # Lower alpha value is applied to filled polygons to make them transparent
        blended_img = Image.blend(image, img, self.region_alpha)
        return blended_img

    def plot_lines(self, region_data, image):
        """Plots the detected text lines."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for region in region_data:
            img_h, img_w = region['img_shape']
            line_polygons = region['lines']
            region_type = region['region_name']
            color = self.region_colors[region_type]
            for i, polygon in enumerate(line_polygons):
                draw.polygon(polygon, fill = color)
        # Lower alpha value is applied to filled polygons to make them transparent
        blended_img = Image.blend(image, img, self.line_alpha)
        return blended_img
