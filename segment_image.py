from shapely.validation import make_valid
from shapely.geometry import Polygon
from ultralyticsplus import YOLO
from PIL import Image
import numpy as np
import os

from reading_order import OrderPolygons

class SegmentImage:
    """Class for segmenting document image regions and text lines."""
    def __init__(self, 
                line_model_path, 
                device, 
                line_iou=0.5,
                region_iou=0.5,
                line_overlap=0.5, 
                line_nms_iou=0.7,
                region_nms_iou=0.3, 
                line_conf_threshold=0.25, 
                region_conf_threshold=0.25, 
                region_model_path=None, 
                order_regions=True, 
                region_half_precision=False, 
                line_half_precision=False):

        # Path to text line detection model
        self.line_model_path = line_model_path 
        # Path to text region detection model
        self.region_model_path = region_model_path 
        # Defines the IoU threshold used in the non-maximum suppression (NMS) process to 
        # determine which prediction boxes should be suppressed or discarded based on their overlap with other boxes
        self.line_nms_iou = line_nms_iou
        self.region_nms_iou = region_nms_iou
        # Defines the IoU threshold for text lines
        self.line_iou = line_iou  
        # Defines the IoU threshold for text regions
        self.region_iou = region_iou 
        # Defines the extent of line polygon overlap used for merging the polygons
        self.line_overlap = line_overlap  
        # Defines confidence threshold for line detection
        self.line_conf_threshold = line_conf_threshold 
        # Defines confidence threshold for region detection
        self.region_conf_threshold = region_conf_threshold 
        # Defines the device to be used ('cpu', gpu '0', gpu '1' etc.)
        self.device = device 
        # Defines whether a reading order is also estimated for the region detections
        self.order_regions = order_regions 
        # Defines whether half precision (FP16) is used by the region and line prediction models
        self.region_half_precision = region_half_precision 
        self.line_half_precision = line_half_precision 
        self.order_poly = OrderPolygons()
        # Initialize segmentation model(s)
        self.line_model = self.init_line_model()
        if self.region_model_path:
            self.region_model = self.init_region_model()

    def init_line_model(self):
        """Function for initializing the line detection model."""
        try:
            # Load the trained line detection model
            line_model = YOLO(self.line_model_path, hf_token=os.getenv("HF_TOKEN"))
            return line_model
        except Exception as e:
            print('Failed to load the line detection model: %s' % e)

    def init_region_model(self):
        """Function for initializing the region detection model."""
        try:
            # Load the trained line detection model
            region_model = YOLO(self.region_model_path, hf_token=os.getenv("HF_TOKEN"))
            return region_model
        except Exception as e:
            print('Failed to load the region detection model: %s' % e)
        
    def get_region_ids(self, coords, max_min, classes, names, box_confs, img_shape):
        """Function for creating unique id for each detected region."""
        n = min(len(classes), len(coords))
        res = []
        for i in range(n):
            # Creates a simple index-based id for each region
            region_id = str(i)
            # Extracts region name corresponding to the index
            region_type = names[classes[i]] 
            poly_dict = {'coords': coords[i], 
                        'max_min': max_min[i], 
                        'class': str(classes[i]), 
                        'name': region_type, 
                        'conf': box_confs[i],
                        'id': region_id, 
                        'img_shape': img_shape}
            res.append(poly_dict)
        return res

    def get_max_min(self, polygons):
        """Creates an array with the minimum and maximum 
        x and y values of the input polygons."""
        n_rows = len(polygons)
        xy_array = np.zeros([n_rows, 4])
        for i, poly in enumerate(polygons):
            x = [point[0] for point in poly]
            y = [point[1] for point in poly]
            if x:
                xy_array[i,0] = max(x)
                xy_array[i,1] = min(x)
            if y:
                xy_array[i,2] = max(y)
                xy_array[i,3] = min(y)
        return xy_array

    def validate_polygon(self, polygon):
        """"Function for testing and correcting the validity of polygons."""
        if len(polygon) > 2:
            polygon = Polygon(polygon)
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            return polygon
        else:
            return None

    def get_iou(self, poly1, poly2):
        """Function for calculating Intersection over Union (IoU) values."""
        # If the polygons don't intersect, IoU is 0
        iou = 0
        poly1 = self.validate_polygon(poly1)
        poly2 = self.validate_polygon(poly2)

        if poly1 and poly2:
            if poly1.intersects(poly2):
                # Calculates intersection of the 2 polygons
                intersect = poly1.intersection(poly2).area
                # Calculates union of the 2 polygons
                uni = poly1.union(poly2)
                # Calculates intersection over union
                iou = intersect / uni.area
        return iou

    def merge_polygons(self, polygons, iou_threshold, overlap_threshold = None):
        """Merges polygons that have an IoU value 
        above the given threshold."""
        new_polygons = []
        dropped = set()
        # Loops over all input polygons and merges them if the
        # IoU value is over the given threshold
        for i in range(0, len(polygons)):
            poly1 = self.validate_polygon(polygons[i])
            merged = None
            for j in range(i+1, len(polygons)):
                poly2 = self.validate_polygon(polygons[j])
                if poly1 and poly2: 
                    if poly1.intersects(poly2):
                        overlap = False
                        intersect = poly1.intersection(poly2)
                        uni = poly1.union(poly2)
                        # Calculates intersection over union
                        iou = intersect.area / uni.area
                        if overlap_threshold:
                            overlap = intersect.area > (overlap_threshold * min(poly1.area, poly2.area))
                        if (iou > iou_threshold) or overlap:
                            if merged:
                                # If there are multiple overlapping polygons
                                # with IoU over the threshold, they are all merged together
                                merged = uni.union(merged)
                                dropped.add(j)
                            else:
                                merged = uni
                                # Polygons that are merged together are dropped from
                                # the list
                                dropped.add(i)
                                dropped.add(j)       
            if merged:
                if merged.geom_type in ['GeometryCollection','MultiPolygon']:
                    for geom in merged.geoms:                
                        if geom.geom_type == 'Polygon':
                            new_polygons.append(list(geom.exterior.coords))
                elif merged.geom_type == 'Polygon':
                    new_polygons.append(list(merged.exterior.coords))
        res = [i for j, i in enumerate(polygons) if j not in dropped]
        res += new_polygons
        
        return res

    def get_region_preds(self, img):
        """Function for predicting text region coordinates."""
        results = self.region_model.predict(source=img,     
                                            device=self.device, 
                                            conf=self.region_conf_threshold, 
                                            half=bool(self.region_half_precision), 
                                            iou=self.region_nms_iou)
        results = results[0].cpu()
        if results.masks:
            # Extracts detected region polygons
            coords = results.masks.xy
            # Merge overlapping polygons
            coords = self.merge_polygons(coords, self.region_iou)
            # Maximum and minimum x and y axis values for detected polygons used for ordering the polygons
            max_min = self.get_max_min(coords).tolist() 
            # Gets a list of the predicted class labels for detected regions
            classes = results.boxes.cls.tolist() 
            # A dictionary with class ids as keys and class names as values
            names = results.names 
            # Confidence values for detections
            box_confs = results.boxes.conf.tolist()
            # A tuple containing the shape of the original image
            img_shape = results.orig_shape 
            res = self.get_region_ids(list(coords), max_min, classes, names, box_confs, img_shape)
            return res
        else:
            return None


    def get_line_preds(self, img):
        """Function for predicting text line coordinates."""
        results = self.line_model.predict(source=img, 
                                        device=self.device, 
                                        conf=self.line_conf_threshold, 
                                        half=bool(self.line_half_precision),
                                        iou=self.line_nms_iou)
        results = results[0].cpu()
        if results.masks:
            # Detected text line polygons 
            coords = results.masks.xy
            # Merge overlapping polygons
            coords = self.merge_polygons(coords, self.line_iou, self.line_overlap)
            # Maximum and minimum x and y axis values for detected polygons
            max_min = self.get_max_min(coords).tolist()
            # Confidence values for detections
            box_confs = results.boxes.conf.tolist()
            res_dict = {'coords': list(coords), 'max_min': max_min, 'confs': box_confs}
            return res_dict
        else:
            return None

    def get_dist(self, line_polygon, regions):
        """Function for finding the closest region to the text line."""
        dist, reg_id = 1000000, None
        line_polygon = self.validate_polygon(line_polygon)

        if line_polygon:
            for region in regions:
                # Calculates dictance between line and regions polygons
                region_polygon = self.validate_polygon(region['coords'])
                if region_polygon:
                    line_reg_dist = line_polygon.distance(region_polygon)
                    if line_reg_dist < dist:
                        dist = line_reg_dist
                        reg_id = region['id']
        return reg_id
    
    def get_line_regions(self, lines, regions):
        """Function for connecting each text line to one region."""
        lines_list = []
        for i in range(len(lines['coords'])):
            iou, reg_id, conf = 0, '', 0.0
            max_min = [0.0, 0.0, 0.0, 0.0]
            polygon = lines['coords'][i]
            for region in regions:
                line_reg_iou = self.get_iou(polygon, region['coords']) 
                if line_reg_iou > iou:
                    iou = line_reg_iou
                    reg_id = region['id']
            # If line polygon does not intersect with any region, a distance metric is used for defining 
            # the region that the line belongs to
            if iou == 0:
                reg_id = self.get_dist(polygon, regions)

            if (len(lines['max_min']) - 1) >= i:
                max_min = lines['max_min'][i]
                
            if (len(lines['confs']) - 1) >= i:
                conf = lines['confs'][i]

            new_line = {'polygon': polygon, 'reg_id': reg_id, 'max_min': max_min, 'conf': conf}
            lines_list.append(new_line)
        return lines_list

    def order_regions_lines(self, lines, regions):
        """Function for ordering line predictions inside each region."""
        regions_with_rows = []
        region_max_mins = []
        for i, region in enumerate(regions):
            line_max_mins = []
            line_confs = []
            line_polygons = []
            for line in lines:
                if line['reg_id'] == region['id']:
                    line_max_mins.append(line['max_min'])
                    line_confs.append(line['conf'])
                    line_polygons.append(line['polygon'])
            if line_polygons:
                # If one or more lines are connected to a region, line order inside the region is defined
                # and the predicted text lines are joined in the same python dict
                line_order = self.order_poly.order(line_max_mins)
                line_polygons = [line_polygons[i] for i in line_order]
                line_confs = [line_confs[i] for i in line_order]
                new_region = {'region_coords': region['coords'], 
                            'region_name': region['name'], 
                            'lines': line_polygons, 
                            'line_confs': line_confs,
                            'region_conf': region['conf'],
                            'img_shape': region['img_shape']}
                region_max_mins.append(region['max_min'])
                regions_with_rows.append(new_region)
            else:
                continue
        # Creates an ordering of the detected regions based on their polygon coordinates
        if self.order_regions:
            region_order = self.order_poly.order(region_max_mins)
            regions_with_rows = [regions_with_rows[i] for i in region_order]
            
        return regions_with_rows

    def get_default_region(self, image):
        """Function for creating a default region if no regions are detected."""
        w, h = image.size 
        region = {'coords': [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], 
                        'max_min': [w, 0.0, h, 0.0], 
                        'class': '0', 
                        'name': "paragraph", 
                        'conf': 0.0,
                        'id': '0', 
                        'img_shape': (h, w)}
        return [region]

    def get_segmentation(self, image):
        """Segment input image into ordered text lines or ordered text regions and text lines."""
        line_preds = self.get_line_preds(image)
        if line_preds:
            # If region detection model is defined, text regions and text lines are detected
            region_preds = self.get_region_preds(image)
            if not region_preds:
                region_preds = self.get_default_region(image)
                print(f'No regions detected from image {image}')
            lines_with_regions = self.get_line_regions(line_preds, region_preds)
            ordered_regions = self.order_regions_lines(lines_with_regions, region_preds)
            return ordered_regions
        else:
            print(f'No text lines detected from image {image}')
            return None

    