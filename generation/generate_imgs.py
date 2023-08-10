import cv2
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import math, random
from typing import List, Tuple

class BackgroundDataset():
    """Face Landmarks dataset."""

    def __init__(self, 
                 root_dir = "/datasets/COCO-2017/train2017/",
                 label_file = "/datasets/COCO-2017/anno2017/instances_train2017.json",
                 save_path = "/training_inputs",
                 irregularity = 0.9, 
                 spikiness = 0.1
                ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.save_path = save_path
        with open(label_file, 'r') as f:
            self.data = json.load(f)
        self.irregularity = irregularity
        self.spikiness = spikiness
        self.rad_low = 0.12
        self.rad_high = 0.2

    def clip(self, value, lower, upper):
        """
        Given an interval, values outside the interval are clipped to the interval
        edges.
        """
        return min(upper, max(value, lower))

    def random_angle_steps(self, steps: int, irregularity: float) -> List[float]:
        """Generates the division of a circumference in random angles.

        Args:
            steps (int):
                the number of angles to generate.
            irregularity (float):
                variance of the spacing of the angles between consecutive vertices.
        Returns:
            List[float]: the list of the random angles.
        """
        # generate n angle steps
        angles = []
        lower = (2 * math.pi / steps) - irregularity
        upper = (2 * math.pi / steps) + irregularity
        cumsum = 0
        for i in range(steps):
            angle = random.uniform(lower, upper)
            angles.append(angle)
            cumsum += angle

        # normalize the steps so that point 0 and point n+1 are the same
        cumsum /= (2 * math.pi)
        for i in range(steps):
            angles[i] /= cumsum
        return angles

    def generate_polygon(self, center: Tuple[float, float], avg_radius: float,
                        irregularity: float, spikiness: float,
                        num_vertices: int) -> List[Tuple[float, float]]:
        """
        Start with the center of the polygon at center, then creates the
        polygon by sampling points on a circle around the center.
        Random noise is added by varying the angular spacing between
        sequential points, and by varying the radial distance of each
        point from the centre.

        Args:
            center (Tuple[float, float]):
                a pair representing the center of the circumference used
                to generate the polygon.
            avg_radius (float):
                the average radius (distance of each generated vertex to
                the center of the circumference) used to generate points
                with a normal distribution.
            irregularity (float):
                variance of the spacing of the angles between consecutive
                vertices.
            spikiness (float):
                variance of the distance of each vertex to the center of
                the circumference.
            num_vertices (int):
                the number of vertices of the polygon.
        Returns:
            List[Tuple[float, float]]: list of vertices, in CCW order.
        """
        # Parameter check
        if irregularity < 0 or irregularity > 1:
            raise ValueError("Irregularity must be between 0 and 1.")
        if spikiness < 0 or spikiness > 1:
            raise ValueError("Spikiness must be between 0 and 1.")

        irregularity *= 2 * math.pi / num_vertices
        spikiness *= avg_radius
        angle_steps = self.random_angle_steps(num_vertices, irregularity)

        # now generate the points
        points = []
        angle = random.uniform(0, 2 * math.pi)
        for i in range(num_vertices):
            radius = self.clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
            point = (center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle))
            points.append(point)
            angle += angle_steps[i]

        return points
    
    def generate_random_polygons(self, shape):

        rad_per = random.uniform(self.rad_low, self.rad_high)
        avg_rad = int(min(shape[0], shape[1]) * rad_per)
        num_vertices = random.randint(45, 65)
        center_x = random.randint(avg_rad, shape[0]-avg_rad)
        center_y = random.randint(avg_rad, shape[1]-avg_rad)
        vertices = self.generate_polygon(center=(center_x, center_y),
                            avg_radius=avg_rad,
                            irregularity= self.irregularity,
                            spikiness=self.spikiness,
                            num_vertices=num_vertices)
        return vertices
    
    def generate_segmented_img(self, shape, polygons, value=True):
        mask = np.ones((shape[0], shape[1]), dtype = np.uint8)*255
        for coords in polygons:
            coords = np.array(coords).astype(np.int32)
            coords = coords.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [coords], color=(0,0))
        return mask
    
    def generate_random_image(self, shape, num_segments):
        mask = np.ones((shape[0], shape[1]), dtype = np.uint8)*255
        num_rand = int(0.4*num_segments)
        if num_rand < 2:
            num_rand = 2
        while num_rand > 0:
            vertices = self.generate_random_polygons(mask.shape)
            vertices = np.array(vertices).astype(np.int32)
            vertices = vertices.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [vertices], color=(0,0))
            num_rand -= 1
        return mask

    def _get_labels(self, img_id):
        polygons = []
        for annotation in self.data['annotations']:
            if annotation['image_id'] == img_id:
                if len(annotation['segmentation']) < 1:
                    continue
                try:
                    segmentation = annotation['segmentation'][0]
                except Exception as err:
                    continue
                pair_of_tuples = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
                polygons.append(pair_of_tuples)
        return polygons

    def __len__(self):
        return len(self.data['annotations'])
    
    def gen_images(self):
        
        from tqdm import tqdm
        for idx in tqdm(range(len(self.data['annotations']))):
            try:
                img_id = self.data['annotations'][idx]['image_id']
                image_id = "{:012d}.jpg".format(img_id)
                # print(image_id)
                img_name = os.path.join(self.root_dir, image_id)

                image = cv2.imread(img_name)
                
                image = image/255

                segmented_polygons = self._get_labels(img_id)
                num_segments = len(segmented_polygons)

                seg_mask = self.generate_segmented_img(image.shape, segmented_polygons)

                rand_mask = self.generate_random_image(image.shape, num_segments)

                mask = np.logical_and(seg_mask==255, np.logical_not(rand_mask==255))

                gen_img = image + np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                
                gen_img = np.clip(gen_img, 0, 1)
                
                gen_img = gen_img.astype(np.float32)
                
                idimg = "{:012d}.jpg".format(img_id)
                
                file_path = f"{self.save_path}/{idimg}"

                cv2.imwrite(file_path, (gen_img * 255).astype(np.uint8))
                        
            except Exception as err:
                print(err)
                break
                continue
    

directory = "training_inputs"

if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully.")
else:
    print(f"Directory '{directory}' already exists.")
    
root_dir = "/datasets/COCO-2017/train2017/",
label_file = "/datasets/COCO-2017/anno2017/instances_train2017.json",
save_path = "training_inputs",
bg = BackgroundDataset(root_dir, label_file, save_path)
bg.gen_images()

directory = "testing_inputs"

if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully.")
else:
    print(f"Directory '{directory}' already exists.")

root_dir = "/datasets/COCO-2017/val2017/",
label_file = "/datasets/COCO-2017/anno2017/instances_val2017.json",
save_path = "testing_inputs",
bg = BackgroundDataset(root_dir, label_file, save_path)
bg.gen_images()