# For machine learning
import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import requests
from io import BytesIO
import numpy as np
import json
import torch
from PIL import Image

from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
from sklearn import cluster

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.cluster import KMeans
from skimage.io import imread, imsave
from skimage.color import rgb2lab
import math

class ImageAnalysis:
    output_dir = "outputs"   
    temp_path = f'{output_dir}/original_image.png'
    face_landmarker_checkpoint = 'colorDetectionFastApi/checkpoints/face_landmarker_v2_with_blendshapes.task'
    hair_segmenter_checkpoint = 'colorDetectionFastApi/checkpoints/hair_segmenter.tflite' 
    box_threshold = 0.3   
    text_threshold = 0.25   
    device = "cuda"
    lum_threshold = 70
    hue_threshold = 68
    ita_threshold = 60
    hair_skin_ratio_threshold_1 = 0.4
    hair_skin_ratio_threshold_2 = 0.7

    def __init__(self, image_url: str):
        """Initializes a yolov8 detector with a binary image
        
        Arguments:
            chunked (bytes): A binary image representation
        """
        self.image_url = image_url
        self.detector = self._load_detector()
        self.segmenter = self._load_segmenter()
        

    def _load_detector(self):
        """Loads Yolo8 model from pytorch hub or a path on disk

        Returns:
            model (Model) - Trained Pytorch model
        """
        organs_base_options = python.BaseOptions(model_asset_path=ImageAnalysis.face_landmarker_checkpoint)
        organs_options = vision.FaceLandmarkerOptions(base_options=organs_base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(organs_options)
        return detector
    
    def _load_segmenter(self):
        """Loads Yolo8 model from pytorch hub or a path on disk

        Returns:
            model (Model) - Trained Pytorch model
        """
        hair_base_options = python.BaseOptions(model_asset_path=ImageAnalysis.hair_segmenter_checkpoint)
        hair_options = vision.ImageSegmenterOptions(base_options=hair_base_options,
                                            output_category_mask=True)
        segmenter = vision.ImageSegmenter.create_from_options(hair_options)
        return segmenter

    async def __call__(self):
        """This function is called when class is executed.
        It analyzes a single image passed to its constructor
        and returns the annotated image and its labels
        
        Returns:
            frame (numpy.ndarray): Frame with bounding boxes and labels ploted on it.
            labels (list(str)): The corresponding labels that were found
        """
        
        return analyze_image(self.image_url, self.detector, self.segmenter)
    

def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

def orientation_correction(img): 
    width, height = img.size
    if width > height: 
        img = img.rotate(90, expand=True)
    return img

def sort_points(points):
    
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sort_order = np.argsort(angles)
    return points[sort_order]

def extract_parts(face_landmarks, part_connections):
    
    initial_set = set()
    parts_dict = {}
    for connection in part_connections:
        initial_set.update(connection)
        for point in connection:
            connected_points = parts_dict.get(point, [])
            connected_points.extend([pt for pt in connection if pt != point])
            parts_dict[point] = connected_points
    
    def get_connected_component(start_point, visited):
        
        stack = [start_point]
        component = set()
        while stack:
            point = stack.pop()
            if point not in visited:
                visited.add(point)
                component.add(point)
                stack.extend(parts_dict[point])
        return list(component)

    visited = set()
    components = []
    for point in initial_set:
        if point not in visited:
            component = get_connected_component(point, visited)
            components.append(component)
    return components

def draw_parts_mask(rgb_image, detection_result, part_connections):
    
    face_landmarks_list = detection_result.face_landmarks
    mask_image = np.zeros_like(rgb_image)
    
    for face_landmarks in face_landmarks_list:
        components = extract_parts(face_landmarks, part_connections)
        area_and_components = []
        for component in components:
            points = np.array([(int(face_landmarks[landmark].x * rgb_image.shape[1]),
                                int(face_landmarks[landmark].y * rgb_image.shape[0])) for landmark in component])
            points = sort_points(points)
            area = cv2.contourArea(points.reshape(-1, 1, 2))
            area_and_components.append((area, points))

       
        if area_and_components:
            max_area_component = max(area_and_components, key=lambda x: x[0])[1]

           
            cv2.drawContours(mask_image, [max_area_component], -1, (255, 255, 255), thickness=cv2.FILLED)

      
            for _, component in area_and_components:
                if not np.array_equal(component, max_area_component):
                    cv2.drawContours(mask_image, [component], -1, (0,0,0), thickness=cv2.FILLED)

    # mask_image = cv2.bitwise_and(rgb_image, mask_image)
    return mask_image

def draw_mask(rgb_image, detection_result, parts):
  face_landmarks_list = detection_result.face_landmarks
  mask_image = np.zeros_like(rgb_image)
  
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    lips_dict = {}
    if parts == "lips": 
        masked_image = draw_parts_mask(rgb_image, detection_result, solutions.face_mesh.FACEMESH_LIPS)
    elif parts == "nose":
      masked_image = draw_parts_mask(rgb_image, detection_result, solutions.face_mesh.FACEMESH_NOSE)
    elif parts == "face oval": 
      face_oval_image  = draw_parts_mask(rgb_image, detection_result, solutions.face_mesh.FACEMESH_CONTOURS)
      nose_image  = draw_parts_mask(rgb_image, detection_result, solutions.face_mesh.FACEMESH_NOSE)
      masked_image =  face_oval_image & ~nose_image
    elif parts == "irises":
      left_iris_masked_image = draw_parts_mask(rgb_image, detection_result, solutions.face_mesh.FACEMESH_LEFT_IRIS)
      right_iris_masked_image = draw_parts_mask(rgb_image, detection_result, solutions.face_mesh.FACEMESH_RIGHT_IRIS)
      masked_image = left_iris_masked_image | right_iris_masked_image

    rgb_masked_image = cv2.bitwise_and(rgb_image, masked_image)
    masked_image = np.any(masked_image != 0, axis=2)  
  return masked_image, rgb_masked_image


def read_img(fpath):
    img = Image.open(fpath).convert('RGB')
    return img


def get_hue(a_values, b_values, eps=1e-8):
    """Compute hue angle"""
    return np.degrees(np.arctan(b_values / (a_values + eps)))


def mode_hist(x, bins='sturges'):
    """Compute a histogram and return the mode"""
    hist, bins = np.histogram(x, bins=bins)
    mode = bins[hist.argmax()]
    return mode


def clustering(x, n_clusters=5, random_state=2021):
    model = cluster.KMeans(n_clusters, random_state=random_state)
    model.fit(x)
    return model.labels_, model


def get_scalar_values(skin_smoothed_lab, labels, topk=3, bins='sturges'):
    # gather values of interest
    hue_angle = get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
    skin_smoothed = lab2rgb(skin_smoothed_lab)

    # concatenate data to be clustered (L, h, and RGB for visualization)
    data_to_cluster = np.vstack([skin_smoothed_lab[:, 0], hue_angle,
                                 skin_smoothed[:, 0], skin_smoothed[:, 1], skin_smoothed[:, 2]]).T

    # Extract skin pixels for each mask (by clusters)
    n_clusters = len(np.unique(labels))
    masked_skin = [data_to_cluster[labels == i, :] for i in range(n_clusters)]
    n_pixels = np.asarray([np.sum(labels == i) for i in range(n_clusters)])

    # get scalar values per cluster
    keys = ['lum', 'hue', 'red', 'green', 'blue']
    res = {}

    for i, key in enumerate(keys):
        res[key] = np.array([mode_hist(part[:, i], bins=bins)
                             for part in masked_skin])

    # only keep top3 in luminance and avarage results
    idx = np.argsort(res['lum'])[::-1][:topk]
    total = np.sum(n_pixels[idx])

    res_topk = {}
    for key in keys:
        res_topk[key] = np.average(res[key][idx], weights=n_pixels[idx])
        res_topk[key+'_std'] = np.sqrt(np.average((res[key][idx]-res_topk[key])**2, weights=n_pixels[idx]))
    return res_topk


def get_skin_values(img, mask, n_clusters=5):
    # smoothing
    img_smoothed = gaussian(img, sigma=(1, 1, 1), truncate=4)

    # get skin pixels (shape will be Mx3) and go to Lab
    skin_smoothed = img_smoothed[mask]
    skin_smoothed_lab = rgb2lab(skin_smoothed)

    res = {}

    # L and hue
    hue_angle = get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
    data_to_cluster = np.vstack([skin_smoothed_lab[:, 0], hue_angle]).T
    labels, model = clustering(data_to_cluster, n_clusters=n_clusters)
    tmp = get_scalar_values(skin_smoothed_lab, labels)
    res['lum'] = tmp['lum']
    res['hue'] = tmp['hue']
    res['lum_std'] = tmp['lum_std']
    res['hue_std'] = tmp['hue_std']

    # also extract RGB for visualization purposes
    res['red'] = tmp['red']
    res['green'] = tmp['green']
    res['blue'] = tmp['blue']
    res['red_std'] = tmp['red_std']
    res['green_std'] = tmp['green_std']
    res['blue_std'] = tmp['blue_std']

    return res

def visualize_masked_image(type, image, mask):
    image = np.array(image)
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(masked_image)
    ax[1].set_title(f'{type} Masked Image')
    ax[1].axis('off')

    plt.savefig(os.path.join(ImageAnalysis.output_dir, f'Masked_{type}_Image.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.show()
    
def get_mask_or_false(masked_data,key,mask_shape):
    # print(mask_shape)
    mask = masked_data.get(key, None)
    if mask is None:
        mask = np.full(mask_shape, False, dtype=bool)
    if mask.shape != mask_shape:
        print(mask.shape != mask_shape)
        print(mask.shape)
        new_mask = np.full(mask_shape, False, dtype=bool)
        
        min_shape = tuple(min(o, n) for o, n in zip(mask.shape, mask_shape))
     
        new_mask[tuple(slice(0, m) for m in min_shape)] = mask[tuple(slice(0, m) for m in min_shape)]
        # print(new_mask.shape)
        mask = new_mask
    return mask

def visualize_colors(type, colors, output_dir):
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        plt.fill_between([i, i + 1], 0, 1, color=color)
    plt.xlim(0, len(colors))
    plt.yticks([])
    plt.xticks(range(len(colors)), labels=[f"Color {i+1}" for i in range(len(colors))])
    plt.title(f'{type} Dominant Colors')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{type} Dominant_Colors.jpg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.show()

def draw_skin_color_scatterplot(skin_colors, indices=[]):
    hue_angle, lightness = zip(*skin_colors)

    plt.figure(figsize=(20, 12))
    plt.scatter(hue_angle, lightness, color="blue", marker="o")

    for i, (h, L) in enumerate(skin_colors):
        if len(indices) == 0:
            plt.text(h, L + 1, str(i), fontsize=9, ha="center")
        else:
            plt.text(h, L + 1, indices[i], fontsize=9, ha="center")

    plt.xlim(30, 80)
    plt.ylim(30, 80)

    plt.axvline(x=55, color='red', linestyle='--', linewidth=1)
    plt.axhline(y=60, color='grey', linestyle='--', linewidth=1)

    plt.title('Hue Angle vs Lightness')
    plt.xlabel('Hue Angle (h)')
    plt.ylabel('Lightness (L)')

    # Showing the plot
    plt.grid(True)
    plt.show()
    plt.savefig(
        os.path.join(ImageAnalysis.output_dir, 'skin_color_scatterplot.jpg'),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

def get_rgb_lh(type, image_mask, image, is_skin, lhs=[],colors=[]):

    visualize_masked_image(type, image, image_mask)
    attrslh = ['lum', 'hue']
    attrsrgb = ['red', 'green', 'blue']
    reslh = {}
    resrgb = {}

    img_original = image
    mask = image_mask

    # get values
    tmp = get_skin_values(np.asarray(img_original),
                            mask)

    for attr in attrslh:
        reslh[attr] = tmp[attr]
    for attr in attrsrgb:
        resrgb[attr] = tmp[attr]
        # res[attr+'_std'].append(tmp[attr+'_std'])

    
    colors.append(resrgb)
    # color_list = [(color['red'], color['green'], color['blue']) for color in colors]
    color_list = [(resrgb['red'], resrgb['green'], resrgb['blue'])]
    visualize_colors(type, color_list, ImageAnalysis.output_dir)
    if is_skin:
        lhs.append(reslh)
        hum_hue_list = [(lh['hue'], lh['lum']) for lh in lhs]
        draw_skin_color_scatterplot(hum_hue_list)
    
    # print(reslh)
    # print(resrgb)
    return resrgb, reslh
    
def extract_masked_pixels(image, mask):
    
    return np.array(image)[mask]

def kmeans_colors(pixels, n_clusters=1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_
    return colors

def visualize_colors_1(type, colors):
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        plt.fill_between([i, i + 1], 0, 1, color=color/255)
    plt.xlim(0, len(colors))
    plt.yticks([])
    plt.xticks(range(len(colors)))
    plt.title(f'{type} Dominant Colors')
    plt.show()
    plt.savefig(os.path.join(ImageAnalysis.output_dir, f'{type} Dominant Colors.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

def get_rgb(type,image_mask, image):
    masked_pixels = extract_masked_pixels(image, image_mask)
    masked_pixels = masked_pixels.reshape(-1, 3)
    n_clusters = 1 
    dominant_colors = kmeans_colors(masked_pixels, n_clusters)
    visualize_masked_image(type, image, image_mask)
    visualize_colors_1(type, dominant_colors)

    return dominant_colors

def rgb_to_hex(rgb):
    normalized_rgb = [int(round(x * 255 if x <= 1 else x)) for x in rgb]
    return '#' + ''.join(f"{value:02x}" for value in normalized_rgb) 

def get_ita(rgb_val):
    rgb_pixel = np.array([[rgb_val]], dtype=np.float32) * 255
    rgb_pixel = rgb_pixel.astype(np.uint8)
    
    lab_pixel = rgb2lab(rgb_pixel)
    lab_val = lab_pixel[0][0].tolist()
    
    lightness = lab_val[0]
    b_component = lab_val[2]
    
    if b_component == 0:
        b_component = 1e-10 
    
    ita = math.atan((lightness - 50) / b_component) * (180 / math.pi)
    return ita

def rgb_to_gray(rgb_color):

    rgb_color = np.array([int(255 * c) for c in rgb_color], dtype=np.uint8)

    color_image = np.tile(rgb_color, (100, 100, 1))

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    return gray_image[0, 0]

def rgb_to_saturation(rgb_color):
    rgb_color = np.array([int(255 * c) for c in rgb_color], dtype=np.uint8)
    color_image = np.full((1, 1, 3), rgb_color, dtype=np.uint8)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
    saturation = hsv_image[0, 0, 1]
    saturation = saturation / 255.0
    return saturation


def analyze_image(image_url, detector, segmenter):
    os.makedirs(ImageAnalysis.output_dir, exist_ok=True)
    img = download_image(image_url)
    image_pil = orientation_correction(img)
    image_pil.save(ImageAnalysis.temp_path)
    image = mp.Image.create_from_file(ImageAnalysis.temp_path)
    

    # image = mp.Image.create_from_file("/root/autodl-tmp/Grounded-Segment-Anything/faces/face.png")
    detection_result = detector.detect(image)
    iris_mask, masked_iris_image = draw_mask(image.numpy_view(), detection_result, "irises")
    face_mask, masked_face_skin_image = draw_mask(image.numpy_view(), detection_result, "face oval")
    lips_mask, masked_lips_image = draw_mask(image.numpy_view(), detection_result, "lips")
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask
    hair_mask = category_mask.numpy_view().astype(bool)


    skin_resrgb, skin_reslh = get_rgb_lh('skin', face_mask, image_pil, True)
    hair_resrgb, hair_reslh = get_rgb_lh('hair', hair_mask, image_pil, True)
    hair_color = get_rgb('hair',hair_mask,image_pil)
    mouth_color = get_rgb('lips',lips_mask,image_pil)
    
    output = {   
        'color': {  
            'rgb': {  
                'skin_rgb': [skin_resrgb['red'], skin_resrgb['green'], skin_resrgb['blue']],  
                'hair_rgb': hair_color[0].tolist(),  
                'mouth_rgb': mouth_color[0].tolist()  
            }
        },
        'categorize_info': {  
            'hair_skin_ratio': hair_reslh['lum']/skin_reslh['lum'],
            'skin_rgb': [skin_resrgb['red'], skin_resrgb['green'], skin_resrgb['blue']],
            'lum': skin_reslh['lum'],
            'hue': skin_reslh['hue']
        },
        'lh' : {
            'skin_lh': [skin_reslh['lum'], skin_reslh['hue']],
            'hair_lh': [hair_reslh['lum'], hair_reslh['hue']],
        }  
    }  

    hair_hex = rgb_to_hex(output['color']['rgb']['hair_rgb'])  
    mouth_hex = rgb_to_hex(output['color']['rgb']['mouth_rgb'])  
    skin_hex = rgb_to_hex(output['color']['rgb']['skin_rgb'])  
 
    output['color']['hex'] = {  
        'hair_hex': hair_hex,  
        'mouth_hex': mouth_hex,  
        'skin_hex': skin_hex  
    }

    ita = get_ita(output['color']['rgb']['skin_rgb'])
    output['categorize_info']['skin_ita'] = float(ita)
    grey_scale = rgb_to_gray(output['color']['rgb']['skin_rgb'])
    output['categorize_info']['skin_grey_scale'] = int(grey_scale)
    saturation_scale = rgb_to_saturation(output['color']['rgb']['skin_rgb'])
    output['categorize_info']['skin_saturation_scale'] = saturation_scale 

    
    if ita <= ImageAnalysis.ita_threshold:
        season = 'Autumn'
    elif output['categorize_info']['hair_skin_ratio'] <= ImageAnalysis.hair_skin_ratio_threshold_1:
        season = 'Winter'
    elif output['categorize_info']['hair_skin_ratio'] <= ImageAnalysis.hair_skin_ratio_threshold_2:
        season = 'Spring'
    else:
        season = 'Summer'
    output['categorize_info']['predicted_season'] = season
    output['season'] = season
    return output
