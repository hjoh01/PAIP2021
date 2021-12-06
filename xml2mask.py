# Pubilc
import cv2
import numpy as np
import os
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects
from tqdm import tqdm, trange
from xml.etree.ElementTree import Element, ElementTree, SubElement

def indent(elem, level=0): #자료 출처 https://goo.gl/J8VoDK
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def mask2xml(
    mask: np.ndarray,
    ) -> ElementTree:
    
    # Annotations
    annotations = Element("Annotations")

    ## Annotation
    annotation = SubElement(annotations, "Annotation")
    annotation.set("Id", "1")
    annotation.set("Name", "Prostate Prediction")

    ### Region
    regions = SubElement(annotation, "Regions")

    ### Reshape mask from level1 to level0
    # mask = np.kron(mask, np.ones((4,4)))

    #### Region
    mask = mask.astype(np.uint8)
    print("Computing connected components")
    labels, connected_component_map = cv2.connectedComponents(mask, connectivity=8, )

    print(connected_component_map.shape)
    cv2.imwrite(str(Path(xml_save_dir) / f"{filename}.tif"), connected_component_map.astype(np.uint8))

    print("Converting to XML")
    for n in trange(1, labels): # Exclude background label
        region = Element("Region", Id=str(n))

        n_mask = (connected_component_map == n).astype(np.uint8)
        contours, hierarchy = cv2.findContours(n_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_NONE

        # print(len(contours), len(contours[1]))

        for contour in tqdm(contours, leave=False):
            vertices = Element("Vertices")
            for pt in tqdm(contour, leave=False):
                x, y = pt[0][0], pt[0][1]
                vertex = Element("Vertex", X=str(4*x), Y=str(4*y), Z="0")
                vertices.append(vertex)

        region.append(vertices)
        regions.append(region)

    mask_xml = annotations

    return mask_xml

def remove_noise(mask, min_output_size=8192):
    # Remove small lines
    mask = mask > 0
    mask = remove_small_holes(mask, min_output_size, connectivity=8)
    mask = remove_small_objects(mask, min_output_size, connectivity=8)
    mask = (mask > 0).astype(np.uint8)
    return mask

if __name__ == "__main__":
    xml_save_dir = "./save_xml/"
    os.makedirs(xml_save_dir, exist_ok=True)

    filename = "prostate_0001"
    # sample_mask_filepath = f"/storage_2/prostate_data/predicted_mask/{filename}.png"
    
    print("Reading image")
    sample_mask = cv2.imread(sample_mask_filepath)
    sample_mask = cv2.cvtColor(sample_mask, cv2.COLOR_RGB2GRAY)
    sample_mask = (sample_mask > 0).astype(np.uint8)

    print("Removing noisy objects")
    sample_mask = remove_noise(sample_mask)

    sample_xml = mask2xml(sample_mask)

    print("Saving image")
    indent(sample_xml)
    ElementTree(sample_xml).write(Path(xml_save_dir) / f"{filename}.xml")

    print("Done")
