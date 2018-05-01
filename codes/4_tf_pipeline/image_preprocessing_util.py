import os
from bs4 import BeautifulSoup

ANNOTATION_DIR = "Annotation"


def _get_annotation_info(bleed, filename):
    target_file = os.path.join(ANNOTATION_DIR, bleed,filename)
    f = open(target_file, "r")
    doc = BeautifulSoup(f.read(), 'lxml')
    f.close()
    return doc


def get_bounding_size_info(bleed, filename):
    doc = _get_annotation_info(bleed, filename.split(".")[0])
    size_info = get_orginal_size_info(bleed, filename)

    coordinates = []
    bndboxes = doc.find_all("bndbox")
    for box in bndboxes:
        temp = []
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        width = int(box.find("xmax").text) - int(box.find("xmin").text)
        height = int(box.find("ymax").text) - int(box.find("ymin").text)

        bounding_width = width if size_info["width"] > (xmin+width) else \
                    size_info["width"] - xmin
        bounding_height = height if size_info["height"] > (ymin+height) else \
                    size_info["height"] - ymin

        temp.append(xmin)
        temp.append(ymin)
        temp.append(bounding_width)
        temp.append(bounding_height)
    coordinates.append(temp)
    return coordinates


def get_orginal_size_info(bleed, filename):
    doc = _get_annotation_info(bleed, filename.split(".")[0])
    size_info = {}
    size_info["width"]  = (int(doc.find("width").text))
    size_info["height"] = (int(doc.find("height").text))
    return size_info
