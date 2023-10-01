import os
import xml.etree.ElementTree as ET

# Define the YOLOv3 annotation folder and the path to save XML annotations.
yolo_annotation_folder = 'static/dataset'
xml_annotation_dir = 'static/xml_annotations'

# Create a mapping of class IDs to object names.
class_id_to_name = {
    '0': 'pothole',
    '1': 'alligator crack',
    '2': 'lateral crack',
    '3': 'longitudinal crack',
}

# Create the XML directory if it doesn't exist.
os.makedirs(xml_annotation_dir, exist_ok=True)

# List all YOLO annotation files in the folder.
yolo_annotation_files = [f for f in os.listdir(yolo_annotation_folder) if f.endswith('.txt')]

for yolo_annotation_file in yolo_annotation_files:
    # Read the YOLO annotations file.
    with open(os.path.join(yolo_annotation_folder, yolo_annotation_file), 'r') as file:
        lines = file.readlines()
    
    # Extract image dimensions (you need to know this beforehand).
    image_width = 1920  # Update with the actual image width
    image_height = 1080  # Update with the actual image height
    
    # Create the XML tree.
    root = ET.Element("annotation")
    
    folder = ET.SubElement(root, "folder")
    folder.text = "train"
    
    # Use the YOLO annotation file name as the XML filename.
    xml_filename = os.path.splitext(yolo_annotation_file)[0] + '.xml'
    filename = ET.SubElement(root, "filename")
    filename.text = xml_filename
    
    path = ET.SubElement(root, "path")
    image_filename = os.path.splitext(yolo_annotation_file)[0] + '.jpg'
    path.text = os.path.join("/home/nuna/road-damage-detector/static/dataset", image_filename)  # Update with your path
    
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    size = ET.SubElement(root, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(image_width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(image_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    
    # Loop through the YOLO annotations and convert them to XML.
    for line in lines:
        line = line.strip().split()
        
        # Extract relevant information.
        class_id = line[0]
        x_center = float(line[1])
        y_center = float(line[2])
        width = float(line[3])
        height = float(line[4])
        
        # Map class ID to object name.
        object_name = class_id_to_name.get(class_id, 'unknown')
        
        # Calculate bounding box coordinates.
        xmin = int((x_center - width/2) * image_width)
        ymin = int((y_center - height/2) * image_height)
        xmax = int((x_center + width/2) * image_width)
        ymax = int((y_center + height/2) * image_height)
        
        # Create an object element for each class.
        obj = ET.SubElement(root, "object")
        
        name = ET.SubElement(obj, "name")
        name.text = object_name
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin_elem = ET.SubElement(bndbox, "xmin")
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox, "ymin")
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox, "xmax")
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox, "ymax")
        ymax_elem.text = str(ymax)
    
    # Create the XML string.
    xml_str = ET.tostring(root, encoding='utf8').decode('utf8')
    
    # Write the XML annotation to a file with the same name as the TXT file.
    xml_file_name = os.path.join(xml_annotation_dir, xml_filename)
    with open(xml_file_name, "w") as xml_file:
        xml_file.write(xml_str)
