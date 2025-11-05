!nvidia-smi

---------------------------------------------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

---------------------------------------------------------------------------------------------------------------
!pip install -i https://test.pypi.org/simple/ supervision==0.3.0
!pip install -q transformers
!pip install -q pytorch-lightning
!pip install -q roboflow
!pip install -q timm

---------------------------------------------------------------------------------------------------------------
%cd {HOME}
!wget https://media.roboflow.com/notebooks/examples/dog.jpeg

---------------------------------------------------------------------------------------------------------------
import os
HOME = os.getcwd()
print(HOME)
IMAGE_NAME = "dog.jpeg"
IMAGE_PATH = os.path.join(HOME, IMAGE_NAME)


---------------------------------------------------------------------------------------------------------------
Loading model
---------------------------------------------------------------------------------------------------------------
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor


# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)


---------------------------------------------------------------------------------------------------------------
!pip install supervision


---------------------------------------------------------------------------------------------------------------
import cv2
import torch
import supervision as sv
import matplotlib.pyplot as plt

coco_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "hair brush"
]

with torch.no_grad():
    # Load image and predict
    image = cv2.imread(IMAGE_PATH)
    inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
    outputs = model(**inputs)

    # Post-process
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    results = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=CONFIDENCE_TRESHOLD,
        target_sizes=target_sizes
    )[0]

# Get the detections with NMS (Non-Maximum Suppression)
detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_TRESHOLD)

# Adjust class_ids by subtracting 1 (because array indexing starts at 0)
adjusted_class_ids = [class_id - 1 for class_id in detections.class_id]

# Get the labels based on the adjusted class_ids
labels = [coco_labels[class_id] for class_id in adjusted_class_ids]

# Print the labels
print("\nDetected Labels:")
print(labels)

# Print out the detections to see what we are passing to the BoxAnnotator
print("Bounding Box Coordinates (xyxy):")
print(detections.xyxy)  # Bounding boxes (x1, y1, x2, y2)

print("\nConfidence Scores:")
print(detections.confidence)  # Confidence scores for each detection

print("\nClass IDs (adjusted):")
print(adjusted_class_ids)  # Adjusted class IDs (after subtracting 1)

# Annotate the bounding boxes and labels on the image
box_annotator = sv.BoxAnnotator()

# Annotate the image directly using the detections and labels
frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

# Display the annotated frame using matplotlib
plt.figure(figsize=(16, 16))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes for a cleaner view
plt.show()
|
|
|
|
|
For another version...of supervision (new)
---------------------------------------------------------------------------------------------------------------
# pip install torch torchvision transformers==4.45.0 pillow opencv-python supervision==0.26.1 matplotlib

import cv2
import torch
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
from transformers import DetrForObjectDetection, DetrImageProcessor

# ---------- Config ----------
IMAGE_PATH = "dog.jpeg"  # replace with your image path
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "facebook/detr-resnet-50"
CONFIDENCE_THRESHOLD = 0.5

# ---------- Load model ----------
image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE).eval()

# ---------- COCO labels (DETR expects 91 ids; these match the processor's post-processing) ----------
coco_labels = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","street sign","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "hat","backpack","umbrella","shoe","eye glasses","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "mirror","dining table","window","desk","toilet","door","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "hair brush"
]

# ---------- Load image ----------
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read image at path: {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ---------- Inference ----------
with torch.no_grad():
    inputs = image_processor(images=image_rgb, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

target_sizes = torch.tensor([image_rgb.shape[:2]]).to(DEVICE)  # (h, w)
post = image_processor.post_process_object_detection(
    outputs=outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes
)[0]

# ---------- Convert to numpy ----------
boxes = post["boxes"].cpu().numpy()        # [N, 4] xyxy
scores = post["scores"].cpu().numpy()      # [N]
class_ids = post["labels"].cpu().numpy()   # [N]

# ---------- Supervision Detections ----------
detections = sv.Detections(
    xyxy=boxes.astype(np.float32),
    confidence=scores.astype(np.float32),
    class_id=class_ids.astype(np.int32),
).with_nms(threshold=0.5)  # ✅ Apply NMS here

# Adjust class IDs (if needed)
adjusted_class_ids = [cid - 1 for cid in detections.class_id]

# Build readable labels
labels = [
    f"{coco_labels[cid]} {score:.2f}"
    for cid, score in zip(adjusted_class_ids, detections.confidence)
]

# ---------- Annotate ----------
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6, text_padding=4)

annotated = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

# # Print the labels
print("\nDetected Labels:")
print(labels)

# Print out the detections to see what we are passing to the BoxAnnotator
print("Bounding Box Coordinates (xyxy):")
print(detections.xyxy)

print("\nConfidence Scores:")
print(detections.confidence)

print("\nClass IDs (adjusted):")
print(adjusted_class_ids)


# ---------- Show ----------
plt.figure(figsize=(12, 12))
plt.imshow(annotated)
plt.axis("off")
plt.tight_layout()
plt.show()







---------------------------------------------------------------------------------------------------------------
Checking done inference , now make dataset
import os
import torchvision
from transformers import AutoProcessor  # Assuming you're using a processor like AutoProcessor for image pre-processing

# settings
ANNOTATION_FILE_NAME = "_annotations.coco.json"

# Update dataset location (assuming the dataset is in Google Drive)
DATASET_LOCATION = "/content/drive/MyDrive/colab_files/wildcocodataset"
TRAIN_DIRECTORY = os.path.join(DATASET_LOCATION, "train")
VAL_DIRECTORY = os.path.join(DATASET_LOCATION, "valid")
TEST_DIRECTORY = os.path.join(DATASET_LOCATION, "test")

# You need an image processor (e.g., from HuggingFace Transformers)
# If you're using a pre-trained model like a DETR processor or any other, initialize it here.
image_processor = AutoProcessor.from_pretrained("facebook/detr-resnet-50")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        # Get the image and annotations from the parent class (CocoDetection)
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}

        # Process the image and annotations using the image processor
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")

        # Extract pixel values and target labels from the processed output
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

# Create instances of the dataset for train, validation, and test sets
TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY,
    image_processor=image_processor,
    train=True
)
VAL_DATASET = CocoDetection(
    image_directory_path=VAL_DIRECTORY,
    image_processor=image_processor,
    train=False
)
TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY,
    image_processor=image_processor,
    train=False
)

# Print out the dataset sizes
print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))







---------------------------------------------------------------------------------------------------------------
**Now after loading the data.......make some visualizations **
---------------------------------------------------------------------------------------------------------------
import random
import cv2
import numpy as np


# select random image
image_ids = TRAIN_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
# print('Image #{}'.format(image_id))

# load image and annotatons
image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])

print(image_path)
image = cv2.imread(image_path)
# print(image)
# print("its", annotations[:5])

# annotate
detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

# we will use id2label function for training
categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}

labels = [
    f"{id2label[class_id]}"
    for _, _, class_id, _
    in detections
]

box_annotator = sv.BoxAnnotator()
frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

%matplotlib inline
sv.show_frame_in_notebook(image, (8, 8))








---------------------------------------------------------------------------------------------------------------
from torch.utils.data import DataLoader

def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible 
    # to directly batch together images. Hence they pad the images to the biggest 
    # resolution in a given batch, and create a corresponding binary pixel_mask 
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)




---------------------------------------------------------------------------------------------------------------
Train model with PyTorch Lightning
---------------------------------------------------------------------------------------------------------------
!pip install pytorch-lightning -q






---------------------------------------------------------------------------------------------------------------
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch


class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER






---------------------------------------------------------------------------------------------------------------
%cd {HOME}

%load_ext tensorboard
%tensorboard --logdir lightning_logs/





---------------------------------------------------------------------------------------------------------------
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

batch = next(iter(TRAIN_DATALOADER))
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])



---------------------------------------------------------------------------------------------------------------
outputs.logits.shape



---------------------------------------------------------------------------------------------------------------
import pytorch_lightning as pl
print(pl.__version__)



---------------------------------------------------------------------------------------------------------------
batch = next(iter(TRAIN_DATALOADER))
print(type(batch))            # the batch type
print(batch.keys())           # keys in the dict
print(type(batch["labels"]))  # type of labels
print(batch["labels"][:2])    # see first two elements




---------------------------------------------------------------------------------------------------------------
from pytorch_lightning import Trainer

%cd {HOME}

# settings
MAX_EPOCHS = 10

# pytorch_lightning < 2.0.0
# trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

# pytorch_lightning >= 2.0.0
trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

trainer.fit(model)



