import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models, transforms
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from io import BytesIO
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import gdown
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Prepare data path
new_demo_folder = './demo'
new_test_folder = './test-data/data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the VGG16 model
class VGG16Model(nn.Module):
    def __init__(self):
        super(VGG16Model, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad = False
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 1024),  # VGG16 features output size
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Output classes (NORMAL, PNEUMONIA)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Cache model loading
@st.cache_resource
def load_model():
    model = VGG16Model()  # Load the VGG16 model
    
    file_id = '1o5pWPFJwNFKKtf0DCjzIrC9_CFqae5D5'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    output_path = 'vgg16_pneumonia_model.pth' 
    gdown.download(url, output_path, quiet=False)
    model.load_state_dict(torch.load('vgg16_pneumonia_model.pth', map_location=torch.device('cpu')))

    model.eval() 
    return model

model = load_model() 

def preprocess_image(uploaded_image):
    pil_image = Image.open(uploaded_image).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize((255, 255)), 
        transforms.ToTensor(),        
    ])
    
    input_tensor = preprocess(pil_image)
    
    return input_tensor

def load_images_from_demo_folder(new_demo_folder):
    dataset = torchvision.datasets.ImageFolder(
        root=new_demo_folder,
        transform=transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.ToTensor(),
        ])
    )
    
    normal_images = []
    pneumonia_images = []
    image_paths = []
    
    for img, label in dataset:
        image_path = dataset.imgs[dataset.targets.index(label)][0]
        class_name = dataset.classes[label]       
        image_paths.append(image_path)
        
        if class_name == 'NORMAL':
            normal_images.append((img, class_name, image_path))
        elif class_name == 'PNEUMONIA':
            pneumonia_images.append((img, class_name, image_path))
    
    all_images = normal_images + pneumonia_images
    
    return all_images, dataset.class_to_idx, dataset.classes, image_paths



def display_title(model, original_class_label, idx_to_cls_names,predicted_class):
    st.markdown("---")
    st.markdown(f"Original Class: **:blue[{original_class_label}]**  | Predicted Class: **:green[{idx_to_cls_names[predicted_class]}]**")

# GradCAM
def get_model_outputs(model, img_tensor, layer_name):
    model.eval()
    return_nodes = {
                    layer_name: "last_conv_op",
                    "classifier.6": "out_classifier"
                   }
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    output_nodes_dict = feature_extractor(img_tensor)
    last_conv_layer_output = output_nodes_dict["last_conv_op"]
    preds_classifier = output_nodes_dict["out_classifier"]
    return last_conv_layer_output, preds_classifier

def make_gradcam_heatmap(trans_img, model, last_layer_name):
    last_conv_layer_output, preds = get_model_outputs(model, trans_img, last_layer_name)
    layer_grads = []
    hook = last_conv_layer_output.register_hook(lambda grad: layer_grads.append(grad.detach()))
    pred_index = torch.argmax(preds.squeeze(0))
    class_channel = preds[:, pred_index]
    class_channel.sum().backward()
    pooled_grads = torch.mean(layer_grads[0], (0, 2, 3))
    conv_op_reshape = last_conv_layer_output[0].permute(1, 2, 0).detach()
    heatmap = conv_op_reshape @ pooled_grads
    heatmap = heatmap.clamp(min=0.) / heatmap.max()
    hook.remove()
    return heatmap.cpu().numpy(), int(pred_index.detach())

def generate_gradcam(model, img_tensor, original_image, predicted_class, idx_to_cls_names, layer_name="features.28"):
    heatmap, pred_cls = make_gradcam_heatmap(img_tensor, model, layer_name)
    cam_colored = np.uint8(255 * heatmap)
    cam_colored = cv2.resize(cam_colored, (original_image.shape[1], original_image.shape[0]))
    cam_colored = cv2.applyColorMap(cam_colored, cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    blended_gradcam = cv2.addWeighted(cam_colored, 0.6, original_image, 0.4, 0) 
    return heatmap, blended_gradcam

def visualize_gradcam_map(model, image, idx_to_cls_names, image_names, img_tensor, original_image, predicted_class, original_class_label, layer_name="features.28"):
    heatmap, blended_gradcam = generate_gradcam(model, img_tensor, original_image, predicted_class, idx_to_cls_names, layer_name)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Image")
        st.image(original_image, caption=f"Original Image - Label: {idx_to_cls_names[predicted_class]}", width=170)

    with col2:
        st.subheader("Heatmap")
        dpi = 100
        width_in_inches = 2
        height_in_inches = width_in_inches
        fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        st.image(buf, caption=f"Grad-CAM Heatmap - Label: {idx_to_cls_names[predicted_class]}", width=170)
        plt.close(fig)

    with col3:
        st.subheader("Grad-CAM")
        st.image(blended_gradcam, caption=f"Blended Grad-CAM - Label: {idx_to_cls_names[predicted_class]}", width=170)

# LIME
def model_predict(images):
    images = torch.stack([transforms.ToTensor()(img) for img in images], dim=0)
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.detach().cpu().numpy()

def explain_with_lime(model, image, idx_to_cls_names):

    if isinstance(image, tuple):
        input_image, original_class_label, image_name = image
    else:
        input_image = image
        
    original_image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    original_image = np.clip(original_image, 0, 1)
    original_image = np.uint8(255 * original_image)

    explainer = lime_image.LimeImageExplainer()

    if st.session_state.lime_explanation is None:
        explainer = lime_image.LimeImageExplainer()

        with st.spinner('Generating LIME explanation...'):
            explanation = explainer.explain_instance(
                original_image, model_predict, 
                top_labels=2, hide_color=0,
                num_samples=30,
                distance_metric='cosine'
            )
        st.session_state.lime_explanation = explanation
    else:
        explanation = st.session_state.lime_explanation

    col1, col2, col3= st.columns(3)
    with col1:
        temp, mask = explanation.get_image_and_mask(
            label=1, positive_only=True, 
            num_features=25, hide_rest=True,
            min_weight=0.00000000001
        )
        tempp = np.interp(temp, (temp.min(), temp.max()), (0, 1))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(mark_boundaries(tempp, mask))
        ax.axis('off')
        
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        st.subheader("LIME")
        st.image(buf, caption="LIME Explanation (Hide Rest)", width=170)

    with col2:
        temp, mask = explanation.get_image_and_mask(
            label=1, positive_only=True, 
            num_features=25, hide_rest=False,
            min_weight=0.00000000001
        )
        tempp = np.interp(temp, (temp.min(), temp.max()), (0, 1))
        
        tempp_rgb = (tempp * 255).astype(np.uint8)
        tempp_rgb = cv2.cvtColor(tempp_rgb, cv2.COLOR_BGR2RGB)
        
        # Create a green mask
        green_mask = np.zeros_like(tempp_rgb)
        green_mask[mask == 1] = [0, 255, 0]  
        
        # Blend the original image with the green mask
        blended_image = cv2.addWeighted(tempp_rgb, 0.7, green_mask, 0.3, 0)
        
        st.subheader("LIME")
        st.image(blended_image, caption="LIME Explanation (Blended)", width=170)


# opt - Saliency map
def make_saliency_map(img_tensor, model):
    model.eval()
    img_tensor.requires_grad_()
    output = model(img_tensor)
    pred_index = torch.argmax(output, dim=1)
    model.zero_grad()
    output[0, pred_index].backward()
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = np.maximum(saliency, 0)
    saliency = saliency / saliency.max()
    return saliency

def generate_saliency_map(model, img_tensor, original_image, predicted_class, idx_to_cls_names):
    saliency_map = make_saliency_map(img_tensor, model)
    saliency_map_colored = np.uint8(255 * saliency_map)
    saliency_map_colored = cv2.applyColorMap(saliency_map_colored, cv2.COLORMAP_JET)
    saliency_map_colored_rgb = cv2.cvtColor(saliency_map_colored, cv2.COLOR_BGR2RGB)
    
    blended_saliency = cv2.addWeighted(saliency_map_colored_rgb, 0.6, original_image, 0.4, 0)
    
    if st.checkbox("Show Raw Saliency Map"):

        col1, col2,col3 = st.columns(3)
        with col1:
            st.subheader("Raw Map")
            st.image(saliency_map_colored_rgb, caption=f"Raw Saliency Map - Label: {idx_to_cls_names[predicted_class]}", width=170)
        
        with col2:
            st.subheader("Saliency Map")
            st.image(blended_saliency, caption=f"Saliency Map - Label: {idx_to_cls_names[predicted_class]}", width=170)


def main_visualization(model, image, idx_to_cls_names, image_names, layer_name="features.28"):
    model.eval()

    if isinstance(image, tuple):
        input_image, original_class_label, image_name = image
    else:
        input_image = image  

    original_image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    original_image = np.clip(original_image, 0, 1)
    original_image = np.uint8(255 * original_image)

    img_tensor = input_image.unsqueeze(0).to(device)
    img_tensor.requires_grad_()

    # Make the prediction
    outputs = model(input_image.unsqueeze(0).to(device))
    predicted_class = torch.argmax(outputs).item()


    # performance
    display_title(model,original_class_label, idx_to_cls_names,predicted_class)
    # gradcam
    visualize_gradcam_map(model, image, idx_to_cls_names, image_names, img_tensor, original_image, predicted_class, original_class_label, layer_name )
    #lime
    explain_with_lime(model, input_img, idx_to_cls_names)

    st.markdown("---")

    st.subheader("Explanation for GradCAM and LIME:")
    st.markdown("**GradCAM:**  Warmer colors such as red and yellow indicate areas with higher influence on the decision, green color regions are still relevant but contribute less significantly to the prediction, while cooler colors like blue represent regions with minimal impact. " )
    st.markdown("**LIME:** The image is break into superpixels and identify which segments positively contributed to the prediction. Positive superpixels which highlighted in green represent areas that support the model's classification such as regions showing clear signs of normal or pneumonia." )
    # saliency map
    # generate_saliency_map(model, img_tensor, original_image, predicted_class, idx_to_cls_names)



# Main App Structure
images, class_to_idx, classes, image_paths = load_images_from_demo_folder(new_demo_folder)
idx_to_cls_names = {idx: cls for cls, idx in class_to_idx.items()}
uploaded_image = None
input_img = None

if 'image_selected' not in st.session_state:
    st.session_state['image_selected'] = -1

def set_image_selected(i):
    st.session_state['image_selected'] = i
    global uploaded_image
    uploaded_image = None

def on_file_changed():
    st.session_state['image_selected'] = -1

if "lime_explanation" not in st.session_state:
    st.session_state.lime_explanation = None
    
if "confusion_matrix" not in st.session_state:
    st.session_state.confusion_matrix = None

if "classification_report" not in st.session_state:
    st.session_state.classification_report = None

st.title(f"Pneumonia X-Ray Classification - Grad-CAM & LIME Visualization")

col1, col2 = st.columns(2)
with col1:
    normal_Image_Filename = os.path.basename(image_paths[0])
    pneumonia_Image_Filename = os.path.basename(image_paths[1])
    st.caption("Select example image")
    st.button(f"Normal case: {normal_Image_Filename}", "select_normal_button", use_container_width=True,
        on_click=set_image_selected, args=[0])
    st.button(f"Pneumonia case: {pneumonia_Image_Filename}", "select_pneumonia_button", use_container_width=True,
        on_click=set_image_selected, args=[1])

    # Extract all convolutional layers
    layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    layer_name = st.selectbox("Select Convolutional Layer for Grad-CAM", layers, index=len(layers)-1)
    st.caption("** The selection only applicable to AI Engineers. Others please use default layer: feature.28")

with col2:
    uploaded_image = st.file_uploader("Upload an X-Ray Image", type=["jpg", "png", "jpeg"],on_change=on_file_changed)


if st.session_state['image_selected'] > -1:
    image_selected = st.session_state['image_selected']
    input_img = images[image_selected]
    # Reset LIME explanation when a new image is selected
    st.session_state.lime_explanation = None

if uploaded_image is not None:
    preprocessed_image = preprocess_image(uploaded_image)
    input_img = (preprocessed_image,"-","uploaded_image")
    st.session_state.lime_explanation = None

if input_img is not None:
    main_visualization(model, input_img, idx_to_cls_names, image_paths, layer_name)


# Main execution
if __name__ == "__main__":
    pass