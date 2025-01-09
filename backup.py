# test 2 images
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

# Prepare data path
new_test_folder = './images'
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
    model.load_state_dict(torch.load('./vgg16_pneumonia_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()  # Load the model once

def preprocess_image(uploaded_image):
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
    ])
    img = Image.open(uploaded_image)
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img

def load_images_from_test_folder(new_test_folder):
    # Initialize ImageFolder with the test folder
    dataset = torchvision.datasets.ImageFolder(
        root=new_test_folder,
        transform=transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.ToTensor(),
        ])
    )
    
    # Initialize lists to store images and their respective labels
    normal_images = []
    pneumonia_images = []
    image_paths = []

    # Print class-to-index mapping for debugging purposes
    print(f"Class to index mapping: {dataset.class_to_idx}")
    
    # Iterate through dataset to collect image paths
    for img, label in dataset:
        # Access the image path from dataset.imgs (image path is at index 0 of the tuple)
        image_path = dataset.imgs[dataset.targets.index(label)][0]
        class_name = dataset.classes[label]       
        image_paths.append(image_path)
        
        # Add image to respective class list
        if class_name == 'NORMAL':
            normal_images.append((img, class_name, image_path))
        elif class_name == 'PNEUMONIA':
            pneumonia_images.append((img, class_name, image_path))
    
    # Combine images into one list
    all_images = normal_images + pneumonia_images
    
    return all_images, dataset.class_to_idx, dataset.classes, image_paths

# Function to extract features and predictions from model
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

# Function to generate Grad-CAM heatmap
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

# Function to generate saliency map
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

if 'image_selected' not in st.session_state:
    st.session_state['image_selected'] = -1

def set_image_selected(i):
    st.session_state['image_selected'] = i

def on_file_changed():
    st.session_state['image_selected'] = -1

# if st.session_state['image_selected'] > -1:
#     image_selected = st.session_state['image_selected']
# elif uploaded_image is not None:
#     # show the uploaded

# Define Streamlit app structure
def visualize_saliency_and_gradcam_map(model, images, idx_to_cls_names, image_names, uploaded_image=None, last_layer_name="features.30"):
    model.eval()

    # Use uploaded image if present, otherwise use default image from the test folder
    if uploaded_image:
        input_image, original_class_label = preprocess_image(uploaded_image)
        original_image = input_image.permute(1, 2, 0).cpu().numpy()
        original_image = np.clip(original_image, 0, 1)
        original_image = np.uint8(255 * original_image)
    else:
        # Get the image_id from Streamlit's session state or use default value
        image_id = st.session_state.get("image_id", 0)

        # Select the image and its label
        input_image, original_class_label, image_name = images[image_id]
        original_image = input_image.permute(1, 2, 0).cpu().numpy()
        original_image = np.clip(original_image, 0, 1)
        original_image = np.uint8(255 * original_image)

        img_tensor = input_image.unsqueeze(0).to(device)
        img_tensor.requires_grad_()

    img_tensor = input_image.unsqueeze(0).to(device)
    img_tensor.requires_grad_()

    # Create Saliency Map
    saliency_map = make_saliency_map(img_tensor, model)
    saliency_map_colored = np.uint8(255 * saliency_map)
    saliency_map_colored = cv2.applyColorMap(saliency_map_colored, cv2.COLORMAP_JET)

    # Create Grad-CAM Heatmap
    heatmap, pred_cls = make_gradcam_heatmap(img_tensor, model, last_layer_name)
    cam_colored = np.uint8(255 * heatmap)
    cam_colored = cv2.resize(cam_colored, (original_image.shape[1], original_image.shape[0]))
    cam_colored = cv2.applyColorMap(cam_colored, cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # Make the prediction
    outputs = model(input_image.unsqueeze(0).to(device))
    predicted_class = torch.argmax(outputs).item()

    st.title(f"Pneumonia X-Ray Classification - Saliency & Grad-CAM Visualization {predicted_class}")

    col1, col2 = st.columns(2)
    with col1:
        # Change the image_id using session state
        image_id = st.selectbox("Select Image ID", range(len(images)), format_func=lambda x: os.path.basename(image_names[x]), index=image_id)
        st.session_state.image_id = image_id  # Save the updated image_id in session state
        st.button("Select Normal", "select_normal_button", use_container_width=True,
            on_click=set_image_selected, args=[0])
        st.button("Select Pneumonia", "select_pneumonia_button", use_container_width=True,
            on_click=set_image_selected, args=[0])

    with col2:
        uploaded_image = st.file_uploader("Upload an X-Ray Image", type=["jpg", "png", "jpeg"],on_change=on_file_changed)

    # Display the selected image and its results
    st.markdown(f"Original Class: **:blue[{original_class_label}]**")
    st.markdown(f"Predicted Class: **:green[{idx_to_cls_names[predicted_class]}]**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Image")
        st.image(original_image, caption=f"Original Image - Label: {original_class_label}", width=170)

    with col2:
        st.subheader("Saliency Map")
        saliency_map_colored_rgb = cv2.cvtColor(saliency_map_colored, cv2.COLOR_BGR2RGB)
        blended_saliency = cv2.addWeighted(saliency_map_colored_rgb, 0.6, original_image, 0.4, 0)
        st.image(blended_saliency, caption=f"Saliency Map - Label: {original_class_label}", width=170)

    with col3:
        st.subheader("Grad-CAM")
        blended_gradcam = cv2.addWeighted(cam_colored, 0.6, original_image, 0.4, 0)
        st.image(blended_gradcam, caption=f"GradCAM - Label: {original_class_label}", width=170)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Saliency Map")
        st.image(saliency_map_colored_rgb, caption=f"Saliency Map: {original_class_label}", width=250)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        dpi = 100
        width_in_inches = 2
        height_in_inches = width_in_inches
        fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        st.image(buf, caption=f"Grad-CAM Heatmap: {original_class_label}", width=250)
        plt.close(fig)


# Main execution
if __name__ == "__main__":
    # Load all images from the test folder (both NORMAL and PNEUMONIA)
    images, class_to_idx, classes, image_paths = load_images_from_test_folder(new_test_folder)
    idx_to_cls_names = {idx: cls for cls, idx in class_to_idx.items()}

    # Initialize session state for image_id if it doesn't exist
    if "image_id" not in st.session_state:
        st.session_state.image_id = 0  # Default to the first image

    # Display the visualization for the selected image or uploaded image
    uploaded_image = st.file_uploader("Upload an X-Ray Image", type=["jpg", "png", "jpeg"], key="file_uploader_2")
    visualize_saliency_and_gradcam_map(model, images, idx_to_cls_names, image_paths, uploaded_image=uploaded_image)