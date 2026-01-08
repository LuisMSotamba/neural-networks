import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from .datasets import VideoDataset


def evaluate_cnn_model(model, loader, device, class_names, title="Test", lstm=False):
    """
    Gets predictions and plots confusion matrix.
    
    Params:
    - model: model to get outputs.
    - loader: pytorch dataloader to get samples. The sample must be ´input, label´ shape
    - device: cpu, gpu
    - class_names: classes to display in the confusion matrix.
        They must be arranged as how they were considered in training.
    - title: text to add to confusion matrix plot. 
    - lstm: flag to specify the model is an lstm instance
    """
    model.eval()
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    # handle state and memory if model is an LSTM network
    h = None
    c = None
    
    with torch.no_grad():
        tqdm_dataloader = tqdm(loader, desc=f"Evaluation", unit="batch")
        for inputs, labels in tqdm_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if lstm and h is not None:
                outputs, h, c = model(inputs, h,c)
            if lstm and h is None:
                outputs, h, c = model(inputs)
            if not lstm:
                outputs = model(inputs)
                
            _, preds = torch.max(outputs, 1)
            
            all_preds = torch.cat((all_preds, preds))
            all_labels = torch.cat((all_labels, labels))
    
    total_labels = all_labels.size(0)
    correct_pred = (all_preds == all_labels).sum().item()
            
    # Move to CPU for sklearn
    all_preds_cpu = all_preds.cpu().numpy()
    all_labels_cpu = all_labels.cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(all_labels_cpu, all_preds_cpu)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title(f'Confusion Matrix on {title} Set ({round(correct_pred/total_labels, 3)})')
    plt.show()
    
    return all_preds_cpu, all_labels_cpu

def tensor_normalization(image):
    """
    Description:
    Normalize tensor image ranging pixels between [0,1]
    
    image: 3D tensor (C,H,W)
    """
    return (image - image.min()) / (image.max() - image.min())

def compute_grad_cam_heatmaps(
    input_tensors,
    true_labels,
    model,
    target_layers
):
    """
    Description:
    Computes heatmap of input tensors

    Params:
    - input_tensors: input tensors the model will receive
    - true_labels: ground truth labels
    - model: model for running grad-cam
    - target_layers: grad-cam's target layers
    """
    # setup grad_cam
    cam = GradCAM(model=model, target_layers=target_layers)

    # compute heatmaps
    heatmaps = []
    for idx in range(input_tensors.shape[0]):
        # define input tensor
        input_tensor = input_tensors[idx].unsqueeze(0)
        input_tensor.requires_grad_(True)

        # define targets
        targets = [ClassifierOutputTarget(true_labels[idx])]

        # comput heatmaps using grad_cam method
        heatmap = cam(input_tensor=input_tensor, targets=targets)
        heatmaps.append(torch.from_numpy(heatmap))

    return torch.stack(heatmaps, dim=0)

def plot_grad_cam(
    images,
    heatmaps,
    true_labels=None,
    pred_labels=None,
    main_title="",
    use_fig_title=True
):
    """
    Description:
    Plot tensor images, heatmaps and a mixed image-heatmap to 
    see clearly where image's regions the network is looking at 
    for discrimate its class.
    
    images: 4D tensor with shape (N,C,H,W)
    heatmaps: 4D tensor with shape (N,C,H,W)
    true_labels: array of N true labels
    pred_labels: array of N predicted labels
    main_title: figure title. Used when use_fig_title is True
    use_fig_title: indicates if a fig title should be used.
    """
    # Set number of cols and rows
    rows = 3
    cols = images.shape[0]
    fig, axs = plt.subplots(rows, cols, figsize=(25,12))
    if use_fig_title:
        fig.suptitle(main_title, fontsize=16, fontweight='bold')

    for idx, (image, heatmap) in enumerate(zip(images, heatmaps)):
        # normalization
        image = tensor_normalization(image)
        heatmap = tensor_normalization(heatmap)
    
        # rearrange dimensions
        _image = image.permute(1,2,0)
        _image = _image.detach().cpu().numpy()
    
        _heatmap = heatmap.permute(1,2,0)
        _heatmap = _heatmap.detach().cpu().numpy()

        axs[0, idx].imshow(_image)
        axs[0, idx].axis('off')
        
        axs[1, idx].imshow(_heatmap, cmap="jet")
        axs[1, idx].axis('off')
    
        axs[2, idx].imshow(_image)
        axs[2, idx].imshow(_heatmap, cmap="jet", alpha=0.4)
        axs[2, idx].axis('off')

        # define titles
        if not use_fig_title:
            axs[0][idx].set_title(true_labels[idx])
            axs[1][idx].set_title(pred_labels[idx])
    
    plt.tight_layout()
    plt.show()

def _filter_images(
    true_label,
    predicted_label,
    trues,
    preds,
    label_encoder,
    dataset,
    n_images=10,
):
    """
    Description:
    Filter test images based on the true label's value. The true label's value 
    must be a string.
    """
    
    # encode labels
    true_label = label_encoder.transform([true_label])
    predicted_label = label_encoder.transform([predicted_label])

    # filter images by true and predicted label, then get the intersection between both
    trues_idx = np.argwhere(true_label==trues).flatten().tolist()
    pred_idx = np.argwhere(predicted_label==preds).flatten().tolist()

    set_trues = set(trues_idx)
    set_preds = set(pred_idx)

    true_preds_idx = np.array(list(set_trues & set_preds))
    print(f"Total samples: {len(true_preds_idx)}")

    if n_images > len(true_preds_idx):
        warnings.warn(f"Filtered images length is less than number of images required: {n_images} > {len(true_preds_idx)}")
    else:
        true_preds_idx = np.random.choice(true_preds_idx, size=n_images)

    return torch.stack([dataset[idx][0] for idx in true_preds_idx], dim=0)

def inspect_images(
    true_pred_labels,
    n_images,
    trues,
    preds,
    label_encoder,
    dataset,
    model,
    target_layers
):
    """
    Description:
    Iterate over true, pred labels tuple ploting grad_cam heatmaps

    Params:
    - true_pred_labels: list of (true,pred) pairs
    - n_images: number of images to plot
    - trues: list of ground truth values returned by the model in inference process
    - preds: list of prediction made by a cnn. This must be arranged same as trues list
    - label_encoder: encoder used for transforming classes labels to numbers
    - dataset: pytorch dataset used for true-preds calculation
    - model: model for applying grad-cam
    - target_layers: grad-cam's target layers.
    """
    for true_label, pred_label in true_pred_labels:
        
        filtered_images = _filter_images(
            true_label=true_label,
            predicted_label=pred_label,
            n_images=n_images,
            trues=trues,
            preds=preds,
            label_encoder=label_encoder,
            dataset=dataset
        )
        print(f"Filtered images: {filtered_images.shape}")

        n_images = filtered_images.shape[0]
        true_labels = [ true_label for _ in range(n_images) ]
        pred_labels = [ pred_label for _ in range(n_images) ]
        
        encoded_true_labels = label_encoder.transform(true_labels)        
        heatmaps = compute_grad_cam_heatmaps(
            input_tensors=filtered_images,
            true_labels=encoded_true_labels,
            model=model,
            target_layers=target_layers
        )
        print(f"Heatmaps' shape: {heatmaps.shape}")
        
        plot_grad_cam(filtered_images, heatmaps, main_title=f"True: {true_label} - Pred: {pred_label}")

def recreate_video_with_embedding_distance(
    df_video,
    similarity_measures,
    img_dir,
    image_transform,
    output_dir = "videos/",
    fps = 6
):
    """
    Recreate a video from a dataframe of frames displaying 2 views side by side:
    - Original Frame | Distances' line plot

    Params:
    - df_video: pandas dataframe of the video
    - similarity_measures: numpy array of frames' similarity distances ([[i, score, label, image_name]])
    - img_dir: directory where images are stored in
    - image_transform: transforms the image will be applied
    - output_dir: directory where videos will be saved into
    - fps: frames per second defined for the generated video
    """

    pass

def recreate_video(
    model,
    df_video,
    img_dir,
    target_layers,
    output_video_name,
    batchsize,
    num_workers,
    output_dir: str ="videos/",
    fps: int = 6,
):
    """
    Recreate a video from a dataframe of frames displaying 3 diferent views side by side:
    - Original Frame | Grad-CAM Heatmap | Overlayed Frame + Heatmap

    Params:
    - model: neural network model
    - df_video: pandas dataframe of videos
    - img_dir: base directory where images are stored
    - target_layers: list of grad-cam´s target layers
    - output_video_name: name of the recreated video
    - batchsize: 
    """
    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Output directory doesn't exist. Creating {output_dir}")
        os.makedirs(output_dir)

    # Define transforms to apply to each image
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Create a dataset object
    dataset = VideoDataset(
        dataframe=df_video,
        img_dir=img_dir,
        transform=img_transforms
    )
    
    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # create tqdm bar for tracking progress
    tqdm_dataloader = tqdm(dataloader, desc="Video generation", unit="batch")

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None

    model.eval()
    for batch in tqdm_dataloader:
        batch = batch.to(model.device)
        batch_cpu = batch.cpu()

        # compute predictions for each tensor
        with torch.no_grad():
            logits = model(batch)
            predictions = logits.argmax(dim=1) # prototype´s list in case the model is swav
        
        # compute heatmaps
        heatmaps = compute_grad_cam_heatmaps(
            input_tensors=batch,
            true_labels=predictions,
            target_layers=target_layers,
            model = model
        )
        
        # Process each frame in the batch
        for i in range(batch.size(0)):
            img = batch_cpu[i]
            hmap = heatmaps[i]
            
            # Normalize
            img_norm = tensor_normalization(img)
            hmap_norm = tensor_normalization(hmap)
            
            # To numpy
            img_np = img_norm.permute(1, 2, 0).numpy()  # HWC, RGB
            hmap_np = hmap_norm[0].numpy()  # Grayscale
            
            # Overlay
            overlay_np = show_cam_on_image(img_np, hmap_np, use_rgb=True, colormap=cv2.COLORMAP_JET)
            
            # Convert to BGR and uint8
            orig_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            hmap_bgr = cv2.cvtColor((np.stack([hmap_np] * 3, axis=-1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            over_bgr = cv2.cvtColor((overlay_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Concatenate horizontally
            combined_frame = cv2.hconcat([orig_bgr, hmap_bgr, over_bgr])
            
            # Initialize video writer on first frame
            if out_video is None:
                height, width = combined_frame.shape[:2]
                out_video = cv2.VideoWriter(os.path.join(output_dir, f'{output_video_name}.mp4'), fourcc, fps, (width, height))
            
            # Write frame
            out_video.write(combined_frame)
    
    if out_video is not None:
        out_video.release()
    
    print("Combined video saved successfully!")