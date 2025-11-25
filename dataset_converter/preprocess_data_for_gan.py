import torch
import argparse
import os
import sys
import json
import torchvision.transforms.v2 as transforms
from PIL import Image
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path if it's not already in there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from experiments.data import DataLoading
import experiments.custom_transforms as custom_transforms
from experiments.custom_datasets import SubsetWithTransform

# --- Multilabel Helper ---
class CompactMultilabelEncoder:
    def __init__(self, dataset, num_binary_classes):
        """
        dataset: torchvision-style dataset → yields (image, labels_tensor)
        num_binary_classes: number of binary label dimensions
        """
        self.num_binary_classes = num_binary_classes
        self.mapping = {}
        self.reverse_mapping = {}
        self._build_mapping(dataset)

    def _build_mapping(self, dataset):
        """Scan dataset once and assign compact IDs to each unique combination."""
        next_id = 0
        for _, labels in dataset:
            labels = labels.view(self.num_binary_classes).long()
            key = tuple(labels.tolist())

            if key not in self.mapping:
                self.mapping[key] = next_id
                self.reverse_mapping[next_id] = torch.tensor(key, dtype=torch.long)
                next_id += 1

        self.num_unique = next_id

    def multilabel_to_id(self, multilabel):
        """
        Convert labels → compact ID.
        """
        key = tuple(multilabel.view(self.num_binary_classes).long().tolist())
        return self.mapping[key]

    def id_to_multilabel(self, class_id):
        """
        Convert compact ID → multilabel tensor.
        """
        return self.reverse_mapping[class_id]

def load_user_dataset(dataset):
    print(f"Loading dataset: {dataset}...")
    
    data_class = DataLoading(dataset=dataset)
    data_class.create_transforms(train_aug_strat_orig='None', train_aug_strat_gen='None')
    
    #Adjust training image preprocessing. 
    #Normally uses lower-size training images (FixRes recipe) and random resized crop that are different to test transforms.
    #This could induce bias, hence we now use the test images preprocessing on train image as well.
    #Distance is measured with those transforms for later evaluation on test images.
    if dataset in ['Casting-Product-Quality', 'Describable-Textures', 'Flickr-Material']:
        data_class.transforms_preprocess_train = transforms.Compose([transforms.Resize(272, antialias=True), 
                                                                    transforms.CenterCrop(256)])
        imagesize = 256
        multilabel = False
    elif dataset in ['GTSRB']:
        data_class.transforms_preprocess_train = transforms.Compose([transforms.ToImage(), 
                            transforms.ToDtype(torch.float32, scale=True), 
                            transforms.Resize((32,32), antialias=True)])
        imagesize = 32
        multilabel = False
        class_to_idx = {
            'Speed limit (20km/h)': 0,
            'Speed limit (30km/h)': 1,
            'Speed limit (50km/h)': 2,
            'Speed limit (60km/h)': 3,
            'Speed limit (70km/h)': 4,
            'Speed limit (80km/h)': 5,
            'End of speed limit (80km/h)': 6,
            'Speed limit (100km/h)': 7,
            'Speed limit (120km/h)': 8,
            'No passing': 9,
            'No passing for vehicles over 3.5 metric tons': 10,
            'Right-of-way at the next intersection': 11,
            'Priority road': 12,
            'Yield': 13,
            'Stop': 14,
            'No vehicles': 15,
            'No vehicles over 3.5 metric tons': 16,
            'No entry': 17,
            'General caution': 18,
            'Dangerous curve to the left': 19,
            'Dangerous curve to the right': 20,
            'Double curve': 21,
            'Bumpy road': 22,
            'Slippery road': 23,
            'Road narrows on the right': 24,
            'Road work': 25,
            'Traffic signals': 26,
            'Pedestrians': 27,
            'Children crossing': 28,
            'Bicycles crossing': 29,
            'Beware of ice/snow': 30,
            'Wild animals crossing': 31,
            'End of all speed and passing limits': 32,
            'Turn right ahead': 33,
            'Turn left ahead': 34,
            'Ahead only': 35,
            'Go straight or right': 36,
            'Go straight or left': 37,
            'Keep right': 38,
            'Keep left': 39,
            'Roundabout mandatory': 40,
            'End of no passing': 41,
            'End of no passing by vehicles over 3.5 metric tons': 42
        }
    elif dataset in ['EuroSAT']:
        data_class.transforms_preprocess_train = transforms.Compose([transforms.ToImage(), 
                                                                     transforms.ToDtype(torch.float32, scale=True)])
        imagesize = 64
        multilabel = False
        class_to_idx = {
            'AnnualCrop': 0,
            'Forest': 1,
            'HerbaceousVegetation': 2,
            'Highway': 3,
            'Industrial': 4,
            'Pasture': 5,
            'PermanentCrop': 6,
            'Residential': 7,
            'River': 8,
            'SeaLake': 9
        }
    elif dataset in ['WaferMap']:
        #getting rid of any random padding
        data_class.transforms_preprocess_train = transforms.Compose([
                    transforms.ToImage(), 
                    transforms.ToDtype(torch.float32), #no scaling
                    custom_transforms.DivideBy3(),
                    custom_transforms.ExpandGrayscaleTensorTo3Channels(),
                    transforms.Pad(6) #our test time transform just pads 6px. 
                ])
        imagesize = 64
        multilabel = True

    data_class.load_base_data()
    trainset = SubsetWithTransform(data_class.base_trainset, data_class.transforms_preprocess_train)

    if dataset in ['Casting-Product-Quality', 'Describable-Textures', 'Flickr-Material']:
        if hasattr(trainset, 'class_to_idx'):
            # This works if user_trainset is the main dataset (e.g., ImageFolder)
            class_to_idx = trainset.class_to_idx
        elif hasattr(trainset, 'dataset') and hasattr(trainset.dataset, 'class_to_idx'):
            # This works if user_trainset is a torch.utils.data.Subset
            # The original dataset is stored in the .dataset attribute
            class_to_idx = trainset.dataset.class_to_idx
        elif hasattr(trainset, 'subset') and hasattr(trainset.subset, 'class_to_idx'):
            # This works if user_trainset is a torch.utils.data.Subset
            # The original dataset is stored in the .dataset attribute
            class_to_idx = trainset.subset.class_to_idx
        elif hasattr(trainset, 'subset') and hasattr(trainset.subset, 'dataset') and hasattr(trainset.subset.dataset, 'class_to_idx'):
            # This works if user_trainset is a torch.utils.data.Subset
            # The original dataset is stored in the .dataset attribute
            class_to_idx = trainset.subset.dataset.class_to_idx
        else:
            # If this fails, your object is neither a dataset with class_to_idx
            # nor a Subset of one.
            raise AttributeError("Could not find 'class_to_idx' attribute on "
                                "user_trainset or user_trainset.dataset")
    elif dataset in ['WaferMap']:
        class_to_idx = data_class.num_classes  #number of binary classes

    return trainset, imagesize, multilabel, class_to_idx

def main(args):
    print("--- Starting Preprocessing Step 1 ---")
    
    user_trainset, image_size, multilabel, class_to_idx = load_user_dataset(args.dataset_name)

    # --- 1. Handle Class Info for Human Mapping ---
    idx_to_class = {}
    if multilabel:
        num_binary_classes = class_to_idx
        print(f"Multilabel dataset detected with {num_binary_classes} binary classes.")
        id_encoder = CompactMultilabelEncoder(user_trainset, num_binary_classes)
    else:
        num_classes = len(class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"Single-label dataset detected with {num_classes} classes.")
        
    # --- 2. Setup Paths ---
    data_dir = os.path.join(args.gan_data_dir, f'{args.dataset_name}_for_gan')
    os.makedirs(data_dir, exist_ok=True)
    
    save_transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    
    # List to store entries for NVIDIA's dataset.json
    # Format: [ ["subfolder/image.png", class_int], ... ]
    dataset_json_labels = []

    print(f"Saving images and generating dataset.json in: {data_dir}")

    # --- 3. Iterate, Save Images, and Build Label List ---
    for i in tqdm(range(len(user_trainset)), desc="Preprocessing images"):
        image_tensor, label = user_trainset[i]
        
        # Get Class ID and Folder Name
        if multilabel:
            class_id = id_encoder.multilabel_to_id(label)
            class_name_str = '-'.join(str(int(x)) for x in label)
        else:
            class_id = int(label)
            class_name_str = str(class_id)
            
        # Create Class Subfolder
        class_dir = os.path.join(data_dir, class_name_str)
        os.makedirs(class_dir, exist_ok=True)
        
        # Process Image
        if image_tensor.min() < 0:
             image_tensor = (image_tensor + 1) / 2.0
        
        pil_image = save_transform(image_tensor)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        filename = f'img_{i:06d}.png'
        save_path = os.path.join(class_dir, filename)
        pil_image.save(save_path)
        
        # Add to NVIDIA Label List
        # Path must be relative to dataset root: "class_folder/filename.png"
        relative_path = f"{class_name_str}/{filename}"
        dataset_json_labels.append([relative_path, class_id])

    # --- 4. Save NVIDIA's dataset.json (REQUIRED FOR TRAINING) ---
    # This goes INSIDE the gan_data_dir
    dataset_json_path = os.path.join(data_dir, 'dataset.json')
    with open(dataset_json_path, 'w') as f:
        json.dump({"labels": dataset_json_labels}, f)
    print(f"Saved NVIDIA dataset.json to {dataset_json_path}")

    # --- 5. Save Human Readable Mapping (OPTIONAL BUT RECOMMENDED) ---
    # This goes OUTSIDE (or beside) the gan_data_dir for your later use
    if idx_to_class:
        mapping_json_path = os.path.join(data_dir, 'class_mapping.json')
        try:
            with open(mapping_json_path, 'w') as f:
                json.dump(idx_to_class, f, indent=4)
            print(f"Saved human-readable class mapping to {mapping_json_path}")
        except Exception as e:
            print(f"Warning: Could not save class mapping. {e}")

    print("--- Preprocessing Step 1 Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='WaferMap', help="Name of the dataset to preprocess")
    parser.add_argument('--gan_data_dir', type=str, default='../data', help="Folder for images")
    args = parser.parse_args()
    main(args)