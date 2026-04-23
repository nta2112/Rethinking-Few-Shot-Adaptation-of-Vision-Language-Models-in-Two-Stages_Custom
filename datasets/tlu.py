import os
import json
import random
from .utils import Datum, DatasetBase, read_json

template = ['a photo of a {}.']

class TLU(DatasetBase):
    dataset_dir = 'tlu-states'

    def __init__(self, root, num_shots, setting="standard", seed=1):
        # Extremely robust path resolution: find where split.json actually is
        current_check = root
        found_root = None
        for _ in range(4): # Check up to 4 levels up
            if os.path.exists(os.path.join(current_check, 'split.json')):
                found_root = current_check
                break
            current_check = os.path.dirname(current_check)
        
        if found_root:
            self.root = found_root
            print(f"Detected dataset root at: {self.root}")
            
            # Smart image path detection
            if os.path.isdir(os.path.join(self.root, 'tlu-states', 'images')):
                self.image_dir = os.path.join(self.root, 'tlu-states', 'images')
            elif os.path.isdir(os.path.join(self.root, 'images')):
                self.image_dir = os.path.join(self.root, 'images')
            else:
                # If split.json is inside 'images' folder already
                self.image_dir = self.root
        else:
            self.root = root
            self.image_dir = os.path.join(self.root, 'tlu-states', 'images')
            print(f"Warning: split.json not found. Using default image path: {self.image_dir}")

        self.split_path = os.path.join(self.root, 'split.json')
        print(f"Final Image Directory: {self.image_dir}")

        self.template = template

        # Read split.json
        split = read_json(self.split_path)
        train_classes = split['train']
        val_classes = split['val']
        test_classes = split['test']

        # Map all classes to unique IDs to maintain consistency
        all_classes = train_classes + val_classes + test_classes
        class_to_label = {cls_name: i for i, cls_name in enumerate(all_classes)}

        # Load data for each split
        train = self.read_data(train_classes, class_to_label)
        val = self.read_data(val_classes, class_to_label)
        test = self.read_data(test_classes, class_to_label)

        # Few-shot sampling for training
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        # Note: we can also sample val if it's too large, but usually val is kept as is
        # n_shots_val = min(num_shots, 4)
        # val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)

        if setting == "base2new":
            # In base2new, we train on base classes and evaluate on base + new
            # For TLU, we define: 
            # Base = train_classes
            # New = test_classes
            # Val = val_classes (as base-validation or mixed)
            
            # Re-index labels for the loaders (each set starts from 0 for its own classes)
            train_base = self.relabel(train)
            val_base = self.relabel(val) # Assuming val classes are also 'base' or just for validation
            test_base = self.relabel(val) # Usually base-test is same as val classes or subset
            test_new = self.relabel(test)
            
            # If we want to strictly follow the repo's logic where test_base and train classes are the same:
            # But TLU has 3 disjoint sets. 
            # Let's treat 'train' as base-train, 'val' as base-val, and 'test' as novel.
            train, val, test_base, test_new = train_base, val_base, val_base, test_new
        else:
            # Standard: train and test on the same classes?
            # Or train on 'train' and test on 'test'?
            # Usually 'standard' means train-classes == test-classes.
            # We'll just return everything as is.
            test_base = test
            test_new = None

        super().__init__(train_x=train, val=val, test=test_base, test_new=test_new)

    def read_data(self, class_names, class_to_label):
        items = []
        for cls_name in class_names:
            cls_dir = os.path.join(self.image_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Warning: Directory not found: {cls_dir}")
                continue
            
            imnames = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            label = class_to_label[cls_name]
            for imname in imnames:
                impath = os.path.join(cls_dir, imname)
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=cls_name
                )
                items.append(item)
        return items

    def relabel(self, data):
        """Re-indexes labels to 0..N-1 for the given data subset."""
        labels = sorted(list(set([item.label for item in data])))
        relabeler = {old_label: new_label for new_label, old_label in enumerate(labels)}
        
        new_data = []
        for item in data:
            new_item = Datum(
                impath=item.impath,
                label=relabeler[item.label],
                classname=item.classname
            )
            new_data.append(new_item)
        return new_data
