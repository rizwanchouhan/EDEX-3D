import imgaug
import imgaug.augmenters.meta as meta
import imgaug.augmenters as aug

# Function to create an augmenter from a given key-value pair
def augmenter_from_key_value(name, kwargs):
    if hasattr(meta, name):  # Check if the augmenter belongs to the meta module
        sub_augmenters = []
        kwargs_ = {}
        for item in kwargs:
            key = list(item.keys())[0]
            if hasattr(aug, key):  # Check if the sub-augmenter belongs to the aug module
                sub_augmenters += [augmenter_from_key_value(key, item[key])]
            else:
                kwargs_[key] = item[key]
        cl = getattr(imgaug.augmenters, name)  # Get the augmenter class
        args_ = []
        if len(sub_augmenters) > 0:
            args_ += [sub_augmenters]
        return cl(*args_, **kwargs_)

    if hasattr(imgaug.augmenters, name):  # Check if the augmenter belongs to the augmenters module
        cl = getattr(imgaug.augmenters, name)
        kwargs_ = {k: v for d in kwargs for k, v in d.items()}  # Flatten the kwargs dictionary
        for key in kwargs_.keys():
            if isinstance(kwargs_[key], list):
                kwargs_[key] = tuple(kwargs_[key])  # Convert lists to tuples
        return cl(**kwargs_)

    raise RuntimeError(f"Augmenter with name '{name}' is either not supported or it does not exist")


# Function to create an augmenter from a dictionary of augmentations
def augmenter_from_dict(augmentation):
    augmenter_list = []
    for aug in augmentation:
        if len(aug) > 1:
            raise RuntimeError("This should be just a single element")
        key = list(aug.keys())[0]
        augmenter_list += [augmenter_from_key_value(key, kwargs=aug[key])]
    return imgaug.augmenters.Sequential(augmenter_list)


# Function to create an image augmenter
def create_image_augmenter(im_size, augmentation=None) -> imgaug.augmenters.Augmenter:
    augmenter_list = []
    if augmentation is not None:
        augmenter_list += [augmenter_from_dict(augmentation)]  # Add augmentations if provided
    augmenter_list += [imgaug.augmenters.Resize(im_size)]  # Resize the image
    augmenter = imgaug.augmenters.Sequential(augmenter_list)
    return augmenter
