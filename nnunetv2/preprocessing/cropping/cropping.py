import numpy as np


# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    from scipy.ndimage import binary_fill_holes
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)

    slicer = bounding_box_to_slice(bbox)
    data = data[tuple([slice(None), *slicer])]

    if seg is not None:
        seg = seg[tuple([slice(None), *slicer])]

    nonzero_mask = nonzero_mask[slicer][None]
    if seg is not None:
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(np.int8)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


def crop_to_liver(data, seg, liver_label=1):
    """
    Crop the CT scan data, while preserving the liver region based on the segmentation mask.

    :param data: The input image data, shape (C, X, Y, Z) or (C, X, Y)
    :param seg: The segmentation mask, shape (C, X, Y, Z) or (C, X, Y)
    :param liver_label: The label corresponding to the liver in the segmentation mask
    :return: Cropped data, cropped segmentation, and bounding box
    """
    margin = 10
    
    #to-do: what if seg is None
    assert seg is not None, "Segmentation mask (seg) is required for liver-based cropping."
    assert data.shape[1:] == seg.shape[1:], "Data and segmentation mask must have matching spatial dimensions."
    
    if seg.ndim == 4:  # 4D case: (C, X, Y, Z)
        liver_mask = seg[0] == liver_label
    elif seg.ndim == 3:  # 3D case: (X, Y, Z)
        liver_mask = seg == liver_label
    else:
        raise ValueError("Segmentation mask must be 3D or 4D. Got shape: {}".format(seg.shape))
    
    # get the original bounding box of the liver
    bbox = get_bbox_from_mask(liver_mask)

    # expand the bounding box by the margin, ensuring it stays within image bounds
    expanded_bbox = []
    for dim, (low, high) in enumerate(bbox):
        expanded_low = max(low - margin, 0)
        expanded_high = min(high + margin, seg.shape[-3 + dim])  # use the spatial dimensions only
        expanded_bbox.append((expanded_low, expanded_high))

    slicer = bounding_box_to_slice(expanded_bbox)
        
    if data.ndim == 4:  # If data is (C, X, Y, Z)
        cropped_data = data[(slice(None),) + slicer]
    elif data.ndim == 3:  # If data is (C, X, Y)
        cropped_data = data[slicer]
    else:
        raise ValueError("Data must be 3D or 4D. Got shape: {}".format(data.shape))
    
    if data.ndim == 4:  # If data is (C, X, Y, Z)
        cropped_seg = seg[(slice(None),) + slicer]
    elif data.ndim == 3:  # If data is (C, X, Y)
        cropped_seg = seg[slicer]
    else:
        raise ValueError("Data must be 3D or 4D. Got shape: {}".format(data.shape))

    return cropped_data, cropped_seg, expanded_bbox