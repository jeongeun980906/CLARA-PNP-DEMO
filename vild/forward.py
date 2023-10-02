import numpy as np
from PIL import Image
from vild.nms import nms
from vild.vis import *
from vild.embedding import FLAGS,build_text_embedding
import tensorflow.compat.v1 as tf
'''
This file includes code derived from [saycan] by Google LLC.
Copyright 2022 Google LLC.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Source Code: https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb

Modifications made on 2023-10-03:
Extracted the vision part of the original code.
'''
overall_fig_size = (18, 24)
class vild():
  def __init__(self, clip_model, category_name_string,params):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, )
    self.session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
    saved_model_dir = "./image_path_v2"
    _ = tf.saved_model.loader.load(self.session, ["serve"], saved_model_dir)
    numbered_categories = [{"name": str(idx), "id": idx,} for idx in range(50)] 
    self.numbered_category_indices = {cat["id"]: cat for cat in numbered_categories}
    self.clip_model = clip_model
    self.category_name_string = category_name_string
    self.params = params

  def infer(self,image_path,plot_on=True, prompt_swaps=[], name=None):
    # Preprocessing categories and get params
    for a, b in prompt_swaps:
      self.category_name_string = self.category_name_string.replace(a, b)
    category_names = [x.strip() for x in self.category_name_string.split(";")]
    category_names = ["background"] + category_names
    categories = [{"name": item, "id": idx+1,} for idx, item in enumerate(category_names)]
    category_indices = {cat["id"]-1: cat for cat in categories}
    # print(category_indices)
    max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area = self.params
    fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)


    #################################################################
    # Obtain results and read image
    roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = self.session.run(
          ["RoiBoxes:0", "RoiScores:0", "2ndStageBoxes:0", "2ndStageScoresUnused:0", "BoxOutputs:0", "MaskOutputs:0", "VisualFeatOutputs:0", "ImageInfo:0"],
          feed_dict={"Placeholder:0": [image_path,]})
      
    roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
    # no need to clip the boxes, already done
    roi_scores = np.squeeze(roi_scores, axis=0)

    detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
    scores_unused = np.squeeze(scores_unused, axis=0)
    box_outputs = np.squeeze(box_outputs, axis=0)
    detection_masks = np.squeeze(detection_masks, axis=0)
    visual_features = np.squeeze(visual_features, axis=0)

    image_info = np.squeeze(image_info, axis=0)  # obtain image info
    image_scale = np.tile(image_info[2:3, :], (1, 2))
    image_height = int(image_info[0, 0])
    image_width = int(image_info[0, 1])

    rescaled_detection_boxes = detection_boxes / image_scale # rescale

    # Read image
    image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
    assert image_height == image.shape[0]
    assert image_width == image.shape[1]


    #################################################################
    # Filter boxes

    # Apply non-maximum suppression to detected boxes with nms threshold.
    nmsed_indices = nms(
        detection_boxes,
        roi_scores,
        thresh=nms_threshold
        )

    # Compute RPN box size.
    box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

    # Filter out invalid rois (nmsed rois)
    valid_indices = np.where(
        np.logical_and(
          np.isin(np.arange(len(roi_scores), dtype=np.int), nmsed_indices),
          np.logical_and(
              np.logical_not(np.all(roi_boxes == 0., axis=-1)),
              np.logical_and(
                roi_scores >= min_rpn_score_thresh,
                np.logical_and(
                  box_sizes > min_box_area,
                  box_sizes < max_box_area
                  )
                )
          )    
        )
    )[0]

    detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
    detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
    detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
    detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
    rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]


    #################################################################
    # Compute text embeddings and detection scores, and rank results
    text_features = build_text_embedding(self.clip_model,categories)
    
    raw_scores = detection_visual_feat.dot(text_features.T)
    
    scores_all = raw_scores

    indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
    indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

    
    #################################################################
    # Print found_objects
    found_objects = []
    det_res = []
    for a, b in prompt_swaps:
      category_names = [name.replace(b, a) for name in category_names]  # Extra prompt engineering.
    for anno_idx in indices[0:int(rescaled_detection_boxes.shape[0])]:
      scores = scores_all[anno_idx]
      if np.argmax(scores) == 0:
        continue
      found_object = category_names[np.argmax(scores)]
      temp_box = rescaled_detection_boxes[anno_idx]
      if found_object == "background":
        continue
      print("Found a", found_object, "with score:", np.max(scores))
      found_objects.append(category_names[np.argmax(scores)])
      det_res.append(temp_box)
    if not plot_on:
      return found_objects, det_res
    

    #################################################################
    # Plot detected boxes on the input image.
    ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
    processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

    if len(indices_fg) == 0:
      display_image(np.array(image), size=overall_fig_size)
      print("ViLD does not detect anything belong to the given category")

    else:
      image_with_detections = visualize_boxes_and_labels_on_image_array(
          np.array(image),
          rescaled_detection_boxes[indices_fg],
          valid_indices[:max_boxes_to_draw][indices_fg],
          detection_roi_scores[indices_fg],    
          category_indices,
          instance_masks=segmentations[indices_fg],
          use_normalized_coordinates=False,
          max_boxes_to_draw=max_boxes_to_draw,
          min_score_thresh=min_rpn_score_thresh,
          skip_scores=False,
          skip_labels=True)

      plt.figure(figsize=overall_fig_size)
      plt.imshow(image_with_detections)
      plt.axis("off")
      plt.title("ViLD detected objects and RPN scores.")
      if name != None:
        plt.savefig("./res/init/{}.png".format(name))
      else:
        plt.show()

    return found_objects, det_res