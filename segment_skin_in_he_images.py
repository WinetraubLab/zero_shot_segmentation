# Segment skin from gel in H&E images utilizing facebook's SAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

class SegmentSkinInHEImages:
  def __init__(self, sam):
      self.predictor = SamPredictor(sam)

  # This function evaluates the neural network on input image
  # Inputs:
  #  he_image - H&E image loaded in RGB (N by M by 3)
  #  visualize_results - set to True to create a visualization of the results
  # Outputs:
  #  binary_mask - True - for every region that is skin, false otherwise
  def run_network(self, he_image, visualize_results=False):
    # Find points of interest
    points_array, points_label = self._compute_points_of_interest(he_image)
    
    # Do the SAM thing
    self.predictor.set_image(he_image)
    masks, scores, logits = self.predictor.predict(
      point_coords=points_array,
      point_labels=points_label,
      multimask_output=False,
      )
    mask = masks[0]

    if visualize_results:
      self._visualize_results(he_image, mask, points_array, points_label)

    return mask

  # This function predicts points that are to be used for SAM
  # Inputs:
  #  he_image - H&E image in RGB
  # Outputs:
  #  points_array - an array of points (in pixels) [[x,y],[x,y],...]
  #  points_label - an array of the same size as points_array where value can be 1 for including the point, 0 for excluding
  def _compute_points_of_interest(self, he_image):
    # Create a mask that outlines the borders of th image, exclude darked out areas
    im_gray = cv2.cvtColor(he_image, cv2.COLOR_RGB2GRAY)
    mask = np.array(im_gray > 0)

    # Find center of mass of the main blob
    moments = cv2.moments(mask.astype(np.uint8))
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # From center of mass, go down and right, find the x and y of the points that are still in the tissue (i.e. not in dark area)
    lowest_y = np.where(mask[:, cx])[0]
    lowest_y = lowest_y[len(lowest_y)-1]
    rightest_x = np.where(mask[cy, :])[0]
    rightest_x = rightest_x[len(rightest_x)-1]

    # Finish up by creating the points to be used
    points_array = np.array([[cx, cy], [cx, lowest_y-2], [rightest_x-2, cy], [cx, 1]])
    points_label = np.array([1,1,1,0])

    return(points_array, points_label)

  # This function visualizes results
  def _visualize_results(self, he_image, mask, points_array, points_label):
    def show_points(coords, labels, ax, marker_size=375):
      pos_points = coords[labels==1]
      neg_points = coords[labels==0]
      ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
      ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_mask(mask, ax, random_color=False):
      if random_color:
          color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
      else:
          color = np.array([30/255, 144/255, 255/255, 0.6])
      h, w = mask.shape[-2:]
      mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
      ax.imshow(mask_image)

    plt.figure(figsize=(5,5))
    plt.imshow(he_image)
    show_mask(mask, plt.gca())
    show_points(points_array, points_label, plt.gca())
    plt.axis('off')
    plt.show()

