from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose


def main():
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)

    detector.reset_class(["person"], reuse_weights=['person'])

    x, img = data.transforms.presets.yolo.load_test("JZ20-R-TRU5400-012-1-03.jpg", short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs)

    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

    ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                                  class_IDs, bounding_boxs, scores,
                                  box_thresh=0.5, keypoint_thresh=0.2)
    plt.show()


if __name__ == '__main__':
    main()
