import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import imageio
from datetime import datetime
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

RESEARCH_PATH = 'C:/Users/Alireza/Anaconda3/tensorflow_models/models/research'
PATH_TO_LABELS = RESEARCH_PATH + '/object_detection/data/mscoco_label_map.pbtxt'


def load_model(model_dir):
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model


def run_inference_for_single_image(model, image):
    # image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_inp):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_inp)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_inp,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    return image_inp


def get_argument():
    parser = argparse.ArgumentParser(
        description='This Programme tries to detect objects in a picture or video'
                    'Developer : S.Alireza Moazeni')

    parser.add_argument('--content',
                        help='image or video',
                        default="video")
    parser.add_argument('--path',
                        help='The path of content',
                        default="./traffic.mp4")
    parser.add_argument('--model',
                        help='which model do you want to use',
                        default="ssd_mobilenet_v1_coco_2017_11_17")

    return parser.parse_args()


def process_single_image(image_path, model):
    image_np = show_inference(model, np.array(Image.open(image_path)))
    image_np = Image.fromarray(image_np)
    image_np.save("output.png")
    print("Output image is ready...")


def process_single_video(video_path, model):
    video_reader = imageio.get_reader(video_path)
    out_put_path = video_path[0:video_path.find('.', 1)] + '_annotated.mp4'
    video_writer = imageio.get_writer(out_put_path, fps=10)

    # loop through and process each frame
    t0 = datetime.now()
    n_frames = 0

    frames = []
    for cur_frame in video_reader:
        frames.append(cur_frame)
    print("Number of frames : " + str(len(frames)))

    # for frame in video_reader:
    for frame in frames:
        # rename for convenience
        image_np = frame
        n_frames += 1

        # Actual detection.
        image_np = show_inference(model, image_np)

        # instead of plotting image, we write the frame to video
        video_writer.append_data(image_np)

    fps = n_frames / (datetime.now() - t0).total_seconds()
    print("Frames processed: %s, Speed: %s fps" % (n_frames, fps))

    # clean up
    video_writer.close()


if __name__ == '__main__':
    args = get_argument()
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    PATH_TO_MODEL = RESEARCH_PATH + '/object_detection/' + args.model + '/saved_model'
    detection_model = load_model(model_dir=PATH_TO_MODEL)

    if args.content == 'image':
        process_single_image(args.path, detection_model)
    elif args.content == 'video':
        process_single_video(args.path, detection_model)
