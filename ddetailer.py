import os
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp


def list_models():
    return ['None', 'Selfie Segmentation', 'Face Detection', 'Anime Face Detection', 'Human Segmentation']


def to_model_name(model_name):
    if model_name == "None":
        return model_name

    return model_name.lower().replace(" ", "_")


def update_result_masks(results, masks):
    for i in range(len(masks)):
        boolmask = np.array(masks[i], dtype=bool)
        results[2][i] = boolmask
    return results


def create_segmask_preview(results, image):
    labels = results[0]
    bboxes = results[1]
    segms = results[2]

    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()

    for i in range(len(segms)):
        color = np.full_like(cv2_image, np.random.randint(
            100, 256, (1, 3), dtype=np.uint8))
        alpha = 0.2
        color_image = cv2.addWeighted(cv2_image, alpha, color, 1-alpha, 0)
        cv2_mask = segms[i].astype(np.uint8) * 255
        if cv2_mask.ndim == 2:
            cv2_mask = np.repeat(cv2_mask[:, :, np.newaxis], 3, axis=2)

        cv2_mask_bool = np.array(segms[i], dtype=bool)
        centroid = np.mean(np.argwhere(cv2_mask_bool), axis=0)
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])

        cv2_image = np.where(cv2_mask == 255, color_image, cv2_image)

        text_color = tuple([int(x) for x in (color[0][0] - 100)])
        name = labels[i]
        score = bboxes[i][4]
        score = str(score)[:4]
        text = name + ":" + score
        cv2.putText(cv2_image, text, (centroid_x - 30, centroid_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    if (len(segms) > 0):
        preview_image = Image.fromarray(
            cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    else:
        preview_image = image

    return preview_image


def is_allblack(mask):
    cv2_mask = np.array(mask)

    # If the mask has more than one channel, convert it to grayscale
    if cv2_mask.ndim > 2:
        cv2_mask = cv2.cvtColor(cv2_mask, cv2.COLOR_BGR2GRAY)

    return cv2.countNonZero(cv2_mask) == 0


def bitwise_and_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)
    return mask


def subtract_masks(cv2_mask1, cv2_mask2):
    # Convert PIL Images to NumPy arrays
    cv2_mask1 = np.array(cv2_mask1)
    cv2_mask2 = np.array(cv2_mask2)

    # Ensure the masks have the same shape
    if cv2_mask1.shape != cv2_mask2.shape:
        height, width = cv2_mask1.shape[:2]
        cv2_mask2 = cv2.resize(cv2_mask2, (width, height))

    # Ensure both masks have the same number of channels (convert to 3 channels if needed)
    if cv2_mask1.ndim != cv2_mask2.ndim:
        if cv2_mask1.ndim == 2:
            cv2_mask1 = np.stack((cv2_mask1,) * 3, axis=-1)
        elif cv2_mask2.ndim == 2:
            cv2_mask2 = np.stack((cv2_mask2,) * 3, axis=-1)

    cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)

    return mask


def dilate_masks(masks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return masks

    dilated_masks = []
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)

    for mask in masks:
        cv2_mask = np.array(mask)

        if cv2_mask.dtype != np.uint8:
            cv2_mask = cv2_mask.astype(np.uint8)

        if cv2_mask.size > 0:
            dilated_mask = cv2.dilate(cv2_mask, kernel, iterations=iter)
            dilated_masks.append(Image.fromarray(dilated_mask))

    return dilated_masks


def offset_masks(masks, offset_x, offset_y):
    if (offset_x == 0 and offset_y == 0):
        return masks
    offset_masks = []
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        offset_mask = cv2_mask.copy()
        offset_mask = np.roll(offset_mask, -offset_y, axis=0)
        offset_mask = np.roll(offset_mask, offset_x, axis=1)

        offset_masks.append(Image.fromarray(offset_mask))
    return offset_masks


def combine_masks(masks):
    initial_cv2_mask = np.array(masks[0])
    combined_cv2_mask = initial_cv2_mask
    for i in range(1, len(masks)):
        cv2_mask = np.array(masks[i])
        combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)

    combined_mask = Image.fromarray(combined_cv2_mask)
    return combined_mask


def create_segmasks(results):
    if results is None:
        return []

    labels = results[0]
    bboxes = results[1]
    segms = results[2]

    masks = []
    for i in range(len(segms)):
        mask = segms[i]
        masks.append(mask)

    return masks


def inference(image, modelname, conf_thres, label):
    if to_model_name(modelname) == 'selfie_segmentation':
        return inference_selfie_segmentation(image, conf_thres, label="Person")
    elif to_model_name(modelname) == 'face_detection':
        return inference_face_detection(image, conf_thres, label="Face")
    elif to_model_name(modelname) == 'anime_face_detection':
        return inference_anime_face_detection(image, conf_thres, label="Anime Face")
    elif to_model_name(modelname) == 'inference_human_segmentation':
        return inference_human_segmentation(image, conf_thres, label="Human")


def inference_selfie_segmentation(image, conf_thres, label="Person"):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1)
    results = selfie_segmentation.process(
        cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > conf_thres
    bbox_results = []
    segm_results = []
    labels = []
    if np.sum(mask) > 0:
        y, x, _ = np.where(mask)
        bbox_results.append([x.min(), y.min(), x.max(), y.max(), 1])
        segm_results.append(mask)
        labels.append(label)
    return [np.array(labels), np.array(bbox_results), np.array(segm_results)]


def inference_face_detection(image, conf_thres, label="Face"):
    mp_face_detection = mp.solutions.face_detection

    bbox_results = []
    segm_results = []
    labels = []

    with mp_face_detection.FaceDetection(min_detection_confidence=conf_thres, model_selection=1) as face_detection:
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        results_face_detection = face_detection.process(image_rgb)

        if results_face_detection.detections:
            for detection in results_face_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image_rgb.shape
                x0, y0, w, h = int(bboxC.xmin * iw), int(bboxC.ymin *
                                                         ih), int(bboxC.width * iw), int(bboxC.height * ih)
                x1, y1 = x0 + w, y0 + h

                cv2_mask = np.zeros(image_rgb.shape[:2], np.uint8)
                cv2.rectangle(cv2_mask, (int(x0), int(y0)),
                              (int(x1), int(y1)), 255, -1)
                cv2_mask_bool = cv2_mask.astype(bool)

                bbox_results.append([x0, y0, x1, y1, detection.score[0]])
                segm_results.append(cv2_mask_bool)
                labels.append(label)

    return [np.array(labels), np.array(bbox_results), np.array(segm_results)]


def inference_anime_face_detection(image, conf_thres, label='Anime Face'):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    cascade_file_path = os.path.join(
        os.path.dirname(__file__), 'lbpcascade_animeface.xml')
    animeface_cascade = cv2.CascadeClassifier(cascade_file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    animeface_results = animeface_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    bbox_results = []
    segm_results = []
    labels = []  # Create an empty list for labels

    for (x, y, w, h) in animeface_results:
        bbox_results.append([x, y, x+w, y+h, 1])
        cv2_mask = np.zeros(gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x), int(y)),
                      (int(x+w), int(y+h)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segm_results.append(cv2_mask_bool)
        labels.append(label)  # Append the label for each detected face

    return [labels, np.array(bbox_results), np.array(segm_results)]


def inference_human_segmentation(image, conf_thres=0.5, label='Human'):
    mp_pose = mp.solutions.pose
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    bbox_results = []
    segm_results = []
    labels = []

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=conf_thres) as pose:

        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            labels.append(label)
            segmentation_mask = np.stack(
                (results.segmentation_mask,) * 3, axis=-1)
            segm_results.append(segmentation_mask > 0.1)

            # Calculate bounding box
            landmarks = results.pose_landmarks.landmark
            landmarks_points = [(int(landmark.x * image_width),
                                 int(landmark.y * image_height)) for landmark in landmarks]
            xs, ys = zip(*landmarks_points)
            x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
            bbox_results.append([x_min, y_min, x_max, y_max, 1])

    if not labels:
        return [[], np.array([]), np.array([])]

    return [labels, np.array(bbox_results), np.array(segm_results)]
