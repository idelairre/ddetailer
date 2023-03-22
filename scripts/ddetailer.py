import os
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp

from modules import processing, images
from modules import scripts, script_callbacks, shared, devices, modelloader
from modules.shared import opts, cmd_opts, state
from modules.sd_models import model_hash


def list_models():
    return ['None', 'Selfie Segmentation', 'Face Detection', 'Anime Face Detection']


def update_result_masks(results, masks):
    """
    Update the masks for each result in results with the corresponding mask in masks.

    Args:
    results: list of segmentation results
    masks: list of masks to update the results with

    Returns:
    list of updated segmentation results
    """
    updated_results = []
    for i, result in enumerate(results):
        result["mask"] = masks[i]
        updated_results.append(result)
    return updated_results


def generate_output_images(results_a, init_image, n, start_seed, output_images,
                           p, opts, initial_info):
    gen_count = len(results_a)
    state.job_count += gen_count
    label_a = "A"
    segmask_preview_a = create_segmask_preview(results_a, init_image)
    shared.state.current_image = segmask_preview_a

    if (opts.dd_save_previews):
        images.save_image(segmask_preview_a, opts.outdir_ddetailer_previews,
                          "", start_seed, p.prompt, opts.samples_format, p=p)

    print(
        f"Processing {gen_count} model {label_a} detections for output generation {n + 1}.")
    p.seed = start_seed
    p.init_images = [init_image]

    for i in range(gen_count):
        p.image_mask = results_a[i]["mask"]
        if (opts.dd_save_masks):
            images.save_image(results_a[i]["mask"], opts.outdir_ddetailer_masks,
                              "", start_seed, p.prompt, opts.samples_format, p=p)

        processed = processing.process_images(p)

        if initial_info is None:
            initial_info = processed.info
        p.seed = processed.seed + 1
        p.init_images = processed.images

    if (gen_count > 0):
        output_images[n] = processed.images[0]
        if (opts.samples_save):
            images.save_image(processed.images[0], p.outpath_samples, "",
                              start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)
    else:
        print(
            f"No model {label_a} detections for output generation {n} with current settings.")


def save_previews_and_masks(results_a, init_image, p, opts, start_seed):
    masks_a = create_segmasks(results_a)
    segmask_preview_a = create_segmask_preview(results_a, init_image)
    shared.state.current_image = segmask_preview_a
    if opts.dd_save_previews:
        images.save_image(segmask_preview_a, opts.outdir_ddetailer_previews, "", start_seed,
                          p.prompt, opts.samples_format, p=p)
    for i, mask in enumerate(masks_a):
        if opts.dd_save_masks:
            images.save_image(mask, opts.outdir_ddetailer_masks, "", start_seed, p.prompt+f"_{i+1}",
                              opts.samples_format, p=p)


def create_and_process_masks(results_a, dd_model_b, dd_bitwise_op,
                             dd_conf_b, dd_dilation_factor_b,
                             dd_offset_x_b, dd_offset_y_b):
    masks_a = create_segmasks(results_a)
    masks_a = dilate_masks(masks_a, dd_dilation_factor_a, 1)
    masks_a = offset_masks(masks_a, dd_offset_x_a, dd_offset_y_a)

    if (dd_model_b != "None" and dd_bitwise_op != "None"):
        label_b = "B"
        results_b = inference(init_image, dd_model_b, dd_conf_b/100.0, label_b)
        masks_b = create_segmasks(results_b)
        masks_b = dilate_masks(masks_b, dd_dilation_factor_b, 1)
        masks_b = offset_masks(masks_b, dd_offset_x_b, dd_offset_y_b)
        if (len(masks_b) > 0):
            combined_mask_b = combine_masks(masks_b)
            for i in reversed(range(len(masks_a))):
                if (dd_bitwise_op == "A&B"):
                    masks_a[i] = bitwise_and_masks(masks_a[i], combined_mask_b)
                elif (dd_bitwise_op == "A-B"):
                    masks_a[i] = subtract_masks(masks_a[i], combined_mask_b)
                if (is_allblack(masks_a[i])):
                    del masks_a[i]
                    for result in results_a:
                        del result[i]
        else:
            print("No model B detections to overlap with model A masks")
            results_a = []
            masks_a = []

    return masks_a, results_a


def process_primary_model(init_image, dd_model_a, dd_model_b, dd_bitwise_op,
                          dd_conf_a, dd_conf_b, dd_dilation_factor_a, dd_dilation_factor_b,
                          dd_offset_x_a, dd_offset_y_a, dd_offset_x_b, dd_offset_y_b,
                          n, start_seed, output_images, p, opts, initial_info):
    label_a = "A" if dd_model_b == "None" else dd_bitwise_op
    results_a = inference(init_image, dd_model_a, dd_conf_a/100.0, label_a)
    masks_a = create_and_process_masks(results_a, dd_model_b, dd_bitwise_op,
                                       dd_conf_b, dd_dilation_factor_b,
                                       dd_offset_x_b, dd_offset_y_b)

    if masks_a:
        results_a = update_result_masks(results_a, masks_a)
        save_previews_and_masks(results_a, init_image, p, opts, start_seed)
        generate_output_images(results_a, init_image, n, start_seed, output_images,
                               p, opts, initial_info)

    else:
        print(
            f"No model {label_a} detections for output generation {n} with current settings.")


def preprocess_secondary_model(init_image, dd_model_b, dd_conf_b, dd_dilation_factor_b, dd_offset_x_b, dd_offset_y_b):
    label_b_pre = "B"
    results_b_pre = inference(init_image, dd_model_b,
                              dd_conf_b/100.0, label_b_pre)
    masks_b_pre = create_segmasks(results_b_pre)
    masks_b_pre = dilate_masks(masks_b_pre, dd_dilation_factor_b, 1)
    masks_b_pre = offset_masks(masks_b_pre, dd_offset_x_b, dd_offset_y_b)

    if len(masks_b_pre) > 0:
        results_b_pre = update_result_masks(results_b_pre, masks_b_pre)
        return results_b_pre, masks_b_pre
    else:
        print(
            f"No model B detections for output generation {n} with current settings.")
        return None, None


def generate_secondary_model_output(init_image, results_b_pre, masks_b_pre, n, start_seed, output_images, p, opts, state):
    if results_b_pre:
        segmask_preview_b = create_segmask_preview(results_b_pre, init_image)
        shared.state.current_image = segmask_preview_b

        if opts.dd_save_previews:
            images.save_image(segmask_preview_b, opts.outdir_ddetailer_previews,
                              "", start_seed, p.prompt, opts.samples_format, p=p)

        gen_count = len(masks_b_pre)
        state.job_count += gen_count
        print(
            f"Processing {gen_count} model B detections for output generation {n + 1}.")
        p.seed = start_seed
        p.init_images = [init_image]

        for i in range(gen_count):
            p.image_mask = masks_b_pre[i]
            if opts.dd_save_masks:
                images.save_image(masks_b_pre[i], opts.outdir_ddetailer_masks,
                                  "", start_seed, p.prompt, opts.samples_format, p=p)
            processed = processing.process_images(p)
            p.seed = processed.seed + 1
            p.init_images = processed.images

        if gen_count > 0:
            output_images[n] = processed.images[0]
            init_image = processed.images[0]
    else:
        print(
            f"No model B detections for output generation {n} with current settings.")


def process_secondary_model(init_image, output_images, opts, p, state, start_seed, dd_model_b, dd_conf_b, dd_preprocess_b, dd_dilation_factor_b, dd_offset_x_b, dd_offset_y_b, n):
    if dd_model_b != "None" and dd_preprocess_b:
        results_b_pre, masks_b_pre = preprocess_secondary_model(
            init_image, dd_model_b, dd_conf_b, dd_dilation_factor_b, dd_offset_x_b, dd_offset_y_b)
        generate_secondary_model_output(
            init_image, results_b_pre, masks_b_pre, n, start_seed, output_images, p, opts, state)


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
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
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
    segms = results[2]
    segmasks = []
    for i in range(len(segms)):
        cv2_mask = segms[i].astype(np.uint8) * 255
        mask = Image.fromarray(cv2_mask)
        segmasks.append(mask)

    return segmasks


def inference(image, modelname, conf_thres, label):
    if modelname == 'selfie_segmentation':
        results = inference_selfie_segmentation(image, conf_thres)
    elif modelname == 'face_detection':
        results = inference_face_detection(image, conf_thres, label)
    elif modelname == 'anime_face_detection':
        results = inference_anime_face_detection(image, conf_thres, label)
    return results


def inference_selfie_segmentation(image, conf_thres):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1)
    results = selfie_segmentation.process(
        cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > conf_thres
    bbox_results = []
    segm_results = []
    if np.sum(mask) > 0:
        y, x, _ = np.where(mask)
        bbox_results.append([x.min(), y.min(), x.max(), y.max(), 1])
        segm_results.append(mask)
    return [['person'], np.array(bbox_results), np.array(segm_results)]


def inference_face_detection(image, conf_thres, label):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    bbox_results = []
    segm_results = []

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

    return [label, np.array(bbox_results), np.array(segm_results)]


def detect_anime_faces(image, conf_thres, label):
    animeface_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    animeface_results = animeface_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    bbox_results = []
    segm_results = []

    for (x, y, w, h) in animeface_results:
        bbox_results.append([x, y, x+w, y+h, 1])
        cv2_mask = np.zeros(gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x), int(y)),
                      (int(x+w), int(y+h)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segm_results.append(cv2_mask_bool)

    return [label, np.array(bbox_results), np.array(segm_results)]
