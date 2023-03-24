import gradio as gr

from modules import scripts, script_callbacks, shared, devices
from modules import processing, images
from modules import scripts, shared, devices
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.shared import opts, state

from extensions.ddetailer import ddetailer


def on_ui_settings():
    shared.opts.add_option("dd_save_previews", shared.OptionInfo(
        False, "Save mask previews", section=("ddetailer", "Detection Detailer")))
    shared.opts.add_option("outdir_ddetailer_previews", shared.OptionInfo(
        "extensions/ddetailer/outputs/masks-previews", 'Output directory for mask previews', section=("ddetailer", "Detection Detailer")))
    shared.opts.add_option("dd_save_masks", shared.OptionInfo(
        False, "Save masks", section=("ddetailer", "Detection Detailer")))
    shared.opts.add_option("outdir_ddetailer_masks", shared.OptionInfo(
        "extensions/ddetailer/outputs/masks", 'Output directory for masks', section=("ddetailer", "Detection Detailer")))


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


class DetectionDetailerScript(scripts.Script):
    def title(self):
        return "Detection Detailer"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        model_list = ddetailer.list_models()
        if is_img2img:
            info = gr.HTML(
                "<p style=\"margin-bottom:0.75em\">Recommended settings: Use from inpaint tab, inpaint at full res ON, denoise <0.5</p>")
        else:
            info = gr.HTML("")
        with gr.Group():
            with gr.Row():
                dd_model_a = gr.Dropdown(label="Primary detection model (A)",
                                         choices=model_list, value="None", visible=True, type="value")

            with gr.Row():
                dd_conf_a = gr.Slider(label='Detection confidence threshold % (A)',
                                      minimum=0, maximum=100, step=1, value=30, visible=False)
                dd_dilation_factor_a = gr.Slider(
                    label='Dilation factor (A)', minimum=0, maximum=255, step=1, value=4, visible=False)

            with gr.Row():
                dd_offset_x_a = gr.Slider(
                    label='X offset (A)', minimum=-200, maximum=200, step=1, value=0, visible=False)
                dd_offset_y_a = gr.Slider(
                    label='Y offset (A)', minimum=-200, maximum=200, step=1, value=0, visible=False)

            with gr.Row():
                dd_preprocess_b = gr.Checkbox(
                    label='Inpaint model B detections before model A runs', value=False, visible=False)
                dd_bitwise_op = gr.Radio(label='Bitwise operation', choices=[
                                         'None', 'A&B', 'A-B'], value="None", visible=False)

        br = gr.HTML("<br>")

        with gr.Group():
            with gr.Row():
                dd_model_b = gr.Dropdown(label="Secondary detection model (B) (optional)",
                                         choices=model_list, value="None", visible=False, type="value")

            with gr.Row():
                dd_conf_b = gr.Slider(label='Detection confidence threshold % (B)',
                                      minimum=0, maximum=100, step=1, value=30, visible=False)
                dd_dilation_factor_b = gr.Slider(
                    label='Dilation factor (B)', minimum=0, maximum=255, step=1, value=4, visible=False)

            with gr.Row():
                dd_offset_x_b = gr.Slider(
                    label='X offset (B)', minimum=-200, maximum=200, step=1, value=0, visible=False)
                dd_offset_y_b = gr.Slider(
                    label='Y offset (B)', minimum=-200, maximum=200, step=1, value=0, visible=False)

        with gr.Group():
            with gr.Row():
                dd_mask_blur = gr.Slider(
                    label='Mask blur ', minimum=0, maximum=64, step=1, value=4, visible=(not is_img2img))
                dd_denoising_strength = gr.Slider(
                    label='Denoising strength (Inpaint)', minimum=0.0, maximum=1.0, step=0.01, value=0.4, visible=(not is_img2img))

            with gr.Row():
                dd_inpaint_full_res = gr.Checkbox(
                    label='Inpaint at full resolution ', value=True, visible=(not is_img2img))
                dd_inpaint_full_res_padding = gr.Slider(
                    label='Inpaint at full resolution padding, pixels ', minimum=0, maximum=256, step=4, value=32, visible=(not is_img2img))

        dd_model_a.change(
            lambda modelname: {
                dd_model_b: gr_show(modelname != "None"),
                dd_conf_a: gr_show(modelname != "None"),
                dd_dilation_factor_a: gr_show(modelname != "None"),
                dd_offset_x_a: gr_show(modelname != "None"),
                dd_offset_y_a: gr_show(modelname != "None")

            },
            inputs=[dd_model_a],
            outputs=[dd_model_b, dd_conf_a, dd_dilation_factor_a,
                     dd_offset_x_a, dd_offset_y_a]
        )

        dd_model_b.change(
            lambda modelname: {
                dd_preprocess_b: gr_show(modelname != "None"),
                dd_bitwise_op: gr_show(modelname != "None"),
                dd_conf_b: gr_show(modelname != "None"),
                dd_dilation_factor_b: gr_show(modelname != "None"),
                dd_offset_x_b: gr_show(modelname != "None"),
                dd_offset_y_b: gr_show(modelname != "None")
            },
            inputs=[dd_model_b],
            outputs=[dd_preprocess_b, dd_bitwise_op, dd_conf_b,
                     dd_dilation_factor_b, dd_offset_x_b, dd_offset_y_b]
        )

        return [info,
                dd_model_a,
                dd_conf_a, dd_dilation_factor_a,
                dd_offset_x_a, dd_offset_y_a,
                dd_preprocess_b, dd_bitwise_op,
                br,
                dd_model_b,
                dd_conf_b, dd_dilation_factor_b,
                dd_offset_x_b, dd_offset_y_b,
                dd_mask_blur, dd_denoising_strength,
                dd_inpaint_full_res, dd_inpaint_full_res_padding
                ]

    def run(self, p, info,
            dd_model_a,
            dd_conf_a, dd_dilation_factor_a,
            dd_offset_x_a, dd_offset_y_a,
            dd_preprocess_b, dd_bitwise_op,
            br,
            dd_model_b,
            dd_conf_b, dd_dilation_factor_b,
            dd_offset_x_b, dd_offset_y_b,
            dd_mask_blur, dd_denoising_strength,
            dd_inpaint_full_res, dd_inpaint_full_res_padding):

        processing.fix_seed(p)
        initial_info = None
        seed = p.seed
        p.batch_size = 1
        ddetail_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        is_txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)
        if (not is_txt2img):
            orig_image = p.init_images[0]
        else:
            p_txt = p
            p = StableDiffusionProcessingImg2Img(
                init_images=None,
                resize_mode=0,
                denoising_strength=dd_denoising_strength,
                mask=None,
                mask_blur=dd_mask_blur,
                inpainting_fill=1,
                inpaint_full_res=dd_inpaint_full_res,
                inpaint_full_res_padding=dd_inpaint_full_res_padding,
                inpainting_mask_invert=0,
                sd_model=p_txt.sd_model,
                outpath_samples=p_txt.outpath_samples,
                outpath_grids=p_txt.outpath_grids,
                prompt=p_txt.prompt,
                negative_prompt=p_txt.negative_prompt,
                styles=p_txt.styles,
                seed=p_txt.seed,
                subseed=p_txt.subseed,
                subseed_strength=p_txt.subseed_strength,
                seed_resize_from_h=p_txt.seed_resize_from_h,
                seed_resize_from_w=p_txt.seed_resize_from_w,
                sampler_name=p_txt.sampler_name,
                n_iter=p_txt.n_iter,
                steps=p_txt.steps,
                cfg_scale=p_txt.cfg_scale,
                width=p_txt.width,
                height=p_txt.height,
                tiling=p_txt.tiling,
            )
            p.do_not_save_grid = True
            p.do_not_save_samples = True
        output_images = []
        state.job_count = ddetail_count
        for n in range(ddetail_count):
            devices.torch_gc()
            start_seed = seed + n
            if (is_txt2img):
                print(
                    f"Processing initial image for output generation {n + 1}.")
                p_txt.seed = start_seed
                processed = processing.process_images(p_txt)
                init_image = processed.images[0]
            else:
                init_image = orig_image

            output_images.append(init_image)
            masks_a = []
            masks_b_pre = []

            # Optional secondary pre-processing run
            if (dd_model_b != "None" and dd_preprocess_b):
                label_b_pre = "B"
                results_b_pre = ddetailer.inference(
                    init_image, dd_model_b, dd_conf_b/100.0, label_b_pre)
                masks_b_pre = ddetailer.create_segmasks(results_b_pre)
                masks_b_pre = ddetailer.dilate_masks(
                    masks_b_pre, dd_dilation_factor_b, 1)
                masks_b_pre = ddetailer.offset_masks(
                    masks_b_pre, dd_offset_x_b, dd_offset_y_b)
                if (len(masks_b_pre) > 0):
                    results_b_pre = ddetailer.update_result_masks(
                        results_b_pre, masks_b_pre)
                    segmask_preview_b = ddetailer.create_segmask_preview(
                        results_b_pre, init_image)
                    shared.state.current_image = segmask_preview_b
                    if (opts.dd_save_previews):
                        images.save_image(segmask_preview_b, opts.outdir_ddetailer_previews,
                                          "", start_seed, p.prompt, opts.samples_format, p=p)
                    gen_count = len(masks_b_pre)
                    state.job_count += gen_count
                    print(
                        f"Processing {gen_count} model {label_b_pre} detections for output generation {n + 1}.")
                    p.seed = start_seed
                    p.init_images = [init_image]

                    for i in range(gen_count):
                        p.image_mask = masks_b_pre[i]
                        if (opts.dd_save_masks):
                            images.save_image(
                                masks_b_pre[i], opts.outdir_ddetailer_masks, "", start_seed, p.prompt, opts.samples_format, p=p)
                        processed = processing.process_images(p)
                        p.seed = processed.seed + 1
                        p.init_images = processed.images

                    if (gen_count > 0):
                        output_images[n] = processed.images[0]
                        init_image = processed.images[0]

                else:
                    print(
                        f"No model B detections for output generation {n} with current settings.")

            # Primary run
            if (dd_model_a != "None"):
                label_a = "A"
                if (dd_model_b != "None" and dd_bitwise_op != "None"):
                    label_a = dd_bitwise_op
                results_a = ddetailer.inference(
                    init_image, dd_model_a, dd_conf_a/100.0, label_a)
                masks_a = ddetailer.create_segmasks(results_a)
                masks_a = ddetailer.dilate_masks(masks_a, dd_dilation_factor_a, 1)
                masks_a = ddetailer.offset_masks(masks_a, dd_offset_x_a, dd_offset_y_a)
                if (dd_model_b != "None" and dd_bitwise_op != "None"):
                    label_b = "B"
                    results_b = ddetailer.inference(
                        init_image, dd_model_b, dd_conf_b/100.0, label_b)
                    masks_b = ddetailer.create_segmasks(results_b)
                    masks_b = ddetailer.dilate_masks(masks_b, dd_dilation_factor_b, 1)
                    masks_b = ddetailer.offset_masks(
                        masks_b, dd_offset_x_b, dd_offset_y_b)
                    if (len(masks_b) > 0):
                        combined_mask_b = ddetailer.combine_masks(masks_b)
                        for i in reversed(range(len(masks_a))):
                            if (dd_bitwise_op == "A&B"):
                                masks_a[i] = ddetailer.bitwise_and_masks(
                                    masks_a[i], combined_mask_b)
                            elif (dd_bitwise_op == "A-B"):
                                masks_a[i] = ddetailer.subtract_masks(
                                    masks_a[i], combined_mask_b)
                            if (ddetailer.is_allblack(masks_a[i])):
                                del masks_a[i]
                                for result in results_a:
                                    del result[i]

                    else:
                        print("No model B detections to overlap with model A masks")
                        results_a = []
                        masks_a = []

                if (len(masks_a) > 0):
                    results_a = ddetailer.update_result_masks(results_a, masks_a)
                    segmask_preview_a = ddetailer.create_segmask_preview(
                        results_a, init_image)
                    shared.state.current_image = segmask_preview_a
                    if (opts.dd_save_previews):
                        images.save_image(segmask_preview_a, opts.outdir_ddetailer_previews,
                                          "", start_seed, p.prompt, opts.samples_format, p=p)
                    gen_count = len(masks_a)
                    state.job_count += gen_count
                    print(
                        f"Processing {gen_count} model {label_a} detections for output generation {n + 1}.")
                    p.seed = start_seed
                    p.init_images = [init_image]

                    for i in range(gen_count):
                        p.image_mask = masks_a[i]
                        if (opts.dd_save_masks):
                            images.save_image(
                                masks_a[i], opts.outdir_ddetailer_masks, "", start_seed, p.prompt, opts.samples_format, p=p)

                        processed = processing.process_images(p)
                        if initial_info is None:
                            initial_info = processed.info
                        p.seed = processed.seed + 1
                        p.init_images = processed.images

                    if (gen_count > 0):
                        output_images[n] = processed.images[0]
                        if (opts.samples_save):
                            images.save_image(
                                processed.images[0], p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)

                else:
                    print(
                        f"No model {label_a} detections for output generation {n} with current settings.")
            state.job = f"Generation {n + 1} out of {state.job_count}"
        if (initial_info is None):
            initial_info = "No detections found."

        return Processed(p, output_images, seed, initial_info)


script_callbacks.on_ui_settings(on_ui_settings)
