import os
import cv2
import glob
import json
import datetime
import requests
import gradio as gr
from tool_for_end2end import *

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
DATADIR = './temp'
_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">Tencet HunyuanvideoCustom Demo</h1>
</div>

''' 
# flask url
URL = "http://127.0.0.1:8080/predict2"

def post_and_get(width, height, num_steps, num_frames, guidance, flow_shift, seed, prompt, id_image, neg_prompt, template_prompt, template_neg_prompt, name):
    now = datetime.datetime.now().isoformat()
    imgdir = os.path.join(DATADIR, 'reference')
    videodir = os.path.join(DATADIR, 'video')
    imgfile = os.path.join(imgdir, now + '.png')
    output_video_path = os.path.join(videodir, now + '.mp4')

    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(videodir, exist_ok=True)
    cv2.imwrite(imgfile, id_image[:,:,::-1])
    
    proxies = {
        "http": None,
        "https": None,
    }
    
    files = {
        "trace_id": "abcd", 
        "image_path": imgfile, 
        "prompt": prompt, 
        "negative_prompt": neg_prompt,
        "template_prompt": template_prompt, 
        "template_neg_prompt": template_neg_prompt,
        "height":  height,
        "width": width,
        "frames": num_frames,
        "cfg": guidance,
        "steps": num_steps,
        "seed": int(seed),
        "name": name,
        "shift": flow_shift,
        "save_fps": 25, 
    }
    r = requests.get(URL, data = json.dumps(files), proxies=proxies)
    ret_dict = json.loads(r.text)
    video_buffer = ret_dict['content'][0]['buffer']
    save_video_base64_to_local(video_path=None, base64_buffer=video_buffer, 
        output_video_path=output_video_path)
    print('='*50)
    return output_video_path

def create_demo():
    
    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)
        with gr.Tab('单主体一致性'):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        prompt = gr.Textbox(label="Prompt", value="a man is riding a bicycle on the street.")
                        neg_prompt = gr.Textbox(label="Negative Prompt", value="")
                    id_image = gr.Image(label="Input reference image", height=480)

                with gr.Column(scale=2):
                    with gr.Group():
                        output_image = gr.Video(label="Generated Video")
                        
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("Options for generate video", open=False):
                        with gr.Row():
                            width = gr.Slider(256, 1536, 1280, step=16, label="Width")
                            height = gr.Slider(256, 1536, 720, step=16, label="Height")
                        with gr.Row():
                            num_steps = gr.Slider(1, 100, 30, step=5, label="Number of steps")
                            flow_shift = gr.Slider(1.0, 15.0, 13, step=1, label="Flow Shift")
                        with gr.Row():
                            num_frames = gr.Slider(1, 129, 129, step=4, label="Number of frames")
                            guidance = gr.Slider(1.0, 10.0, 7.5, step=0.5, label="Guidance")
                            seed = gr.Textbox(1024, label="Seed (-1 for random)")
                        with gr.Row():
                            template_prompt = gr.Textbox(label="Template Prompt", value="Realistic, High-quality. ")
                            template_neg_prompt = gr.Textbox(label="Template Negative Prompt", value="Aerial view, aerial view, " \
                            "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, " \
                            "distortion, blurring, text, subtitles, static, picture, black border. ")
                            name = gr.Textbox(label="Object Name", value="object ")
                with gr.Column(scale=1):
                    generate_btn = gr.Button("Generate")

            generate_btn.click(fn=post_and_get,
                inputs=[width, height, num_steps, num_frames, guidance, flow_shift, seed, prompt, id_image, neg_prompt, template_prompt, template_neg_prompt, name],
                outputs=[output_image],
            )
            
            quick_prompts = [[x] for x in glob.glob('./assets/images/*.png')]
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Other object', samples_per_page=1000, components=[id_image])
            example_quick_prompts.click(lambda x: x[0], inputs=example_quick_prompts, outputs=id_image, show_progress=False, queue=False)
            with gr.Row(), gr.Column():
                gr.Markdown("## Examples")
                example_inps = [
                    [
                        'A woman is drinking coffee at a café.',
                        './assets/images/seg_woman_01.png',
                        1280, 720, 30, 129, 7.5, 13, 1024,
                        "assets/videos/seg_woman_01.mp4"
                    ],
                    [
                        'In a cubicle of an office building, a woman focuses intently on the computer screen, typing rapidly on the keyboard, surrounded by piles of documents.',
                        './assets/images/seg_woman_03.png',
                        1280, 720, 30, 129, 7.5, 13, 1025,
                        "./assets/videos/seg_woman_03.mp4"
                    ],
                    [
                        'A man walks across an ancient stone bridge holding an umbrella, raindrops tapping against it.',
                        './assets/images/seg_man_01.png',
                        1280, 720, 30, 129, 7.5, 13, 1025,
                        "./assets/videos/seg_man_01.mp4"
                    ],
                    [
                        'During a train journey, a man admires the changing scenery through the window.',
                        './assets/images/seg_man_02.png',
                        1280, 720, 30, 129, 7.5, 13, 1026,
                        "./assets/videos/seg_man_02.mp4"
                    ]
                ]
                gr.Examples(examples=example_inps, inputs=[prompt, id_image, width, height, num_steps, num_frames, guidance, flow_shift, seed, output_image],)
    return demo

if __name__ == "__main__":
    allowed_paths = ['/']
    demo = create_demo()
    demo.launch(server_name='0.0.0.0', server_port=80, share=True, allowed_paths=allowed_paths)
