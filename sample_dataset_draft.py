from PIL import Image
import torch
import numpy as np
import torchvision.datasets as dset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import UniPCMultistepScheduler
from tqdm.auto import tqdm
import time
import os
import argconfig
import nltk

# class myCoCoCaptions(dset.CocoCaptions):
#     def __init__(self, root, annFile, transform=None, target_transform=None):
#         super().__init__(root, annFile, transform, target_transform)

#     def __getitem__(self, index: int):
#         img, target = super().__getitem__(index)
#         # do whatever you want
#         return img, target[np.random.randint(0, len(target))]
    
def main(args):
    #path decision
    print(args.randomseed)
    path = "draft_guidance_scale_" + str(args.guidance_scale) + "_alpha_" + str(args.alpha) + "_" + str(time.time())[0:10]
    os.mkdir(path)

    # load model
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", use_safetensors=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", use_safetensors=True
    )

    scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    cap = dset.CocoCaptions(root = '/home/data/coco2017/val2017',
                        annFile = '/home/data/coco2017/captions_val2017.json')

    # sample device
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # sample parameters
    height = args.height  # default height of Stable Diffusion
    width = args.width  # default width of Stable Diffusion
    num_inference_steps = args.num_inference_steps  # Number of denoising steps
    guidance_scale = args.guidance_scale  # Scale for classifier-free guidance
    alpha = args.alpha
    generator = torch.manual_seed(args.randomseed)  # Seed generator to create the inital latent noise
    np.random.seed(args.randomseed)
    batch_size = args.batch_size



    # sample
    for id in tqdm(range(1250)):

        # load from dataset
        begin = id * batch_size if id < 625 else (id - 625) * batch_size
        prompt = []
        for tmp in range(batch_size):
            _ , target = cap[begin+tmp] # load 4th sample
            prompt.append(target[np.random.randint(0, len(target))])

        half_prompt = [" ".join(text[0] if text[1] == 'NN' or text[1] == 'NNS' or text[1] == 'NNPS' or text[1] == 'NNP' else "" for text in nltk.pos_tag(it.replace(".","").split())) for it in prompt]

        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))
            text_embeddings = text_embeddings[0]

        half_text_input = tokenizer(
            half_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            half_text_embeddings = text_encoder(half_text_input.input_ids.to(torch_device))
            half_text_embeddings = half_text_embeddings[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, half_text_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        latents = latents * scheduler.init_noise_sigma



        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 3)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond,half_noise_pred_text, noise_pred_text = noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) + alpha * (noise_pred_text + noise_pred_uncond - 2 * half_noise_pred_text)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1).squeeze().permute(0, 2, 3, 1)
        image = (image * 255).round().to(torch.uint8).cpu().numpy()
        for it in range(batch_size):
            Image.fromarray(image[it]).save(path + "/" + str(batch_size * id + it) + ".png")
            # text_file = open(path + "/" + str(batch_size * id + it) + ".txt", "w")
            # text_file.write(prompt[it])
            # text_file.close()
    return

if __name__ == "__main__":
    parser = argconfig.ArgumentParser(description='argconfig example',config='./config/sample_coco2017.yaml')
    parser.add_argument('--height', type=int,default=512,
                        help='no help')
    parser.add_argument('--width',type=int, default=512,
                        help='no help')
    parser.add_argument('--num_inference_steps',type=int, default=50,
                        help='no help')
    parser.add_argument('--guidance_scale',type=float, default=3,
                        help='no help')
    parser.add_argument('--randomseed', type=int, default=42,
                        help='no help')
    parser.add_argument('--batch_size',type=int,  default=8,
                        help='no help')
    parser.add_argument('--alpha',type=float, default=1,
                        help='no help')
    args = parser.parse_args()

    main(args)