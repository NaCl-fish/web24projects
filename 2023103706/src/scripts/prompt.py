from diffusers import StableDiffusionPipeline
import torch
from pytorch_lightning import seed_everything

seed_everything(42)

model_id = ".cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/ded79e214aa69e42c24d3f5ac14b76d568679cc2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
device = 'cuda:0'
pipe = pipe.to(device)

num_images_per_prompt = 5


'''
prompts = ["a photo of Black footed Albatross, a bird of this species with hooked seabird bill, masked head, buff throat, brown eye, longer than head bill, white forehead, buff nape, large size, long-legged-like shape, buff primary, buff bill, buff crown.", 
        "a photo of Black footed Albatross, a bird of this type with hooked seabird bill, masked head, buff throat, brown eye, longer than head bill, white forehead, buff nape, large size, long-legged-like shape, buff primary, buff bill, buff crown.", 
        "a photo of Black footed Albatross, which is a type of bird has hooked seabird bill, masked head, buff throat, brown eye, longer than head bill, white forehead, buff nape, large size, long-legged-like shape, buff primary, buff bill, buff crown."]
'''

'''
prompts = ["a photo of Teddy Bear.", 
        "a photo of Teddy Bear.", 
        "a photo of Teddy Bear."]
'''

'''
prompts = ["a photo of Black footed Albatross, a bird of this species with hooked seabird bill, masked head, buff throat, brown eye, longer than head bill, white forehead, buff nape, large size, long-legged-like shape, buff primary, buff bill, buff crown.", 
        "a photo of Teddy Bear.", 
        "a photo of Black footed Albatross, which is a type of bird has hooked seabird bill, masked head, buff throat, brown eye, longer than head bill, white forehead, buff nape, large size, long-legged-like shape, buff primary, buff bill, buff crown."]
'''
# prompt_path = 'cub_templete'
prompt_path = 'flo_llama2'


import os
import glob

max_length = pipe.tokenizer.model_max_length

file_list = glob.glob(os.path.join(prompt_path, '*.txt'))

folders = os.listdir(os.path.join(prompt_path, 'result'))

# print(folders)
# print(str(2) in folders)

# import sys
# sys.exit()

for filename in file_list:
    index = int(filename.split('/')[-1].split('.')[0])
    # if index!=2051: continue
    # if index==1: break
    # if str(index) in folders: continue
    print(index)
    save_path = os.path.join(prompt_path, os.path.join('result-try2', str(index)))
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(prompt_path, filename)
    with open(filename, 'r') as f:
        prompt = f.readline()
        prompt = prompt[:min(800,len(prompt))]
        print(prompt)
        input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        
        negative_ids = pipe.tokenizer("", truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
        negative_ids = negative_ids.to(device)

        concat_embeds = []
        neg_embeds = []

        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)


        # image = pipe(prompt = prompt, num_images_per_prompt = num_images_per_prompt).images

        image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_images_per_prompt = num_images_per_prompt).images

        i = 0
        for img in image:
            img.save(os.path.join(save_path, str(i)+".png"))
            i+=1
    break
