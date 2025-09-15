# نصب وابستگی‌ها
!pip install diffusers==0.29.2 transformers==4.39.3 accelerate==0.29.3 torchvision==0.17.2 ftfy==6.2.0 tensorboard==2.16.2 jinja2==3.1.3 peft==0.7.0 datasets==0.18.0 wandb==0.16.6 scipy==1.13.0 lightning==2.2.2 gradio==4.28.3 hf-transfer==0.1.6 torch==2.2.2 safetensors==0.4.3 pillow==10.3.0 huggingface_hub==0.23.4 numpy==1.23.5

# وارد کردن کتابخانه‌ها
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import os

# چک کردن GPU
if torch.cuda.is_available():
    print("GPU فعال است.")
else:
    print("GPU فعال نیست! لطفاً از Runtime > Change runtime type > GPU استفاده کنید.")

# آپلود تصویر ورودی
print("لطفاً یک تصویر (مثلاً JPG یا PNG) آپلود کنید")
uploaded = files.upload()
if not uploaded:
    raise ValueError("هیچ تصویری آپلود نشد! لطفاً یک تصویر آپلود کنید")

# گرفتن نام فایل تصویر
image_name = list(uploaded.keys())[0]

# پیش‌پردازش تصویر ورودی
try:
    init_image = Image.open(image_name).convert("RGB")
except Exception as e:
    print(f"خطا در باز کردن تصویر: {e}")
    raise

# تغییر اندازه تصویر
width, height = init_image.size
aspect_ratio = width / height
target_size = 512

if aspect_ratio > 1:
    new_width = int(target_size * aspect_ratio)
    new_height = target_size
else:
    new_width = target_size
    new_height = int(target_size / aspect_ratio)
init_image_resized = init_image.resize((new_width, new_height), Image.LANCZOS)

# کراپ به 512x512
left = (new_width - 512) // 2
top = (new_height - 512) // 2
init_image_cropped = init_image_resized.crop((left, top, left + 512, top + 512))

# پیش‌نمایش تصویر کراپ‌شده
img_preview = cv2.cvtColor(np.array(init_image_cropped), cv2.COLOR_RGB2BGR)
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB))
plt.title("پیش‌نمایش تصویر کراپ‌شده")
plt.axis('off')
plt.show()

print("آیا تصویر کراپ‌شده مورد تأیید است؟ اگر خیر، لطفاً تصویر را دستی کراپ کنید و دوباره آپلود کنید.")

# تنظیم دستگاه و مدل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

# بارگذاری مدل پایه (به جای مدل فاین‌تیون‌شده)
model_path = "lavaman131/cartoonify"  # مدل پایه
print(f"بارگذاری مدل از: {model_path}")
try:
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        use_safetensors=True
    ).to(device)
    print("مدل با موفقیت بارگذاری شد.")
except Exception as e:
    print(f"خطا در بارگذاری مدل: {e}")
    raise

# غیرفعال کردن Safety Checker
pipeline.safety_checker = None

# تعریف پرامپت و پارامترها
prompt = "A disney style cartoon character of sks person, vibrant colors, highly detailed, smooth shading, preserve exact facial features, dynamic lighting, clean lines, disney character design, in a bright sky with fluffy clouds background"
negative_prompt = "blurry, low quality, realistic style, distorted faces, different people, generic faces, extra limbs, deformed features, grainy, dull colors, added earrings, changed facial features, dark background, no clouds"
guidance_scale = 12.0
strength = 0.6
num_inference_steps = 50

# تولید تصویر کارتونی
print("در حال تولید تصویر کارتونی...")
try:
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image_cropped,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    print("تصویر کارتونی با موفقیت تولید شد.")
except Exception as e:
    print(f"خطا در تولید تصویر: {e}")
    raise

# ذخیره و نمایش تصاویر
image.save("output_cartoon_base.png")

img_output = cv2.imread("output_cartoon_base.png")
if img_output is None:
    print("خطا در بارگذاری تصویر خروجی!")
    raise ValueError("تصویر خروجی بارگذاری نشد")

img_input = cv2.cvtColor(np.array(init_image_cropped), cv2.COLOR_RGB2BGR)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
axes[0].set_title("تصویر ورودی (پس از پیش‌پردازش)")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))
axes[1].set_title("تصویر کارتونی (مدل پایه)")
axes[1].axis('off')

plt.tight_layout()
plt.show()
