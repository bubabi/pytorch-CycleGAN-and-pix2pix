import os
import imageio
from PIL import Image
import numpy as np

# Set your paths
input_dir = "/home/burak/development/pytorch-CycleGAN-and-pix2pix/results/str_driving_rgb2domain1_cyclegan_rect/test_70/images"
output_gif = "/home/burak/development/pytorch-CycleGAN-and-pix2pix/results/str_comparison_v1_epoch_70.gif"

# Get all real and fake images
real_images = sorted([f for f in os.listdir(input_dir) if f.endswith('_real.png')])
fake_images = sorted([f for f in os.listdir(input_dir) if f.endswith('_fake.png')])

print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")

# Make sure we have matching pairs
if len(real_images) != len(fake_images):
    print("Warning: Mismatch in number of real and fake images")
    min_len = min(len(real_images), len(fake_images))
    real_images = real_images[:min_len]
    fake_images = fake_images[:min_len]
    print(f"Using first {min_len} images from each set")

# Create a list to store frames
frames = []

for i, (real_img, fake_img) in enumerate(zip(real_images, fake_images)):
    print(f"Processing pair {i+1}/{len(real_images)}: {real_img} | {fake_img}")
    
    try:
        # Open images
        real = Image.open(os.path.join(input_dir, real_img))
        fake = Image.open(os.path.join(input_dir, fake_img))
        
        # Create a new image with double width
        combined = Image.new('RGB', (real.width * 2, real.height))
        
        # Paste the images side by side
        combined.paste(real, (0, 0))
        combined.paste(fake, (real.width, 0))
        
        # Add text labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("Arial", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "clear", (255, 255, 255), font=font)
        draw.text((real.width + 10, 10), "foggy", (255, 255, 255), font=font)
        
        # Convert to numpy array for imageio
        frames.append(np.array(combined))
        print(f"Successfully added frame {i+1}")
    except Exception as e:
        print(f"Error processing {real_img} | {fake_img}: {str(e)}")
        continue

# Save as GIF if we have any frames
if frames:
    print(f"Saving {len(frames)} frames to {output_gif}")
    imageio.mimsave(output_gif, frames, duration=0.8)  # 0.5 seconds per frame
    print("GIF creation complete!")
else:
    print("No frames were processed. Check the input directory and file patterns.")