import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def main():
    # Set your paths
    input_dir = "/home/burak/development/pytorch-CycleGAN-and-pix2pix/results/str_driving_rgb2domain5_cyclegan_rect_400x300/test_70/images"
    output_mp4 = "/home/burak/development/pytorch-CycleGAN-and-pix2pix/results/str_comparison_v1_domain5_epoch_70.mp4"

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

    # Get the first image to determine the frame size
    first_real = Image.open(os.path.join(input_dir, real_images[0]))
    width, height = first_real.size
    
    # Create video writer (MP4V codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, 5.0, (width * 2, height))  # 5 FPS

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
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("Arial", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "Original", (255, 255, 255), font=font)
            draw.text((real.width + 10, 10), "Augmented", (255, 255, 255), font=font)
            
            # Convert PIL Image to OpenCV format (BGR)
            frame = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
            
            # Write the frame to the video
            out.write(frame)
            
        except Exception as e:
            print(f"Error processing {real_img} | {fake_img}: {e}")
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_mp4}")

if __name__ == "__main__":
    main()
