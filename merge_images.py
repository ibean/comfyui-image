import math
import torch

class MergeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
            "images": ("IMAGE",),
            # "mode": (("horizontal", "vertical"), {"default": "horizontal"}),
          },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "image/postprocessing"

    def merge(self, images):
        # images.shape: batch, h, w, c
        batch = len(images)
        if batch == 1:
            return images.unsqueeze(0)
    
        max_h = max(image.shape[0] for image in images)
        max_w = max(image.shape[1] for image in images)
        c = images[0].shape[2]
    
        x = math.ceil(math.sqrt(batch))
        y = x - 1
        if x * y < batch:
            y = x
        total_images = x * y
    
        # Create a white image for padding
        white_img = torch.ones((max_h, max_w, c))
        black_img = torch.zeros((max_h, max_w, c))
    
        # Resize images and add white images to the list to make the total number of images a perfect square
        resized_images = []
        for i in range(total_images):
            if i < batch:
                img = images[i]
                h, w = img.shape[0], img.shape[1]
                if h < max_h or w < max_w:
                    img = torch.cat((img, white_img[h:, :w]), 0)
                    img = torch.cat((img, white_img[:h, w:]), 1)
                resized_images.append(img)
            else:
                resized_images.append(black_img)
    
        # Reshape the images to a grid
        image = torch.stack(resized_images).view(y, x, max_h, max_w, c)
    
        # Swap the axes to get the correct orientation
        image = image.transpose(1, 2).contiguous().view(1, y*max_h, x*max_w, c)
    
        return (image,)

    

NODE_CLASS_MAPPINGS = {
    "MergeImages": MergeImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MergeImages": "MergeImages Node"
}
