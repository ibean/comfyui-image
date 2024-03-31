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
        batch, h, w, c = images.shape
        if batch == 1:
            return images.unsqueeze(0)
        x = math.ceil(math.sqrt(batch))
        y = x - 1
        if x * y < batch:
            y = x
        total_images = x * y
    
        # Create a black image for padding
        black_img = torch.zeros((h, w, c))
    
        # Add black images to the list to make the total number of images a perfect square
        for i in range(total_images - batch):
            images = torch.cat((images, black_img.unsqueeze(0)), 0)
    
        # Reshape the images to a grid
        image = images.view(y, x, h, w, c)
        
        # Swap the axes to get the correct orientation
        image = image.transpose(1, 2).contiguous().view(1, y*h, x*w, c)
    
        return (image,)

    

NODE_CLASS_MAPPINGS = {
    "MergeImages": MergeImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MergeImages": "MergeImages Node"
}
