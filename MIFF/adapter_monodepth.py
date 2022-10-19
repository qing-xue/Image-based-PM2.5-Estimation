import torch
import os
import PIL.Image as pil
from torchvision import transforms
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

import os
from config import PM_PATH
from common.monodepth2 import networks


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MonoDepth_Adapter(object):

    def __init__(self):
        model_name = 'mono+stereo_640x192'
        model_dir = os.path.join(PM_PATH, r'common/monodepth2/models')

        model_path = os.path.join(model_dir, model_name)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=DEVICE)

        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(DEVICE)
        encoder.eval()

        # print("   Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=DEVICE)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(DEVICE)
        depth_decoder.eval()

        self.feed_height = feed_height
        self.feed_width = feed_width
        self.encoder = encoder
        self.depth_decoder = depth_decoder

    def test(self, input_image):
        # input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(DEVICE)
        features = self.encoder(input_image)
        outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        return im


monodepth_net = MonoDepth_Adapter()