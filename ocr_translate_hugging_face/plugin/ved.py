###################################################################################
# ocr_translate-hugging_face - a plugin for ocr_translate                         #
# Copyright (C) 2023-present Davide Grassano                                      #
#                                                                                 #
# This program is free software: you can redistribute it and/or modify            #
# it under the terms of the GNU General Public License as published by            #
# the Free Software Foundation, either version 3 of the License.                  #
#                                                                                 #
# This program is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                   #
# GNU General Public License for more details.                                    #
#                                                                                 #
# You should have received a copy of the GNU General Public License               #
# along with this program.  If not, see {http://www.gnu.org/licenses/}.           #
#                                                                                 #
# Home: https://github.com/Crivella/ocr_translate-hugging_face                    #
###################################################################################
"""Class to enable hugginface VisionEncoderDecoder models."""
import logging

import torch
from ocr_translate import models as m
from PIL import Image

from .utils import EnvMixin, Loaders

logger = logging.getLogger('plugin')

class HugginfaceVEDModel(m.OCRModel, EnvMixin):
    """OCRtranslate plugin to allow loading of hugginface VisionEncoderDecoder model as text OCR."""
    class Meta: # pylint: disable=missing-class-docstring
        proxy = True

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model = None
        self.image_processor = None

    def load(self):
        """Load the model into memory."""
        logger.info(f'Loading OCR VED model: {self.name}')
        res = Loaders.load(
            self.name, request=['ved_model'],
            root=self.root, dev=self.dev
            )
        self.model = res['ved_model']
        res = Loaders.load(
            self.tokenizer_name or self.name, request=['tokenizer'],
            root=self.root, dev=self.dev
            )
        self.tokenizer = res['tokenizer']
        res = Loaders.load(
            self.processor_name or self.name, request=['image_processor'],
            root=self.root, dev=self.dev
            )
        self.image_processor = res['image_processor']

    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.image_processor is not None:
            del self.image_processor
            self.image_processor = None

        if self.dev == 'cuda':
            torch.cuda.empty_cache()

    def _ocr(
            self,
            img: Image.Image, lang: str = None, options: dict = None
            ) -> str:
        """Perform OCR on an image.

        Args:
            img (Image.Image):  A Pillow image on which to perform OCR.
            lang (str, optional): The language to use for OCR. (Not every model will use this)
            bbox (tuple[int, int, int, int], optional): The bounding box of the text on the image in lbrt format.
            options (dict, optional): A dictionary of options to pass to the OCR model.

        Raises:
            TypeError: If img is not a Pillow image.

        Returns:
            str: The text extracted from the image.
        """
        if self.model is None or self.tokenizer is None or self.image_processor is None:
            raise RuntimeError('Model not loaded')

        if options is None:
            options = {}

        pixel_values = self.image_processor(img, return_tensors='pt').pixel_values
        if self.dev == 'cuda':
            pixel_values = pixel_values.cuda()
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if self.dev == 'cuda':
            torch.cuda.empty_cache()

        return generated_text
