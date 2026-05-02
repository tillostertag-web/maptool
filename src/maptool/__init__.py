"""maptool — climate-painted terrain map builder."""

from PIL import Image

# At ultra quality (max_dim=20480) outputs reach ~200 MP, above PIL's default
# 178 MP safety limit. We trust our own files; lift the cap project-wide.
Image.MAX_IMAGE_PIXELS = None

__version__ = "0.1.0"
