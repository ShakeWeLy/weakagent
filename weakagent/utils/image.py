import numpy as np
import cv2
import base64
import os

def image_to_base64(image: np.ndarray | str) -> str:
    """
    Transform image to base64 data URI
    """

    # numpy image
    if isinstance(image, np.ndarray):
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Image encoding failed")

        base64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"

    # string input
    if isinstance(image, str):

        # data url
        if image.startswith("data:image"):
            return image

        # local path
        if os.path.exists(image):
            with open(image, "rb") as f:
                base64_str = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(image)[1].lower()
            mime = (
                "image/jpeg"
                if ext in {".jpg", ".jpeg"}
                else "image/png"
                if ext == ".png"
                else "image/webp"
                if ext == ".webp"
                else "image/png"
            )
            return f"data:{mime};base64,{base64_str}"

        # raw base64
        try:
            base64.b64decode(image, validate=True)
            return f"data:image/png;base64,{image}"
        except Exception:
            pass

        raise ValueError("Invalid image string")

    raise TypeError(f"Unsupported image type: {type(image)}")
