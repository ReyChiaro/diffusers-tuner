from PIL import Image


def default_processor(**kwargs) -> str | Image.Image | list[str | Image.Image]:
    def _is_image_path(maybe_path: str) -> bool:
        try:
            with Image.open(maybe_path) as maybe_img:
                maybe_img.verify()
            return True
        except (IOError, SyntaxError):
            return False

    return_list = []
    for _, dv in kwargs.items():
        if _is_image_path(dv):
            v = Image.open(dv).convert("RGB")
            return_list.append(v)
        else:
            return_list.append(v)
    return return_list
