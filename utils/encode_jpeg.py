from PIL import Image
import io
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor

def pillow_encode(img, fmt='jpeg', quality=10):
    assert len(img.shape) == 4, "Pillow_encode: tensor should be 4-dim (B,C,H,W)"
    assert img.shape[0] == 1, "Pillow_encode: tensor should be single image"
    img = img[0]
    img = to_pil_image(img)
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    rec = to_tensor(rec)
    rec = rec.unsqueeze(0)
    return rec, bpp

def find_closest_bpp(img, target, fmt='jpeg'):
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp