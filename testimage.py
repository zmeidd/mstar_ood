from PIL import Image
import base64
import io
im = Image.open("/home/bwang30/study/ann/mstar_with_machine_learning/MSTAR-10/newtask1train/BTR60/hb03787.jpeg")
print(im)
data = io.BytesIO()
print(data)