from PIL import Image
import os


###### Cuts the image into size of 64 * 64
w, h = 64,64
im = Image.open('meerut-19/meerut-19.tif')
im_w, im_h = im.size
print ('Image width:%d height:%d  will split into (%d %d) ' % (im_w, im_h, w, h))
w_num, h_num = int(im_w/w), int(im_h/h)
count = 0
for wi in range(0, w_num):
    for hi in range(0, h_num):
        box = (wi*w, hi*h, (wi+1)*w, (hi+1)*h)
        piece = im.crop(box)
        tmp_img = Image.new('RGB', (w, h), 255)
        tmp_img.paste(piece)
        img_path = os.path.join("meerut-19/%d.png" % (count))
        tmp_img.save(img_path)
        count += 1