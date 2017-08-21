# coding: utf-8
import numpy as np
import Image
import ImageFont
import ImageDraw
import IPython
# text='abcdefghijklmnopqrstuvwxyz0123456789'
text='0123456789'
textlist = list(text)
data = np.array(textlist)
char_to_idx = {c:idx for idx,c in enumerate(textlist)}
idx_to_char = {idx:c for idx,c in enumerate(textlist)}
colors = ['red','green','blue','black','white','yellow']
fnt1 = ImageFont.truetype('/usr/share/fonts/truetype/droid/DroidSans.ttf',32)
fnt2 = ImageFont.truetype('/usr/share/fonts/truetype/droid/DroidSans.ttf',26)
fnt3 = ImageFont.truetype('/usr/share/fonts/truetype/droid/DroidSans.ttf',21)
fnts = [fnt1,fnt2,fnt3]
width=96
height=40
char_num = 4
def generate_vcode(width,height,code):
    label=Image.new('RGB',(width,height),(125,155,55))
    draw = ImageDraw.Draw(label)
    line_ys = np.random.randint(5,height,size=4)
    line_xs = np.random.randint(0,20,size=4)
    coloridx = np.random.randint(0,len(colors),size=2)

    draw.line((line_xs[0],line_ys[0],width-line_xs[1],line_ys[1]),fill=colors[coloridx[0]])
    draw.line((line_xs[2],line_ys[2],width-line_xs[3],line_ys[3]),fill=colors[coloridx[1]])

    startx = 10
    y = np.random.randint(3,8,size=len(code))
    padding = np.random.randint(1,5,size=len(code))
    fnt = np.random.randint(0,len(fnts),size=len(code))
    for i,c in enumerate(code):
        w,h = fnts[fnt[i]].getsize(c)
        if y[i]+h <= height:
            yy = y[i]
        else :
            yy=height-h
        draw.text((startx,yy),c,fill=(123,124,111),font=fnts[fnt[i]])
        startx += w+padding[i]
    imgdata=label.getdata()
    imgnp=np.array(imgdata)
    imgnp = imgnp.reshape(height,width,3).astype('uint8')
    return imgnp
def generate_data(data_size):
    imgs=[]
    labels=[]
    for i in range(data_size):
        code = np.random.choice(data,size=char_num).tostring()
        code_img = generate_vcode(width,height,code)
        target = np.zeros(char_num)
        for j,c in enumerate(code):
            target[j] = char_to_idx[c]
        imgs.append(code_img)
        labels.append(target)
    return np.array(imgs),np.array(labels)

if __name__ == '__main__':

    # code = np.random.choice(data,size=4).tostring()
    d,t = generate_data(10)
    IPython.embed()
