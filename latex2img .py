import os
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.font_manager as mfm
from matplotlib import mathtext


class latex2img:
    def __init__(self, text,chara_size = 32, **kwds):
        
        """LaTex数学公式转图片
            
            text        - 文本字符串，其中数学公式须包含在两个$符号之间
            chara_size  - 字体大小, 默认32
            kwds        - 关键字参数
                            dpi         - 输出分辨率(每英寸像素数),默认72
                            family      - 系统支持的字体,None表示当前默认的字体
                            weight      - 笔画轻重,可选项包括:normal(默认)、light和bold
            """
        
        text.replace(" _ ","_")
        text.replace(" ^ ","^")
        self.text = text
        for key in kwds:
            if key not in ['dpi', 'family', 'weight']:
                raise KeyError('不支持的关键字参数：%s'%key)
        
        dpi = kwds.get('dpi', 72)
        family = kwds.get('family', None)
        weight = kwds.get('weight', 'normal')
        
        bfo = BytesIO() 
        prop = mfm.FontProperties(family=family,style='italic', size=chara_size, weight=weight)
        mathtext.math_to_image(self.text, bfo, prop=prop, dpi=dpi)
        im = Image.open(bfo)
        self.img = im

    def get_img(self):
        return self.img

    def get_img_nparray(self):
        im = self.img
        r, g, b, a = im.split()
        im = np.dstack((r,g,b,a)).astype(np.uint8)
        return im

    def change_color(self,color=(0.1,0.1,0.1)):
        #改变颜色
        #color       - 颜色，浮点型三元组，值域范围[0,1]，默认深黑色
        im = self.img
        r, g, b, a = im.split()
        r, g, b = 255-np.array(r), 255-np.array(g), 255-np.array(b)
        a = r/3 + g/3 + b/3
        r, g, b = r*color[0], g*color[1], b*color[2]  
        im = np.dstack((r,g,b,a)).astype(np.uint8)
        im = Image.fromarray(im)
        self.img = im
    def resize(self,size:tuple=(75,83)):
        #self.img = self.img.resize(size)
        im = self.img
        b,a = im.size
        wid = size[0]-a
        hig = size[1]-b
        up = int(np.floor((wid)/2))
        down = int(wid-up)
        right = int(np.floor((hig)/2))
        left = int(hig-right)
        #print(up,down,right,left)
        r, g, b, a = im.split()
        im = np.dstack((r,g,b,a)).astype(np.uint8)
        
        im =  np.pad(im,((up,down),(right,left),(0,0)),'constant', constant_values=(255,255))
        
        im = Image.fromarray(im)
        
        self.img = im

    def save_fig(self,path:str):
        assert path is None or os.path.splitext(path)[1].lower() == '.png', 'only support ".png", please input complete filename'
        im = self.img
        im.save(path)

    
        

        

if __name__  == '__main__':
    text = '$\sum_{i=0} ^ \infty a_i$'
    #aa = [item.name for item in mfm.fontManager.ttflist]
    #for i in range(len(aa)):
    a = latex2img(text)
    b = a.get_img()
    c = latex2img(text,family='STIXSizeThreeSym')
    d = c.get_img()
    b.show()
    d.show()

    
    
    
    #a.save_fig("./demo.png")
    #print("family: ",'HGB1X_CNKI' in [item.name for item in mfm.fontManager.ttflist])
    