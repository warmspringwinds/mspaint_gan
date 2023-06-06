import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from tkinter import * # Note that I dislike the * on the Tkinter import, but all the tutorials seem to do that so I stuck with it.
from tkinter.colorchooser import askcolor
#from tkColorChooser import askcolor # This produces an OS-dependent color selector. I like the windows one best, and can't stand the linux one.
from collections import OrderedDict
from PIL import Image, ImageTk
import numpy as np
import scipy.misc
import skimage.io as io
from lib import g_all as model
from lib import device
import torch

# Create master
master = Tk()
master.title( "Neural Photo Editor" )

# RGB interpreter convenience function
def rgb(r,g,b):
    return '#%02x%02x%02x' % (r,g,b)
    
# Convert RGB to bi-directional RB scale.
def rb(i):
    # return rgb(int(i*int(i>0)),0, -int(i*int(i<0)))
    return rgb(255+max(int(i*int(i<0)),-255),255-min(abs(int(i)),255), 255-min(int(i*int(i>0)),255))

# Convenience functions to go from [0,255] to [-1,1] and [-1,1] to [0,255]    
def to_tanh(input):
    return 2.0*(input/255.0)-1.0
 
def from_tanh(input):
    return 255.0*(input+1)/2.0

Z = torch.randn(1, 512, device=device)

Z_big = torch.nn.Parameter( model.g_mapping(Z).clone())
OPTIMIZER = torch.optim.SGD((Z_big,), lr=100000.0)


## TODO: we should probably remove the latent variables?
### Latent Canvas Variables
# Latent Square dimensions
dim = [23,23]


# Pixel-wise resolution for latent canvas
res = 16

# Array that holds the actual latent canvas
r = np.zeros((res*dim[0],res*dim[1]),dtype=np.float32)

# Painted rectangles for free-form latent painting
painted_rects = []

# Actual latent rectangles
rects = np.zeros((dim[0],dim[1]),dtype=int)

### Output Display Variables

# RGB paintbrush array
# and image that will be completely filled with selected color
myRGB = np.zeros((1, 3, 1024, 1024), dtype=np.float32)

# Canvas width and height
canvas_width = 400
canvas_height = 400

# border width
bd =2 
# Brush color
color = IntVar() 
color.set(0)

# Brush size
d = IntVar() 
d.set(12)#12

# Selected Color
mycol = [0,0,0]

def getColor():
    global myRGB, mycol
    col = askcolor((mycol[0], mycol[1], mycol[2]))
    if col[0] is None:
        return # Dont change color if Cancel pressed.
    print(col)
    mycol[0] = int(col[0][0])
    mycol[1] = int(col[0][1])
    mycol[2] = int(col[0][2])
    for i in range(3): myRGB[0,i,:,:] = int(mycol[i]); # assign

def update_photo(data=None,widget=None):
    
    global Z_big
    
    if data is None:
        
        with torch.no_grad():
            img = model.g_synthesis(Z_big)
            img_normalized = (img.clamp(-1, 1) + 1) / 2.0
            data = (img_normalized.cpu().squeeze().detach().numpy()[:, ::4, ::4] * 255).astype(np.uint8)
    
    if widget is None:
        widget = output
    # Reshape image to canvas
    mshape = (256,256,1)
    im = Image.fromarray(np.concatenate([np.reshape(data[0],mshape),np.reshape(data[1],mshape),np.reshape(data[2],mshape)],axis=2),mode='RGB')
    
    # Make sure photo is an object of the current widget so the garbage collector doesn't wreck it
    widget.photo = ImageTk.PhotoImage(image=im)
    widget.create_image(0,0,image=widget.photo,anchor=NW)
    widget.tag_raise(pixel_rect)

# This function just displays a region that will be affected
# after coloring.
def move_mouse( event ):
    global output
    # using a rectangle width equivalent to d/4 (so 1-16)
    
    # First, get location and extent of local patch
    x,y = event.x//4,event.y//4
    brush_width = ((d.get()//4)+1)
    
    # if x is near the left corner, then the minimum x is dependent on how close it is to the left
    xmin = max(min(x-brush_width//2,64 - brush_width),0) # This 64 may need to change if the canvas size changes
    xmax = xmin+brush_width
    
    ymin = max(min(y-brush_width//2,64 - brush_width),0) # This 64 may need to change if the canvas size changes
    ymax = ymin+brush_width
    
    # update output canvas
    output.coords(pixel_rect,4*xmin,4*ymin,4*xmax,4*ymax)
    output.tag_raise(pixel_rect)
    output.itemconfig(pixel_rect,outline=rgb( int(mycol[0]), int(mycol[1]), int(mycol[2]) ) )

def paint( event ):
    global Z_big, output, myRGB, IM, ERROR, RECON, USER_MASK, SAMPLE_FLAG
    
    # Move the paintbrush
    move_mouse(event)
    
    # Get paintbrush location
    #res = [int(coordinate * 4) for coordinate in output.coords(pixel_rect)]
    [c1,r1,c2,r2] = [int(coordinate * 4) for coordinate in output.coords(pixel_rect)]
    
    
    tanh_target = np.float32(to_tanh(myRGB))
    tanh_target = torch.tensor(tanh_target, dtype=torch.float32).cuda()
    
    OPTIMIZER.zero_grad()
    
    tanh_output = model.g_synthesis(Z_big)
    
    loss = ( (tanh_output[0, :, r1:r2, c1:c2] - tanh_target[0, :, r1:r2, c1:c2])**2 ).sum()
    loss.backward()
    OPTIMIZER.step()
    
    update_canvas(w)
    update_photo(None, output)

# Change brush size
def update_brush(event):
    brush.create_rectangle(0,0,25,25,fill=rgb(255,255,255),outline=rgb(255,255,255))
    brush.create_rectangle( int(12.5-d.get()/4.0), int(12.5-d.get()/4.0), int(12.5+d.get()/4.0), int(12.5+d.get()/4.0), fill = rb(color.get()),outline = rb(color.get()) )

def update_canvas(widget=None):
    global r, Z, res, rects, painted_rects
    if widget is None:
        widget = w
    # Update display values
    # Here the Z values are upsampled using nearest neighbour
    # interpolation
    #r = np.repeat(np.repeat(Z,r.shape[0]//Z.shape[0],0),r.shape[1]//Z.shape[1],1)
    
    # If we're letting freeform painting happen, delete the painted rectangles
    for p in painted_rects:
        w.delete(p)
        
    painted_rects = []
    
    for i in range(dim[0]):
        for j in range(dim[1]):

            flattened_idx = i*dim[1] + j

            if flattened_idx >= 512:
                continue
                
            w.itemconfig(int(rects[i,j]),fill = rb(255*Z[0, flattened_idx]),outline = rb(255*Z[0, flattened_idx]))

def Reset():
    global Z, Z_big, OPTIMIZER
     
    Z = torch.randn(1, 512, device=device)

    Z_big = torch.nn.Parameter( model.g_mapping(Z).clone())
    
    OPTIMIZER = torch.optim.SGD((Z_big,), lr=0.000005)
    
    update_canvas(w)
    update_photo(None, output)

### Prepare GUI
#master.bind("<MouseWheel>",scroll)

# Prepare drawing canvas
f=Frame(master)
f.pack(side=TOP)
#output = Canvas(f,name='output',width=64*4,height=64*4)
output = Canvas(f,name='output',width=256,height=256)
output.bind('<Motion>',move_mouse)
output.bind('<B1-Motion>', paint )
pixel_rect = output.create_rectangle(0,0,4,4,outline = 'yellow')
output.pack()   

# Prepare latent canvas
#f = Frame(master,width=res*dim[0],height=dim[1]*10)
f = Frame(master,width=res*dim[0], height=dim[1] * 10)
f.pack(side=TOP)
w = Canvas(f,name='canvas', width=res*dim[0],height=res*dim[1])
#w.bind( "<B1-Motion>", paint_latents )
# Produce painted rectangles
for i in range(dim[0]):
    for j in range(dim[1]):
        
        flattened_idx = i*dim[1] + j
        
        if flattened_idx >= 512:
            continue
            
        rects[i,j] = w.create_rectangle( j*res, i*res, (j+1)*res, (i+1)*res, fill = rb(255*Z[0, flattened_idx]),outline = rb(255*Z[0, flattened_idx]) )
# w.create_rectangle( 0,0,res*dim[0],res*dim[1], fill = rgb(255,255,255),outline=rgb(255,255,255)) # Optionally Initialize canvas to white 
w.pack()


# Color gradient    
gradient = Canvas(master, width=400, height=20)
gradient.pack(side=TOP)
# gradient.grid(row=i+1)
for j in range(-200,200):
    gradient.create_rectangle(j*255/200+200,0,j*255/200+201,20,fill = rb(j*255/200),outline=rb(j*255/200))
# Color scale slider
f= Frame(master)
Scale(master, from_=-255, to=255,length=canvas_width, variable = color,orient=HORIZONTAL,showvalue=0,command=update_brush).pack(side=TOP)

#Scale(master, from_=-255, to=255,length=canvas_width, variable = color,orient=HORIZONTAL,showvalue=0,command=None).pack(side=TOP)

# Buttons and brushes
#Button(f, text="Sample", command=sample).pack(side=LEFT)
Button(f, text="Sample", command=None).pack(side=LEFT)
Button(f, text="Reset", command=Reset).pack(side=LEFT)
#Button(f, text="Update", command=UpdateGIM).pack(side=LEFT)
Button(f, text="Update", command=None).pack(side=LEFT)
brush = Canvas(f,width=25,height=25)
#Scale(f, from_=0, to=64,length=100,width=25, variable = d,orient=HORIZONTAL,showvalue=0,command=update_brush).pack(side=LEFT) # Brush diameter scale
Scale(f, from_=0, to=64,length=100,width=25, variable = d,orient=HORIZONTAL,showvalue=0,command=None).pack(side=LEFT) # Brush diameter scale
brush.pack(side=LEFT)
#inferbutton = Button(f, text="Infer", command=infer)
inferbutton = Button(f, text="Infer", command=None)
inferbutton.pack(side=LEFT)
colorbutton=Button(f,text='Col',command=getColor)
colorbutton.pack(side=LEFT) 
myentry = Entry()
myentry.pack(side=LEFT)
f.pack(side=TOP)

print('Running')  
# Reset and infer to kick it off
Reset()
#infer() 
mainloop()

# TODO:

# Plug in our network -- add model def -- adjust the size of the window to fit in the picture
