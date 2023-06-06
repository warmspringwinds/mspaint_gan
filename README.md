# :art: :paintbrush:  MS Paint GAN
Neural photo editor based on stylegan that allows to realistically edit images with an interface similar to MS paint.

In some way extends DragGan by adding color controls.

Work done during PhD @ JHU. 

# Examples

https://github.com/warmspringwinds/mspaint_gan/assets/2501383/9c460159-09e1-4692-a088-a5ddb130d89f

https://github.com/warmspringwinds/mspaint_gan/assets/2501383/77e8f47f-399e-453f-a9c4-42e51db4b1c5


https://github.com/warmspringwinds/mspaint_gan/assets/2501383/5f167244-b491-4724-a43d-ba0b42930a13


# Setup

Download stylegan wegiths from [here](https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt) and place into ```mspaint_gan``` folder.

# Possible improvements

The project was done as a part of experiments related to the [paper]([https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt](https://github.com/warmspringwinds/segmentation_in_style).

* Adjust learning rate to make effects more/less dramatic.
* Switch to stylegan-2 and restrict the changes only to certain level of latents to make results more stable.
* Incorporate more ideas from DragGan to restric the changes to a certain region.

# Acknowledgements

The codebase borrows code and extends ideas from:
1. https://github.com/lernapparat/lernapparat
2. https://github.com/ajbrock/Neural-Photo-Editor
