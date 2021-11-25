# Deep-AFM
A deep-learning implementation for the classification and analysis of different DNA conformations

## Helper Scripts
The helper scripts are designed to assisst in the pre-preporcessing pipeline and contain useful functions which aid the TopoStats ("labels.py") outputs to:
1) (json_edittor.py) Sort the produced topostats jsons into a dictionary containing all useful information per image (currently a circular/non circular flag, a contour length measure and the spline coordinates).
2) (image_transforms.py) Create rotated versions of images, grains and splines through 90˚, 180˚ and 270˚ clockwise and append them to the json dictionary created in 1. This is done to expand the dataset.
3) (transform_checker.py) Visualises the splines and grain heatmaps ontop of one another to ensure they match correctly.

The benefit of the new file format is that each image is linked by their name whether it's a:
<br>
png:    `<name+transform>_<channel>_<colour>.png`
<br>
grains: `<name+transform>_grains.txt`
<br>
json: `<name+transform>_spline.json`
<br>
json format:   `{<name>: {Circular: True/False..., Contour Length: float..., Splines: {x_coord: ..., y_coord: ...}}`
<br>
And so all grains, images and info contained in JSONs can be gathered via the `<name>` of the file (a.k.a the json first-leyer keys) as they are consistent between all objects.

