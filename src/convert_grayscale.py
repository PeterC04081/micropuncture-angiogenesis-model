# convert_grayscale.py
# Convert fluorescence microscopy images (.jpg) to grayscale numpy arrays
# for use as initial conditions in the micropuncture model.
# Outputs *_raw.npy files into gray_scales/ directory.
#
# Usage: python src/convert_grayscale.py --source_folder /path/to/images

import os
os.environ["OMPI_MCA_btl"] = "^sm"

import argparse
import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description="Convert microscopy images to grayscale .npy arrays")
parser.add_argument("--source_folder", type=str, required=True, help="folder containing .jpg microscopy images")
parser.add_argument("--output_folder", type=str, default="gray_scales", help="output folder for .npy files")
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

for fname in sorted(os.listdir(args.source_folder)):
    if fname.endswith(".jpg"):
        img = Image.open(os.path.join(args.source_folder, fname)).convert("L")
        img_array = np.array(img, dtype=np.float32) / 255.0
        height, width = img_array.shape

        mesh = UnitSquareMesh(width-1, height-1)
        V = FunctionSpace(mesh, "CG", 1)
        phi = Function(V)

        class ImageFunction(UserExpression):
            def __init__(self, img, **kwargs):
                super().__init__(**kwargs)
                self.img = img
                self.ny, self.nx = img.shape
            def eval(self, values, x):
                xi = min(max(int(x[0] * (self.nx - 1)), 0), self.nx - 1)
                yi = min(max(int((1 - x[1]) * (self.ny - 1)), 0), self.ny - 1)
                values[0] = self.img[yi, xi]
            def value_shape(self):
                return ()

        phi.interpolate(ImageFunction(img_array, degree=1))

        base_name = os.path.splitext(fname)[0]

        # save FE vector and raw image array
        np.save(os.path.join(args.output_folder, f"{base_name}_phi.npy"), phi.vector().get_local())
        np.save(os.path.join(args.output_folder, f"{base_name}_raw.npy"), img_array)

        # save mesh
        mesh_path = os.path.join(args.output_folder, f"{base_name}_mesh.xdmf")
        with XDMFFile(mesh.mpi_comm(), mesh_path) as f:
            f.write(mesh)

        print(f"Saved: {base_name}_raw.npy, {base_name}_phi.npy")

print(f"\nDone. Output in: {args.output_folder}")
