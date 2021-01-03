import splitfolders
from os import path

project_root = path.dirname(path.dirname(__file__))
input_folder = path.join(project_root, 'data1/nomal_data/')
output_folder = path.join(project_root, 'dataset/nomal_data/')

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .1, .1))
