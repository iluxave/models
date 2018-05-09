# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts WHALENET data to TFRecord file format with Example protos.

WHALENET dataset is expected to have the following directory structure:

  + whalenet
    - build_data.py
    - build_whalenet_data.py (current working directory).
    + train
    - whales.json
    - whales-e.json
    + tfrecord

Image folder:
  ./train

Semantic segmentation annotations:
  ./whales.json, ./whales-e.json

list folder:
  ???

This script converts data into sharded data files and saves in tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import build_data
import tensorflow as tf
import io
import json
from PIL import Image, ImageDraw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           './train',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    './',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


_NUM_SHARDS = 4

split_to_file={
    'train': 'whales.json',
    'test': 'whales-e.json',
    'val': 'whales-e.json'
}

class_map = {
    "tail": 1
}

MAX_DIM=900.0

def generate_seg_data(annotations, width, height):
    if height>width: max_dim=height
    else: max_dim=width
    scale=1.0
#    if max_dim > MAX_DIM:
#        scale = MAX_DIM/max_dim
#    else:
#        scale = 1.0
    height = int(height*scale)
    width  = int(width*scale)
    mask=Image.new('P', (width, height))
    draw=ImageDraw.Draw(mask)
    for ann in annotations:
        if 'corrected' not in ann or not ann['corrected']:
            continue
        # load the polygon, and find its bounding box
        polygon = []
        xn = [float(x) for x in ann["xn"].split(";")]
        yn = [float(y) for y in ann["yn"].split(";")]
        for x, y in zip(xn, yn):
            polygon.append((x*scale, y*scale))
        # Now turn the polygon into an image mask
        draw.polygon(polygon, fill=1) #class_map[annotations['class']]
        i=1
        draw.line(polygon, fill=255, width=3)
        draw.line([polygon[0], polygon[len(polygon)-1]], fill=255, width=3)
    pngio=io.BytesIO()
    mask.save(pngio, 'PNG')
    return pngio.getvalue()



def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  sys.stdout.write('Processing ' + dataset_split)
  dataset = dataset_split

  ann_file = os.path.join(FLAGS.semantic_segmentation_folder, split_to_file[dataset_split])
  with open(ann_file, 'rb') as f:
    annotations = json.load(f)
  num_images = len(annotations)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpeg', channels=3)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(annotations), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = annotations[i]['filename']
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_data = generate_seg_data(annotations[i]["annotations"], width, height)
        if seg_data is None:
          continue
        # Save the mask for review
        mask_file = os.path.basename(image_filename)
        mask_file = os.path.join('/tmp', mask_file)
        mask_file = open(mask_file, 'wb')
        mask_file.write(seg_data)
        mask_file.close()
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, annotations[i]['filename'], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  for dataset_split in ['train', 'val']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
