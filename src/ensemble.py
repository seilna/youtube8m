from inference import format_lines
import h5py
import numpy as np
from tensorflow import flags
from tensorflow import app
from tensorflow import gfile
from glob import glob

FLAGS = flags.FLAGS

if __name__ == "__main__":
  flags.DEFINE_string("ensemble_dir", "./ensemble", "checkpoint directory for ensemble")
  flags.DEFINE_integer("topk", 20, "topk")

  models = []
  for model in glob(FLAGS.ensemble_dir + "/*.h5"):
    models.append(h5py.File(model))
  num_models = len(models)

  video_ids = models[0].keys()
  video_ids.sort()

  with gfile.Open("./ensemble_predictions.csv", "w+") as outfile
    out_file.write("VideoId,LabelConfidencePairs\n")

    for video_id in video_ids:
      predictions = [ models[i][video_id] for i in range(num_models) ] # (num_models, 4716)
      ensemble_prediction_for_video = np.average(predictions, axis=0) # (4716)
      for line in format_lines(np.array([video_id]), ensemble_prediction_for_video, topk):
        out_file.write(line)
      out_file.flush()
