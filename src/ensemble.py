from inference import format_lines
import h5py
import numpy as np
from tensorflow import flags
from tensorflow import app
from tensorflow import gfile
from glob import glob
from tqdm import tqdm
from time import time
FLAGS = flags.FLAGS

if __name__ == "__main__":
  flags.DEFINE_string("ensemble_dir", "./ensemble", "checkpoint directory for ensemble")
  flags.DEFINE_integer("topk", 20, "topk")
  flags.DEFINE_integer("batch_size", 1024, "batch size")

  models = []
  for model in glob(FLAGS.ensemble_dir + "/*.h5"):
    models.append(h5py.File(model))
  num_models = len(models)

  video_ids = models[0].keys()
  video_ids.sort()

  num_videos = len(video_ids)
  batch_size = FLAGS.batch_size
  with gfile.Open("./ensemble_predictions.csv", "w+") as out_file:
    out_file.write("VideoId,LabelConfidencePairs\n")
		
		
    for video_id in tqdm(range(0, num_videos, batch_size)): 
      video_id_batch = []
      predictions = [] # (num_model, batch, 4716)
      start_time = time()
      for i in range(num_models):
        predictions_per_model = [] # (batch, 4716)
        for batch_id in range(video_id, min([video_id + batch_size, num_videos])):
          if i==0: video_id_batch.append(video_ids[batch_id])

          predictions_per_model.append(models[i][video_ids[batch_id]]) # (4716)

        predictions_per_model = np.vstack(predictions_per_model) # (batch, 4716)

        predictions.append(predictions_per_model)
      predictions = np.array(predictions) #(num_model, batch, 4716) 
      ensemble_prediction_for_video = np.average(predictions, axis=0) # (batch, 4716)
      for line in format_lines(video_id_batch, ensemble_prediction_for_video, FLAGS.topk):
        out_file.write(line)
      out_file.flush()
