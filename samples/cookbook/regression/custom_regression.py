import argparse
import os

import model

import tensorflow as tf

print('TF Version')
print(tf.__version__)

from tensorflow.python.util.deprecation import deprecated
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam


def generate_experiment_fn(**experiment_args):
  """Create an experiment function.

  See command line help text for description of args.
  Args:
    experiment_args: keyword arguments to be passed through to experiment
      See `tf.contrib.learn.Experiment` for full args.
  Returns:
    A function:
      (tf.contrib.learn.RunConfig, tf.contrib.training.HParams) -> Experiment

    This function is used by learn_runner to create an Experiment which
    executes model code provided in the form of an Estimator and
    input functions.
  """
  def _experiment_fn(run_config, hparams):
    # num_epochs can control duration if train_steps isn't
    # passed to Experiment
    
    train_input = lambda: model.generate_input_fn(
        hparams.train_files,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.train_batch_size,
    )
    # Don't shuffle evaluation data
    eval_input = lambda: model.generate_input_fn(
        hparams.eval_files,
        batch_size=hparams.eval_batch_size,
        shuffle=False,
    )
    return tf.contrib.learn.Experiment(
        model.build_estimator(
            hparams_dict=hparams.values(),
            classes_files=hparams.classes_files,
            embedding_size=hparams.embedding_size,
            # Construct layers sizes with exponetial decay
            hidden_units=[
                max(2, int(hparams.first_layer_size *
                           hparams.scale_factor**i))
                for i in range(hparams.num_layers)
            ],
            config=run_config
        ),
        train_input_fn=train_input,
        eval_input_fn=eval_input,
        **experiment_args
    )
  return _experiment_fn


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--classes-files',
      help='GCS or local paths to classes data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int,
  )
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=100
  )
  parser.add_argument(
      '--eval-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True
  )

  # What features to use
  parser.add_argument(
      '--use-city',
      help='If to use city',
      choices=[
          'True',
          'False',
      ],
      default='False',
      required=False
  )

  parser.add_argument(
      '--use-weather',
      help='If to use weather',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-stat',
      help='If to use stat',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )  

  parser.add_argument(
      '--use-cf',
      help='If to use cf',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-ref',
      help='If to use ref',
      choices=[
          'True',
          'False',
      ],
      default='False',
      required=False
  )

  parser.add_argument(
      '--use-path',
      help='If to use path',
      choices=[
          'True',
          'False',
      ],
      default='False',
      required=False
  )

  parser.add_argument(
      '--use-tab',
      help='If to use tab',
      choices=[
          'True',
          'False',
      ],
      default='False',
      required=False
  )

  parser.add_argument(
      '--use-time',
      help='If to use time',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-country',
      help='If to use use country',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-dist',
      help='If to use use dist',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-mobile',
      help='If to use use mobile',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-ubl',
      help='If to use use ubl',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-expensive',
      help='If to use use expensive',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )
  
  parser.add_argument(
      '--use-visits',
      help='If to use use visits',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-pagelang',
      help='If to use use pagelang',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  parser.add_argument(
      '--use-position',
      help='If to use use position',
      choices=[
          'True',
          'False',
      ],
      default='True',
      required=False
  )

  # Training arguments
  parser.add_argument(
      '--embedding-size',
      help='Number of embedding dimensions for categorical columns',
      default=6,
      type=int
  )
  parser.add_argument(
      '--first-layer-size',
      help='Number of nodes in the first layer of the DNN',
      default=100,
      type=int
  )
  parser.add_argument(
      '--num-layers',
      help='Number of layers in the DNN',
      default=4,
      type=int
  )
  parser.add_argument(
      '--scale-factor',
      help='How quickly should the size of the layers in the DNN decay',
      default=0.7,
      type=float
  )
  parser.add_argument(
      '--drop-out',
      help='How much dropout',
      default=0.3,
      type=float
  )  
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # Argument to turn on all logging
  parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
  )
  # Experiment arguments
  parser.add_argument(
      '--eval-delay-secs',
      help='How long to wait before running first evaluation',
      default=10,
      type=int
  )
  parser.add_argument(
      '--min-eval-frequency',
      help='Minimum number of training steps between evaluations',
      default=None,  # Use TensorFlow's default (currently, 1000 on GCS)
      type=int
  )
  parser.add_argument(
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
  )
  parser.add_argument(
      '--export-format',
      help='The input format of the exported SavedModel binary',
      choices=['JSON', 'CSV', 'EXAMPLE'],
      default='JSON'
  )

  args = parser.parse_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  # learn_runner pulls configuration information from environment
  # variables using tf.learn.RunConfig and uses this configuration
  # to conditionally execute Experiment, or param server code
  learn_runner.run(
      generate_experiment_fn(
          min_eval_frequency=args.min_eval_frequency,
          eval_delay_secs=args.eval_delay_secs,
          train_steps=args.train_steps,
          eval_steps=args.eval_steps,
          export_strategies=[saved_model_export_utils.make_export_strategy(
              model.SERVING_FUNCTIONS[args.export_format],
              exports_to_keep=1,
              default_output_alternative_key=None
          )]
      ),
      run_config=run_config.RunConfig(model_dir=args.job_dir),
      hparams=hparam.HParams(**args.__dict__)
  )
