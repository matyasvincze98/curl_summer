################################################################################
# Copyright 2019 DeepMind Technologies Limited
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
################################################################################
"""Training file with continual features turned off

--- 8< ---

The default parameters corresponding to the first set of experiments in Section
4.2.

For the expansion ablation, run with different ll_thresh values as in the paper.
Note that n_y_active represents the number of *active* components at the
start, and should be set to 1, while n_y represents the maximum number of
components allowed, and should be set sufficiently high (eg. n_y = 100).

For the MGR ablation, setting use_sup_replay = True switches to using SMGR,
and the gen_replay_type flag can switch between fixed and dynamic replay. The
generative snapshot period is set automatically in the train_curl.py file based
on these settings (ie. the data_period variable), so the 0.1T runs can be
reproduced by dividing this value by 10.
"""

from datetime import datetime
import numpy as np

import training


def main(unused_argv):
  date_time = datetime.today().isoformat()

  train_eval_ops, test_eval_ops, sess, params, saver = training.run_training(
      dataset='textures',
      dataset_params={'batch_size': 128,
                      'test_batch_size': 128,
                      'train_every': 1,
                      'test_every': 1,
                      'crop_dim': 20,
                      'path': 'fakelabeled_texture_oatleathersoilcarpetbubbles_subsamp1_filtered_128000_48px.pkl',
                      'offset': 0.0
                      },
      output_type='normal',
      output_sd=0.4,
      n_y=20,
      n_y_samples=3,
      n_y_samples_reconstr=1,
      beta_y_evo=1.0,
      n_z=450,
      # beta_z_evo=np.linspace(0.2, 1.0, num=5),
      beta_z_evo=1.0,
      lr_init=1e-4,
      lr_factor=1.,
      lr_schedule=[1],
      n_steps=20,
      report_interval=10,
      random_seed=None,
      encoder_kwargs={
          'encoder_type': 'mlp',
          'n_enc': [500],
          'enc_strides': [1]
      },
      cluster_encoder_kwargs={
          'encoder_type': 'mlp',
          'n_enc': [400, 250, 100]
      },
      latent_y_to_concat_encoder_kwargs={
          'y_to_concat_encoder_type': 'mlp',
          'y_to_concat_n_enc': [100, 250, 400, 500]
      },
      latent_concat_to_z_encoder_kwargs={
          'concat_to_z_encoder_type': 'mlp',
          'concat_to_z_n_enc': []
      },
      latent_decoder_kwargs={
          'decoder_type': 'mlp',
          'n_dec': [100, 250, 400, 500]
      },
      decoder_kwargs={
          'decoder_type': 'mlp',
          'n_dec': [],
          'dec_up_strides': None
      },
      z1_distr_kwargs={
          'distr': 'laplace',
          'sigma_nonlin': 'exp',
          'sigma_param': 'var'
      },
      z2_distr_kwargs={
          'distr': 'normal',
          'sigma_nonlin': 'exp',
          'sigma_param': 'var'
      },
      l2_lambda_w=0.001,
      l2_lambda_b=0.001,
      gradskip_threshold=0.11e4,
      gradclip_threshold=0.1e4,
      save_dir='logs/cont_z2_test',
      # save_dir='logs/' + date_time
      restore_from=None,
      # restore_from='mycurl-20',
      tb_dir=None
      # tb_dir='tb_logs/' + date_time
  )


if __name__ == '__main__':
  main([])
