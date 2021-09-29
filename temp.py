import pandas
import pickle5 as pickle

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import copy

from scipy import stats
from scipy import special
from scipy import ndimage

import training
import eval

run_training_params = {'dataset': 'textures',
                             'dataset_params': {'batch_size': 128, 'test_batch_size': 128, 'train_every': 2, 'test_every': 1, 'crop_dim': 40, 
                                                'path': '/content/fakelabeled_natural_commonfiltered_640000_40px.pkl', 'offset': 0.0}, 
                             'n_steps': 0, 'random_seed': None, 'lr_init': 0.001, 'lr_factor': 1.0, 'lr_schedule': [1], 
                             'output_type': 'normal', 'output_sd': 0.4, 'n_y': 20, 'n_y_samples': 1, 'n_y_samples_reconstr': 1, 'n_z': 1300, 
                             'beta_y_evo': 1.0, 'beta_z_evo': 1.0, 
                             'encoder_kwargs': {'encoder_type': 'mlp', 'n_enc': [1400], 'enc_strides': [1]}, 
                             'cluster_encoder_kwargs': {'encoder_type': 'mlp', 'n_enc': [1000, 600, 200]}, 
                             'latent_y_to_concat_encoder_kwargs': {'y_to_concat_encoder_type': 'mlp', 'y_to_concat_n_enc': [200, 600, 1000, 1400]}, 
                             'latent_concat_to_z_encoder_kwargs': {'concat_to_z_encoder_type': 'mlp', 'concat_to_z_n_enc': []}, 
                             'l2_lambda_w': 0.001, 'l2_lambda_b': 0.001, 
                             'gradskip_threshold': 0.11e4, 'gradclip_threshold': 0.1e4,
                             'decoder_kwargs': {'decoder_type': 'mlp', 'n_dec': [], 'dec_up_strides': None},
                             'latent_decoder_kwargs': {'decoder_type': 'mlp', 'n_dec': [200, 600, 1000, 1400]}, 
                             'z1_distr_kwargs': {'distr': 'laplace', 'sigma_nonlin': 'exp', 'sigma_param': 'var'}, 
                             'z2_distr_kwargs': {'distr': 'normal', 'sigma_nonlin': 'exp', 'sigma_param': 'var'}, 
                             'report_interval': 250000, 'save_dir': None, 
                             'restore_from': 'mycurl-7500000', 'tb_dir': None,}

bs = run_training_params['dataset_params']['batch_size']
crop_dim = run_training_params['dataset_params']['crop_dim']
train_every = run_training_params['dataset_params']['train_every']
test_every = run_training_params['dataset_params']['test_every']

train_data = pickle.load(open(run_training_params['dataset_params']['path'], 'rb'))

n_px = int(np.sqrt(train_data['train_images'].shape[1]))

train_data['train_images'] = train_data['train_images'].reshape((-1,n_px,n_px))[::train_every,:crop_dim,:crop_dim].reshape((-1,crop_dim**2))
train_data['train_labels'] = train_data['train_labels'][::train_every]
train_data['test_images'] = train_data['test_images'].reshape((-1,n_px,n_px))[::test_every,:crop_dim,:crop_dim].reshape((-1,crop_dim**2))
train_data['test_labels'] = train_data['test_labels'][::test_every]

train_eval_ops, test_eval_ops, sess, params, saver = training.run_training(**run_training_params)

num_of_batches = int(len(train_data['test_images'])/128)
x = np.linspace(0, num_of_batches, num_of_batches)
plot_titles = ['z2_samples_from_x_in', 'z2_mean_from_x_in', 'z2_var_from_x_in', 'z1_sample_from_x_in', 'z1_mean_from_x_in', 'z1_var_from_x_in']

fig, axs = plt.subplots(3, 2, sharey='row')

for num_plot in range(6):
  y = [np.mean(eval.infer_z2_sample_z2_mean_z2_variance_z1_sample_z1_mean_z1_variance(train_eval_ops, sess, params, train_data['test_images'][num_batch*128:(num_batch+1)*128])[num_plot]) for num_batch in range(num_of_batches)]
  if num_plot == 0:
    axs[0,0].plot(x, y, alpha=0.3, marker='o')
    axs[0,0].plot(x, [np.mean(y)]*len(x), linestyle='--')
    axs[0,0].fill_between(x, [np.mean(y)+np.var(y)]*len(x), [np.mean(y)-np.var(y)]*len(x), alpha=0.3)
    axs[0,0].set_title(plot_titles[num_plot])
  elif num_plot == 1:
    axs[1,0].plot(x, y, alpha=0.3, marker='o')
    axs[1,0].plot(x, [np.mean(y)]*len(x), linestyle='--')
    axs[1,0].fill_between(x, [np.mean(y)+np.var(y)]*len(x), [np.mean(y)-np.var(y)]*len(x), alpha=0.3)
    axs[1,0].set_title(plot_titles[num_plot])
  elif num_plot == 2:
    axs[2,0].plot(x, y, alpha=0.3, marker='o')
    axs[2,0].plot(x, [np.mean(y)]*len(x), linestyle='--')
    axs[2,0].fill_between(x, [np.mean(y)+np.var(y)]*len(x), [np.mean(y)-np.var(y)]*len(x), alpha=0.3)
    axs[2,0].set_title(plot_titles[num_plot])
  elif num_plot == 3:
    axs[0,1].plot(x, y, alpha=0.3, marker='o')
    axs[0,1].plot(x, [np.mean(y)]*len(x), linestyle='--')
    axs[0,1].fill_between(x, [np.mean(y)+np.var(y)]*len(x), [np.mean(y)-np.var(y)]*len(x), alpha=0.3)
    axs[0,1].set_title(plot_titles[num_plot])
  elif num_plot == 4:
    axs[1,1].plot(x, y, alpha=0.3, marker='o')
    axs[1,1].plot(x, [np.mean(y)]*len(x), linestyle='--')
    axs[1,1].fill_between(x, [np.mean(y)+np.var(y)]*len(x), [np.mean(y)-np.var(y)]*len(x), alpha=0.3)
    axs[1,1].set_title(plot_titles[num_plot])
  else:
    axs[2,1].plot(x, y, alpha=0.3, marker='o')
    axs[2,1].plot(x, [np.mean(y)]*len(x), linestyle='--')
    axs[2,1].fill_between(x, [np.mean(y)+np.var(y)]*len(x), [np.mean(y)-np.var(y)]*len(x), alpha=0.3)
    axs[2,1].set_title(plot_titles[num_plot])

plt.tight_layout()
plt.savefig('current_plot.png')
