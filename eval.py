import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def grey_image(img, unitscale=True, ylabel=None, fontsize=26,
               border_color=None):
    if img.ndim == 1:
        n_px = int(np.sqrt(img.shape[0]))
        img = img.reshape(n_px, n_px)

    if not isinstance(unitscale, bool):
        vlimits = {'vmin': -unitscale, 'vmax': unitscale}
    elif unitscale:
        vlimits = {'vmin': 0, 'vmax': 1}
    else:
        vlimits = {}

    plt.imshow(img, interpolation='none', cmap=plt.get_cmap('gray'), **vlimits)

    if border_color is not None:
        for spine in ['bottom', 'top', 'left', 'right']:
            plt.gca().spines[spine].set_color(border_color)
            plt.gca().spines[spine].set_linewidth(4)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)


def image_set(images, plotter_func=grey_image, samples_per_row=20,
              unitscale=False, file_name=None):
    samples_per_row = np.minimum(samples_per_row, images.shape[0])
    n_rows = np.ceil(images.shape[0] / samples_per_row)
    f = plt.figure(figsize=(3 * samples_per_row, 3 * n_rows))
    f.patch.set_alpha(1.0)
    for i in range(images.shape[0]):
        plt.subplot(n_rows, samples_per_row, i + 1)
        grey_image(images[i, ], unitscale)
    if file_name is not None:
        f.savefig(file_name, bbox_inches='tight')


def latent_traversal(eval_ops, sess, params, latents, rest='random',
                     n_batch=1, max_val=None, zs=None):
    data_dim = params['dataset_params']['crop_dim']**2
    batch_size = eval_ops.x_in.shape.as_list()[0]
    latent_dim = params['n_z']
    z2_prior_samples = sess.run(eval_ops.z2_prior_samples)

    if max_val is None:
        max_val = zs['samples'].max() if zs is not None else 1

    rfs = np.zeros((len(latents), data_dim))
    for latent in range(len(latents)):
        diff = np.zeros((0, data_dim))
        for b in range(n_batch):
            if rest == 'random':
                activations = \
                    sess.run(eval_ops.z1_samples_from_z2_prior_samples)
            else:
                activations = np.zeros((batch_size, latent_dim))
            activations[:, latents[latent]] = -max_val
            x_mean_low = sess.run(eval_ops.x_mean_from_z1_in_z2_in,
                                  feed_dict={eval_ops.z1_in: activations,
                                             eval_ops.z2_in: z2_prior_samples})
            x_mean_low = x_mean_low.reshape((batch_size, data_dim))
            activations[:, latents[latent]] = max_val
            x_mean_high = sess.run(eval_ops.x_mean_from_z1_in_z2_in,
                                   feed_dict={eval_ops.z1_in: activations,
                                              eval_ops.z2_in: z2_prior_samples})
            x_mean_high = x_mean_high.reshape((batch_size, data_dim))
            diff = np.vstack((diff, x_mean_high - x_mean_low))
        rfs[latent, :] = np.mean(diff, axis=0)
    return np.squeeze(rfs)


def reconstruct(eval_ops, sess, params, test_data, use_mean=True):
    n_px = params['dataset_params']['crop_dim']
    xdim = n_px * n_px

    if len(test_data.shape) == 2:
        if test_data.shape[1] == xdim:
            test_data = test_data.reshape((-1, n_px, n_px, 1))
        else:
            raise ValueError('test_data has wrong dimensions')

    n_samples = test_data.shape[0]
    batch_size = params['dataset_params']['batch_size']
    n_batches, n_remainder = divmod(n_samples, batch_size)
    reconstr = np.zeros_like(test_data)
    eval_op = eval_ops.x_mean_from_x_in if use_mean \
        else eval_ops.x_sample_from_x_in

    for i_batch in range(n_batches):
        i_from = i_batch * batch_size
        reconstr[i_from:i_from + batch_size, :, :, :] = \
            sess.run(eval_op,
                     feed_dict={eval_ops.x_in:
                                test_data[i_from:i_from + batch_size, :, :, :]})

    if n_remainder:
        last_batch = np.zeros((batch_size,) + test_data.shape[1:],
                              dtype=test_data.dtype)
        i_from = n_batches * batch_size
        last_batch[:n_remainder, :, :, :] = test_data[i_from:, :, :, :]
        last_batch = sess.run(eval_op,
                              feed_dict={eval_ops.x_in: last_batch})
        reconstr[i_from:, :, :, :] = last_batch[:n_remainder, :, :, :]

    return reconstr.reshape((-1, xdim))


def infer_z2_sample_z2_mean_z2_variance_z1_sample_z1_mean_z1_variance(eval_ops,
                                                                      sess,
                                                                      params,
                                                                      test_data
                                                                      ):
    n_px = params['dataset_params']['crop_dim']
    xdim = n_px * n_px

    if len(test_data.shape) == 2:
        if test_data.shape[1] == xdim:
            test_data = test_data.reshape((-1, n_px, n_px, 1))
        else:
            raise ValueError('test_data has wrong dimensions')

    return sess.run((eval_ops.z2_samples_from_x_in,
                     eval_ops.z2_mean_from_x_in,
                     eval_ops.z2_variance_from_x_in,
                     eval_ops.z1_sample_from_x_in,
                     eval_ops.z1_mean_from_x_in,
                     eval_ops.z1_variance_from_x_in),
                    feed_dict={eval_ops.x_in: test_data})


def z1_mean_z1_variance(eval_ops, sess, params, test_x, test_z2):
    n_px = params['dataset_params']['crop_dim']
    xdim = n_px * n_px

    if len(test_x.shape) == 2:
        if test_x.shape[1] == xdim:
            test_x = test_x.reshape((-1, n_px, n_px, 1))
        else:
            raise ValueError('test_data has wrong dimensions')

    return sess.run((eval_ops.z1_mean_from_x_in_z2_in,
                     eval_ops.z1_variance_from_x_in_z2_in),
                    feed_dict={eval_ops.x_in: test_x,
                               eval_ops.z2_in: test_z2})


def generate_z1_mean_z1_variance(eval_ops, sess, params, test_data):
    return sess.run((eval_ops.z1_mean_from_z2_in,
                     eval_ops.z1_variance_from_z2_in),
                    feed_dict={eval_ops.z2_in: test_data})


def generate_x(eval_ops, sess, params, mean=True, z2=None):
    n_px = params['dataset_params']['crop_dim']
    xdim = n_px * n_px

    if z2 is None:
        eval_op = eval_ops.x_mean_generated if mean \
            else eval_ops.x_sample_generated
        return sess.run(eval_op).reshape((-1, xdim))
    else:
        eval_op = eval_ops.x_mean_generated_from_z2_in if mean \
            else eval_ops.x_sample_generated_from_z2_in
        return sess.run(eval_op,
                        feed_dict={eval_ops.z2_in: z2}).reshape((-1, xdim))


def generate_x_from_z1(eval_ops, sess, params, z1_in, mean=True):
    n_px = params['dataset_params']['crop_dim']
    xdim = n_px * n_px

    eval_op = eval_ops.x_mean_from_z1_in_z2_in if mean \
        else eval_ops.x_sample_from_z1_in_z2_in

    return sess.run(eval_op,
                    feed_dict={eval_ops.z1_in: z1_in}).reshape((-1, xdim))


def print_num_trainable_params(verbose=True):
    # https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
    if verbose:
        print('trainable variables:')
    num_params_total = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params_var = np.prod(shape)
        if verbose:
            print(' ', variable.name, shape, num_params_var)
        num_params_total += num_params_var
    print('total number of trainable parameters:', num_params_total)
