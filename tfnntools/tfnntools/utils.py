import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle

def arg_helper(params, d_param):
    for key in d_param.keys():
        params[key] = params.get(key, d_param[key])
        if type(params[key]) is dict:
            params[key] = arg_helper(params[key], d_param[key])
    return params

def test_resume(try_resume, params):
    """ Try to load the parameters saved in `params['save_dir']+'params.pkl',`

        Not sure we should implement this function that way.
    """
    resume = False

    if try_resume:
        try:
            with open(params['save_dir']+'params.pkl', 'rb') as f:
                params = pickle.load(f)
            resume = True
            print('Resume, the training will start from the last iteration!')
        except:
            print('No resume, the training will start from the beginning!')

    return resume, params


def saferm(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('Erase recursively directory: ' + path)
    if os.path.isfile(path):
        os.remove(path)
        print('Erase file: ' + path)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)