import os
import numpy as np
import tensorflow as tf

from shutil import copy


def get_ckpts_number(save_dir):
    # Getting all the checkpoints
    ckpts = [p[:-5] for p in os.listdir(save_dir) if p[-5:] == '.meta']
    ckpts_num = [int(p[p.rfind('-') + 1:]) for p in ckpts]
    ckpts_num_set = set(ckpts_num)
    return ckpts_num_set


def get_event_data(event_files, tag, selec_it=None):
    data = []
    it = []
    if selec_it:
        selec_it = set(selec_it)
    for path_event in event_files:
        try:
            for e in tf.train.summary_iterator(path_event):
                for v in e.summary.value:
                    if tag in v.tag:
                        if selec_it is None or e.step in selec_it:
                            data.append(v.simple_value)
                            it.append(e.step)
        except:
            print('Warning corrupted file')
    return np.array(data), np.array(it)


def get_event_files(summary_dir):
    # Getting the event file
    event_files = []
    for filename in os.listdir(summary_dir):
        if 'events.out.tfevents' in filename:
            event_files.append(os.path.join(summary_dir, filename))
    # if len(event_files)>1:
    #     raise ValueError('Multiple event files')
    if len(event_files) == 0:
        raise ValueError('No event files')
    return event_files


def select_ouputs(save_dir, summary_dir, output_dir):
    ckpts_num_set = get_ckpts_number(save_dir)
    event_files = get_event_files(summary_dir)

    # Getting the PSD data
    log_l2_psd, it = get_event_data(
        event_files, 'PSD/log_l2', selec_it=ckpts_num_set)
    log_l2_hist, it = get_event_data(
        event_files, 'MASS_HIST/log_l2', selec_it=ckpts_num_set)
    log_l2_peak, it = get_event_data(
        event_files, 'PEAK_HIST/log_l2', selec_it=ckpts_num_set)
    assert (len(log_l2_psd) == len(log_l2_hist) == len(log_l2_peak))

    # Select a few experiment
    to_keep = list()
    to_keep.extend(it[np.argpartition(log_l2_psd, 3)[:3]])
    to_keep.append(it[np.argmin(log_l2_hist)])
    to_keep.append(it[np.argmin(log_l2_peak)])
    to_keep = set(to_keep)

    # Copy the data
    os.makedirs(output_dir + 'checkpoints', exist_ok=True)
    os.makedirs(output_dir + 'summary', exist_ok=True)
    list_copy = list()
    print('Copy summary')
    for event_file in event_files:
        copy(event_file, output_dir + 'summary/')
        list_copy.append(event_file)
    print('Copy checkpoints')
    for index in to_keep:
        for file in os.listdir(save_dir):
            if str(index) in file:
                copy(save_dir + file, output_dir + 'checkpoints/')
                list_copy.append(save_dir + file)

    print('Copy params')
    copy(save_dir + 'params.pkl', output_dir + 'checkpoints/')
    list_copy.append(save_dir + 'params.pkl')

    return list_copy
