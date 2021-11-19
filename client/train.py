from __future__ import print_function
import sys
import tensorflow as tf
from read_data import read_data
import yaml

from ttictoc import tic,toc
import threading
import psutil
from datetime import datetime

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def ps_util_monitor(round):
    global running
    running = True
    currentProcess = psutil.Process()
    cpu_ = []
    memo_ = []
    time_ = []
    report = {}
    # start loop
    while running:
        cpu_percents = currentProcess.cpu_percent(interval=1)
        mem_percents = currentProcess.memory_percent()
        ps_time = str(datetime.now())
        cpu_.append(cpu_percents)
        memo_.append(mem_percents)
        time_.append(ps_time)

    report['round'] = round
    report['cpu'] = cpu_
    report['memory'] = memo_
    report['time'] = time_

    with open('/app/resources.txt', '+a') as f:
        print(report, file=f)
    # with open('/app/resources.txt', '+a')as fh:
    #     fh.write(json.dumps(report))




def start_monitor(round):
    global t
    # create thread and start it
    t = threading.Thread(target=ps_util_monitor, args=[round])
    t.start()


def stop_monitor():
    global running
    global t
    # use `running` to stop loop in thread so thread will end
    running = False
    # wait for thread's end
    t.join()


def train(model, data, settings):
    """
    Helper function to train the model
    :return: model
    """
    print("-- RUNNING TRAINING --")

    x_train, y_train = read_data(data)

    print(" --------------------------------------- ")
    print("x_train shape: : ", x_train.shape)
    print(" --------------------------------------- ")

    start_monitor(round)
    tic()

    model.fit(x_train, y_train, epochs=settings['epochs'], batch_size=settings['batch_size'], verbose=True)

    elapsed = toc()

    stop_monitor()

    with open('/app/time.txt', '+a') as f:
        print(elapsed, file=f)


    print("-- TRAINING COMPLETED --")
    return model


if __name__ == '__main__':

    from fedn.utils.kerashelper import KerasHelper
    from models.imdb_model import create_seed_model

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model(trainedLayers=settings['trained_Layers'])
    model.set_weights(weights)
    model = train(model, '../data/train.csv', settings)
    helper.save_model(model.get_weights(), sys.argv[2])