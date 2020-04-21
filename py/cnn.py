import math
import os
import argparse
import json
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LeakyReLU, Dense, Flatten
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import regularizers
from utils import Data, InputStandardization
from loadModel import loadModel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convolutional Neural Networks of doi: 10.1016/j.geoderma.2020.114208')
    # Positional and required
    parser.add_argument('-d', '--directory', type=str,
                        help='Select directory to store model and results', required=True)
    # Named and optional
    properties = ['all', 'OC', 'clay', 'silt', 'sand', 'N', 'P', 'K', 'pH.in.H2O', 'CEC', 'CaCO3']
    sources = ['all', 'Absorbances', 'Absorbances-SG1', 'Absorbances-SG1-SNV', 'Absorbances-SNV-DT', 'Absorbances-SG0_SNV', 'CR']
    parser.add_argument('-i', '--input', choices=sources,
                        help='Select input source.  Allowed values are: ' + ', '.join(sources), metavar='')
    parser.add_argument('-o', '--output', choices=properties,
                        help='Select output property.  Allowed values are: ' + ', '.join(properties), metavar='')
    parser.add_argument('-s', '--standardize', help='Select standardization type',
                        type=InputStandardization, choices=list(InputStandardization), default=InputStandardization.NONE)
    parser.add_argument('-f', '--filters', nargs='+', type=int,
                        help='Select number of convolutional filters', default=[24, 48])
    parser.add_argument('-k', '--kernelSize', type=int,
                        help='Select kernel size', default=7)
    parser.add_argument('-b', '--batchSize', type=int,
                        help='Select batch size', default=10)
    parser.add_argument('-r', '--regularizationSize', type=float,
                        help='Select regularization size', default=0.0004)
    parser.add_argument('-fc', '--fully', nargs='+', type=int,
                        help='Select number of dense filters', default=[16, 6])
    parser.add_argument('-lr', '--lr', type=int,
                        help='Select learning rate update interval', default=60)
    parser.add_argument('-a', '--activation', help='Select activation function',
                        choices=['tanh', 'linear'], default='tanh')
    parser.add_argument('-m', '--maxNorm', type=float,
                        help='Select regularization size', default=1)
    return parser.parse_args()
        


def step_decay(epoch):
    init = 0.0001
    drop = 0.5
    max_epoch = 3 * lr
    if epoch > max_epoch:
        return init * math.pow(drop, math.floor(max_epoch/lr))
    else:
        return init * math.pow(drop, math.floor(epoch/lr))


def convUnit(net, filNumber, kernelSize, pool=False):
    out = Conv1D(filters=filNumber, kernel_size=kernelSize, strides=1, padding="same")(net)
    out = BatchNormalization()(out)
    out = LeakyReLU(alpha=.01)(out)

    if pool:
        out = MaxPooling1D(pool_size=2)(out)
    return out


def get_model(filtSizes, kernelSize, regSize, fullyCon, finalActivation, spectral_input, output):
    input_shape = (200, 6) if spectral_input == "all" else (200, 1)
    spectral = Input(input_shape)

    net = convUnit(spectral, filtSizes[0], kernelSize, pool=True)
    net = convUnit(net, filtSizes[1], kernelSize)

    net = Flatten()(net)

    net = Dense(units=fullyCon[0], kernel_regularizer=regularizers.l2(regSize))(net)
    net = LeakyReLU(alpha=.01)(net)

    net = Dense(units=fullyCon[1], kernel_regularizer=regularizers.l2(regSize))(net)
    net = LeakyReLU(alpha=.01)(net)

    out_units = 10 if output == "all" else 1 
    net = Dense(units=out_units, activation=finalActivation)(net)

    model = Model(inputs=spectral, outputs=net)
    model.compile(optimizer=Adam(lr=0.0001), loss='mse')
    return model


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    lr = args.lr
    data = Data(args.standardize, args.output, args.maxNorm)
    
    dirpath = os.path.join(args.directory, args.input, args.output)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(os.path.join(dirpath, "args.json"), "w") as outfile:
        json.dump(vars(args), outfile)

    for f in range(5):
        print("Training fold " + str(f + 1))
        if args.input != "all":
            trnX, trnY, valX, valY = data.get_trn_val(f + 1, key=args.input)
        else:
            trnX, trnY, valX, valY = data.get_m_trn_val(f + 1)

        testingModel = get_model(
            args.filters, args.kernelSize, args.regularizationSize, args.fully,
            args.activation, args.input, args.output)

        if f == 0:
            testingModel.summary()
        early_stop = EarlyStopping(
            monitor='val_loss', patience=80, mode='min', restore_best_weights=True)
        lrate = LearningRateScheduler(step_decay, verbose=0)
        callbacks_list = [early_stop, lrate]
        history = testingModel.fit(
            trnX, trnY, epochs=2000, batch_size=args.batchSize,
            validation_data=(valX, valY), callbacks=callbacks_list, verbose=2)

        testingModel.save(os.path.join(
            dirpath, "bestModel_fold{0}.hdf5".format(f + 1)))
    print('Finished Training')
    loadModel(data, dirpath, args)
