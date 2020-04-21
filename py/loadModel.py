import os
from keras.models import load_model
import pandas
from metrics import error_metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" loads a model and predicts in the testing dataset
:param data an instance of utils.Data
:param outputDir where the results are stored
:param args are the arguments used to produce the results
"""

def loadModel(data, outputDir, args):
    tstX, tstY = data.get_m_tst() if args.input == "all" else data.get_tst(args.input)

    # for each fold
    for f in range(5):
        filename = "bestModel_fold" + str(f + 1) + ".hdf5"
        print('Loading file: ' + filename + ' ...')
        model = load_model(os.path.join(outputDir, filename))

        pred_norm = model.predict(tstX)

        if data.multipleOutput:
            properties = data.properties
            pred_norm = pred_norm.transpose()
            tstYtransp = tstY.transpose()
            res = {}
            for i in range(len(properties)):
                pred = data.unnormalizeOutput(pred_norm[i], properties[i], args.maxNorm)
                y_unnorm = data.unnormalizeOutput(tstYtransp[i], properties[i], args.maxNorm)
                # Compute errors
                print(properties[i], error_metrics(y_unnorm, pred))
                res[properties[i] + "-measured"] = y_unnorm
                res[properties[i] + "-predicted"] = pred
            df = pandas.DataFrame.from_dict(res)

        else:
            pred = data.unnormalizeOutput(pred_norm, args.output, args.maxNorm)
            pred = pred.reshape((pred.shape[0],))
            y_unnorm = data.unnormalizeOutput(tstY, args.output, args.maxNorm)
            # Compute errors
            print(error_metrics(y_unnorm, pred))
            df = pandas.DataFrame({
                args.output + "-measured": y_unnorm,
                args.output + "-predicted": pred
            })
        df.to_csv(os.path.join(outputDir, "results-fold-{0}.csv".format(f+1)), index=False)
