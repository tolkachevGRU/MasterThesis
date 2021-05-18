import os
import KNN
import Plotting
import NeuralNetwork
import PrincipalComponentAnalysis
import NNall
import NNNJ
import Log
import Classes
import SupportVM
import NN
import NB
import RF

if __name__ == '__main__':
    #configure tensorflow
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    #KNN.KNNWithJungle()
    #KNN.KNNWithoutJungle()
    #KNN.KNN10jungle()
    #KNN.KNN10()
    #KNN.KNN15jungle()
    #KNN.KNN15()

    #Log.LogWithJungle()
    #Log.LogWithoutJungle()
    #Log.Log10jungle()
    #Log.Log10()
    #Log.Log15jungle()
    #Log.Log15()


    #SupportVM.SVMWithJungle()
    #SupportVM.SVMwithoutJungle()
    #SupportVM.SVM10jungle()
    #SupportVM.SVM10()
    #SupportVM.SVM15jungle()
    #SupportVM.SVM15()

    #RF.RFjungle()
    #RF.RFwithoutjungle()
    #RF.RF10jungle()
    #RF.RF10()
    #RF.RF15jungle()
    #RF.RF15()

    #NB.NBwithJungle()
    #NB.NBwithoutJungle()
    #NB.NB10jungle()
    #NB.NB10()
    #NB.NB15jungle()
    #NB.NB15()
    
    #NN.NNwithJungle()
    #NN.NNwithoutJungle()
    #NN.NN10jungle()
    #NN.NN10()
    #NN.NN15jungle()
    #NN.NN15()

    #Plotting.plotTest()
    #Plotting.plotWinrate()
    #Plotting.plotGamesPerLeague()
    #Plotting.RF15plot()
    #Plotting.RF15jungleplot()
    Plotting.RFalljungleplot()
    #Plotting.RFallplot()

    #Plotting.plotHistograms()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
