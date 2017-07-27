import numpy as np


def loadSimpleData():

    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [ 1., 1.],
                         [ 2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    Single layer decision Tree
    :param dataMatrix:
    :param dimen:
    :param threshVal:
    :param threshIneq:
    :return:
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))

    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <=threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] >=threshVal] = -1.0
    return retArray


def buildStump(dataArr,classLabels,D, output=False):
    """
    Single layer decision tree(weak classifier)
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf #init error sum, to +infinity
    for i in range(n): #loop over all dimensions
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#call stump classify with i, j, lessThan
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                if output:
                    print "split: dim %d, " \
                      "thresh %.2f, " \
                      "thresh ineqal: %s, " \
                      "the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    AdaBoost building decision tree
    :param dataArr:
    :param classLabels:
    :param numIt:
    :return:
    """
    weakClassArr = []
    m = dataArr.shape[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "iteration:", i, " D:", D.T
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ", classEst.T
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha*classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate, "\n"

        if errorRate == 0.0 : break
    return weakClassArr, aggClassEst


def adaClassify(dataToClass, classifierArr):
    dataMatrix = np.mat(dataToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return np.sign(aggClassEst)


### training a dataset
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))  # get number of fields
    dataMat = [];
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep


if __name__ == '__main__':
    dataMat, classLabels = loadSimpleData()
    D = np.mat(np.ones((5, 1)) / 5)
    bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)

    """
    print '===============SINGLE DT =================='
    print '-------- best result : \n'
    print 'bestDim = ', bestStump['dim'], ' bestThresh = ', bestStump['thresh'], ' inqe = ', bestStump['ineq']
    print 'minError = ', minError
    print 'bestClasEst = ', bestClasEst
    """

    print '==========ADA BOOST DTS ==================='
    classifierArray,_ = adaBoostTrainDS(dataMat, classLabels, 9)

    predict = adaClassify([[5.0, 5.0],[0.0, 0.0]], classifierArray)
    print predict

    print '=========ADA BOOST ON DATASET ============='
    trainArr, trainLabelArr = loadDataSet('horseColicTraining2.txt')
    classifierArrayN, _ = adaBoostTrainDS(np.mat(trainArr), trainLabelArr, 50)

    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction = adaClassify(testArr, classifierArrayN)
    errArr = np.mat(np.ones((67, 1)))
    print 'error count = ', errArr[prediction!=np.mat(testLabelArr).T].sum()

    # plot AUC
    plotROC(_.T, trainLabelArr)
