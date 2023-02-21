import numpy as np
import matplotlib.pyplot as plt
import os




def confusion_Matrix(y_true, y_pred, plot = True):
    num_classes = len(np.unique(y_true))
    '''Generate a confusion matrix given a set of labels and predictions.'''
    ## Generate the confusion matrix
    C = np.zeros((num_classes,num_classes))
    for cx in range(num_classes):
        preds_for_this_label = y_pred[y_true==cx]
        for cy in range(num_classes):
            pred_label_pairs = preds_for_this_label==cy
            C[cx,cy] = np.sum(pred_label_pairs)

    if plot:
        ## Plot the confusion matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        res = ax.imshow(C, cmap="Reds")
        ct = C.max()/2
        for x in range(num_classes):
            for y in range(num_classes):
                if C[x,y] > ct: color=[1,1,1]
                else: color=[0,0,0]
                ax.annotate(str(int(C[x,y])), xy=(y,x),
                            horizontalalignment="center",
                            verticalalignment="center",
                            color=color)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        fig.colorbar(res, ax=ax)
        return C,fig
    else:
        return C


def accuracy(y_true, y_pred):
    '''
    Calculates the accuracy of a model
    y_true: true predictions 
    y_pred: Calculated predictions 
    '''
    return np.where(y_true == y_pred, 1, 0).sum() / len(y_true)


def recall(y_true, y_pred):
    '''
    Calculates recall for a binary classification problem
    y_true: True labels 
    y_pred: Predicted labels 
    ---
    Returns the recall.
    '''
    cm = confusion_Matrix(y_true, y_pred, False)
    
    return cm[0][0] / (cm[0][0] + cm[0][1])

def precission(y_true, y_pred):
    '''
    Calculates recall for a binary classification problem
    y_true: True labels 
    y_pred: Predicted labels 
    ---
    Returns the recall.
    '''
    cm = confusion_Matrix(y_true, y_pred, False)
    return cm[0][0] / (cm[0][0] + cm[1][0])


    

    
    




def save_declared_figure(file_name: str, task_number: int = None) -> 0:
    """
    Saves a figure to a set file path

    task_number: the task number
    file_name: name to be saved as 
    format: handles the axises for spatial or fourier domain
    v_min_max: if the intensity colorbar should be set in a fixed range of 0,255 or not
    
    ---
    saves figure that to the a path result_images/task_[task_number] 
    """

    #getting working directory and making path
    wd = os.getcwd()
    if task_number is None:
        directory_savepath = os.path.join(wd, "result_images")
    else:
        directory_savepath = os.path.join(wd, "result_images", "task_" + str(task_number))

    #checking if path exist if not makes it
    if os.path.exists(directory_savepath) == False:
        os.makedirs(directory_savepath)

    #saving image
    image_savepath = os.path.join(directory_savepath, file_name + ".png")
    plt.savefig(image_savepath, format = "png")
    plt.close()


    return 0





def plot_ROC_curve(true_labels, soft_estimated_labels):

    P = np.sum(true_labels)
    F = len(true_labels) - P

    tp_rate = np.zeros(1001)
    fp_rate = np.zeros(1001)
    print(P, F)
    for i, threshold in enumerate(np.linspace(0,1,1001)):
        
        hard_estimated_labels = np.where(soft_estimated_labels < threshold, 0, 1)

        cm = confusion_Matrix(true_labels, hard_estimated_labels, False)


        print(cm[0][0] + cm[1][0], (cm[1][1] + cm[0][1]))

        
        tp_rate[i] = cm[0][0] / (cm[0][0] + cm[0][1])

        fp_rate[i] = cm[1][0] / (cm[1][1] + cm[1][0])
        
        print(tp_rate[i], fp_rate[i])

    plt.plot(fp_rate, tp_rate)
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')

    return tp_rate, fp_rate


def get_AUC(tp_rate, fp_rate):
    au_curve = np.trapz(tp_rate ,fp_rate)
    
    c = np.cumsum(tp_rate - fp_rate)
    print(c)
    
    print(f'au score {au_curve} (trapz method)')