from sklearn.metrics import classification_report

def print_report(y_true, y_pred):
    target_names = ['class_0', 'class_1']
    classification_report(y_true, y_pred, target_names=target_names)
