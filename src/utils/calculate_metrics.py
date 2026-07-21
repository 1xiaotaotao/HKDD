import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef

def calculate_metrics(y, y_pred, y_pred_proba):
	acc = accuracy_score(y, y_pred)
	auc = roc_auc_score(y, y_pred_proba)
	
	tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
	sn = tp / (tp + fn)
	sp = tn / (tn + fp)
	precision = tp / (tp + fp)
	f1 = 2 * precision * sn / (precision + sn)
	mcc = matthews_corrcoef(y, y_pred)
	return {
		'accuracy': acc,
		'sn(recall)': sn,
		'sp': sp,
		'mcc': mcc,
		'auc': auc,
		'precision': precision,
		'f1_score': f1
	}

def summarize_results(results):
	summary = {}
	metrics_names = ['accuracy', 'sn(recall)', 'sp', 'mcc', 'auc', 'precision', 'f1_score']
	for metric in metrics_names:
		values = [result[metric] for result in results]
		mean_val = np.mean(values)
		summary[metric] = mean_val
	return summary