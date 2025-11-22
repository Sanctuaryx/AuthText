from sklearn.calibration import CalibratedClassifierCV

def calibrate_if_needed(clf, method='sigmoid', cv=3, enabled=True):
    return CalibratedClassifierCV(clf, method=method, cv=cv) if enabled else clf
