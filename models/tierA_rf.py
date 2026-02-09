# models/tierA_rf.py
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=300)
clf.fit(X_train, y_train)
