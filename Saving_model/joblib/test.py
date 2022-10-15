# Swatej Joblist test code
import joblib
model=joblib.load("diabetes_79.pkl")
result=model.predict([[0,1,1,1,1,1,1,0]])

if result[0]==1:
    print("diabetic")
else:
    print("Not diabetic")