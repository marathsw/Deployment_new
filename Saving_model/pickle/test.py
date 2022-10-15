# Swatej Joblist test code
import pickle
model=pickle.load(open('diabetes_79.pkl','rb'))
result=model.predict([[0,1,1,1,1,1,1,0]])

if result[0]==1:
    print("diabetic")
else:
    print("Not diabetic")