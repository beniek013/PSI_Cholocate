import pandas as panda
from sklearn import linear_model
from sklearn.metrics import r2_score

def linearChocolatePrediction():
    df = panda.read_csv("flavors_of_cacao.csv", sep=',')
    reg = linear_model.LinearRegression()
    reg.fit(df[['ReviewDate', 'Percent']], df.Rating)

    yPrediction = reg.predict(df[['ReviewDate', 'Percent']])

    print('Variance score: %.2f' % r2_score(df.Rating, yPrediction))
    print(reg.coef)
    print(reg.intercept_)
    print(yPrediction)

def main():
    linearChocolatePrediction()


if __name__ == "__main__":
    main()
