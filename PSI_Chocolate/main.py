import pandas as panda
import numpy
from sklearn import linear_model
from sklearn.metrics import r2_score

def uniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  return output

def linearPredictionPokemon():
    df = panda.read_csv("flavors_of_cacao.csv", sep=',')
    reg = linear_model.LinearRegression()

    data = panda.DataFrame({'q': df.Company})
    data['q'] = data['q'].astype('category')
    data['q'] = data['q'].cat.reorder_categories(uniq(df.Company), ordered=True)
    data['q'] = data['q'].cat.codes
    data2 = panda.DataFrame({'w': df.CompanyLocation})
    data2['w'] = data2['w'].astype('category')
    data2['w'] = data2['w'].cat.reorder_categories(uniq(df.CompanyLocation), ordered=True)
    data2['w'] = data2['w'].cat.codes
    print(data2['w'])

    regData=df[['Rating']]
    print(regData)

    regData = regData.assign(q=data['q'])
    print(regData)
    reg.fit(regData, df.CocoaPercent)
    yPrediction = reg.predict(regData)

    print('Variance score: %.2f' % r2_score(df.CocoaPercent, yPrediction))
    print(reg.coef_)
    print(reg.intercept_)
    print(yPrediction)
    
    numpy.savetxt('predictions.csv', yPrediction)

def main():
    linearPredictionPokemon()


if __name__ == "__main__":
    main()
