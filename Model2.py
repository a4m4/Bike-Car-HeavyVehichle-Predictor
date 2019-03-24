from sklearn import tree

X = [ [50,30000] , [70,45000] , [100,65000] , [110,80000] , [125,100000] , [150,128000] ,
      [700, 200000] , [ 800, 300000] , [1000, 500000] , [1000,700000] , [ 1300, 900000] , [1300, 1300000],
      [1300,180000] , [1600,2000000] , [1700, 2400000] , [3500,1700000], [4000,2500000] , [5000,4000000] ,
      [7000,7000000] ]

Y = [ 'Bike' , 'Bike' , 'Bike' , 'Bike' , 'Bike' , 'Bike' , 'Car' , 'Car' , 'Car' , 'Car' , 'Car' , 'Car' , 'Car' ,
     'Car' , 'Car' , 'Heavy Vehichle' , 'Heavy Vehichle' , 'Heavy Vehichle' , 'Heavy Vehichle'  ]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

pred = clf.predict([[1300,500000]])

print(pred)