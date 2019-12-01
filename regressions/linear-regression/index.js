require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression.js');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, { 
    learningRate: 0.1,
    iterations: 100,
    batchSize: 10 
});

regression.train();

console.log('Percentage correct: ', regression.test(testFeatures, testLabels) * 100, '%');

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration Number',
    yLabel: 'MSE (Mean Squared Error)'
});

regression.predict([
    [115, 1.4, 100]
]).print();
