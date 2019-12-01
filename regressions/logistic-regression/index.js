require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (value) => value === "TRUE" ? 1 : 0
    }
});

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 10,
    decisionBoundary: 0.5
});

regression.train();

console.log("Percentage correct: ", regression.test(testFeatures, testLabels) * 100, '%');

plot({
    x: regression.costHistory.reverse(),
    xLabel: 'Number of iterations',
    yLabel: 'Cost Value'
})