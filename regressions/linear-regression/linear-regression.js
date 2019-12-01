const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];

        this.options = Object.assign(
            { learningRate: 0.0001, iterations: 1000 },
            options
        );

        this.weights = tf.zeros([this.features.shape[1], 1]);
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights); //Creates (mx + b) term
        const differences = currentGuesses.sub(labels); // takes actual value from (mx + b) term

        const gradients = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);
        
        this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
    }

    predict(observations) {
        return this.processFeatures(observations).matMul(this.weights);
    }

    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        const predictions = testFeatures.matMul(this.weights);

        const res = testLabels.sub(predictions)
                    .pow(2)
                    .sum()
                    .get();

        const tot = testLabels.sub(testLabels.mean())
                    .pow(2)
                    .sum()
                    .get();

        return 1 - (res / tot);
    }

    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

        for(let i = 0; i < this.options.iterations; i++) {
            for(let j = 0; j < batchQuantity; j++) {
                const { batchSize } = this.options;
                const startIndex = j * batchSize;

                const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1])
                const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1])
                this.gradientDescent(featureSlice, labelSlice);
            }
            this.recordMSE();
            this.updateLearningRate();
        }
    }

    processFeatures(features) {
        features = tf.tensor(features);

        if(this.mean && this.variance) { 
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardise(features);
        }

        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    standardise(features) {
        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }

    recordMSE() {
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get();

        this.mseHistory.unshift(mse);
    }

    updateLearningRate() {
        if(this.mseHistory.length < 2) {
            return;
        }

        if(this.mseHistory[0] > this.mseHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }

    //GradientDescent without tensor flow

    // gradientDescent() {
    //     const currentGuessesForMPG = this.features.map(row => {
    //         return this.m * row[0] + this.b
    //     });

    //     const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return guess - this.labels[i][0];
    //     })) * 2 / this.features.length;

    //     const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return ((-1 * this.features[i][0]) * (this.labels[i][0] - guess))
    //     })) * 2 / this.features.length;

    //     this.m -= mSlope * this.options.learningRate;
    //     this.b -= bSlope * this.options.learningRate;
    // }
}

module.exports = LinearRegression;  