const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.costHistory = [];

        this.options = Object.assign(
            { learningRate: 0.0001, iterations: 1000, decisionBoundary: 0.5 },
            options
        );

        this.weights = tf.zeros([this.features.shape[1], 1]);
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).sigmoid(); //Creates (mx + b) term and takes the sigmoid (1/1+e^-(z) where z is the (mx+b) term)
        const differences = currentGuesses.sub(labels); // takes actual value from (mx + b) term

        const gradients = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);
        
        this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
    }

    predict(observations) {
        return this.processFeatures(observations)
            .matMul(this.weights)
            .sigmoid()
            .greaterEqual(this.options.decisionBoundary)
            .cast('float32');
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures); // only works if the descision boundary is at 0.5
        testLabels = tf.tensor(testLabels);

        const incorrect = predictions.sub(testLabels)
            .abs()
            .sum()
            .get();
    
        return (predictions.shape[0] - incorrect) / (predictions.shape[0])
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
            this.recordCost();
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

    recordCost() {
        const guesses = this.features
            .matMul(this.weights)
            .sigmoid()

        const termOne = this.labels
            .transpose()
            .matMul(guesses.log())

        const termTwo = this.labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(
                guesses
                    .mul(-1)
                    .add(1)
                    .log()
            );


        const cost = termOne.add(termTwo)
            .div(this.features.shape[0])
            .mul(-1)
            .get(0, 0);

        this.costHistory.unshift(cost);
    }

    updateLearningRate() {
        if(this.costHistory.length < 2) {
            return;
        }

        if(this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }
}

module.exports = LogisticRegression;  