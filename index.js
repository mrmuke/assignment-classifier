const tf = require('@tensorflow/tfjs')
//require('@tensorflow/tfjs-node')
const use = require('@tensorflow-models/universal-sentence-encoder')
const training_data = require('./training-data.json');
const testing_data = require('./testing-data.json');

function encodeData(data){
    const sentences = data.map(work=>work.text)
    const results = use.load().then(model => {
        return model.embed(sentences).then(embeddings => {
          return embeddings
        });
      });
    return results
}
async function main(){
    const training = await encodeData(training_data)
    const testing = await encodeData(testing_data)

    model.fit(training, outputData, { epochs: 200 })
        .then(() => {
            model.predict(testing).print();
        });
};
const outputData = tf.tensor2d(training_data.map(work => [
  work.intent === 'homework' ? 1 : 0,
  work.intent === 'classwork' ? 1 : 0,
]));
outputData.print()
const model = tf.sequential();

// Add layers to the model
model.add(tf.layers.dense({
    inputShape: [512],
    activation: 'sigmoid',
    units: 2,
}));

model.add(tf.layers.dense({
    inputShape: [2],
    activation: 'sigmoid',
    units: 2,
}))

model.add(tf.layers.dense({ 
    inputShape: [2],
    activation: 'sigmoid',
    units: 2,
}))

// Compile the model
model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(.06), 
});

main()

