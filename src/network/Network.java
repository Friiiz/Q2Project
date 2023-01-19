package network;

import filehandling.FileHandler;

import java.util.*;

public class Network {
    private final double LEARNING_RATE;
    private final int BATCH_SIZE;
    private final Neuron[][] LAYERS;

    public Network(double learningRate, int batchSize, int inputLayerSize, int outputLayerSize, int... hiddenLayerSizes) {
        LEARNING_RATE = learningRate;
        BATCH_SIZE = batchSize;
        LAYERS = new Neuron[1 + hiddenLayerSizes.length + 1][];
        for (int l = 0; l < LAYERS.length; l++) {
            if (l == 0) {
                //creating input layer
                Neuron[] inputLayerNeurons = new Neuron[inputLayerSize];
                Arrays.setAll(inputLayerNeurons, node -> new Neuron(Neuron.NodeType.INPUT, null, BATCH_SIZE));
                LAYERS[l] = inputLayerNeurons;
                System.out.println("Input layer created with " + inputLayerSize + " nodes.");
                continue;
            }
            if (l == LAYERS.length - 1) {
                //creating output layer
                Neuron[] outputLayerNeurons = new Neuron[outputLayerSize];
                int nodeIndex = 0;
                //adding nodes for numbers
                for (int i = 48; i <= 57; i++) {
                    outputLayerNeurons[nodeIndex] = new Neuron(Neuron.NodeType.OUTPUT, (char) i, BATCH_SIZE);
                    nodeIndex++;
                }

                //adding nodes for uppercase letters
                for (int i = 65; i <= 90; i++) {
                    outputLayerNeurons[nodeIndex] = new Neuron(Neuron.NodeType.OUTPUT, (char) i, BATCH_SIZE);
                    nodeIndex++;
                }

                //adding nodes for lowercase letters
                for (int i = 97; i <= 122; i++) {
                    outputLayerNeurons[nodeIndex] = new Neuron(Neuron.NodeType.OUTPUT, (char) i, BATCH_SIZE);
                    nodeIndex++;
                }

                LAYERS[LAYERS.length - 1] = outputLayerNeurons;
                System.out.println("Output layer created with " + outputLayerSize + " nodes.");
                continue;
            }

            //creating hidden layers
            Neuron[] hiddenLayerNeurons = new Neuron[hiddenLayerSizes[l - 1]];
            Arrays.setAll(hiddenLayerNeurons, node -> new Neuron(Neuron.NodeType.HIDDEN, null, BATCH_SIZE));
            LAYERS[l] = hiddenLayerNeurons;
            System.out.println("Hidden layer created with " + hiddenLayerSizes[l - 1] + " nodes.");
        }

        //connecting nodes
        Random random = new Random();
        for (int l = 1; l < LAYERS.length; l++) {
            Neuron[] layer = LAYERS[l];
            for (Neuron neuron : layer) {
                Neuron[] previousLayer = LAYERS[l - 1];
                for (Neuron previousNeuron : previousLayer) {
                    neuron.addInputNode(previousNeuron, random.nextGaussian(0, Math.sqrt(1.0d / previousLayer.length)));
                }
            }
        }
        System.out.println("All nodes connected.");
    }

    public LinkedHashMap<double[], Character> shuffleTrainingData(LinkedHashMap<double[], Character> trainingData) {
        //shuffle data
        System.out.println("Shuffling data...");

        LinkedList<Map.Entry<double[], Character>> tempList = new LinkedList<>(trainingData.entrySet());
        trainingData.clear();
        Collections.shuffle(tempList);

        for (Map.Entry<double[], Character> trainingPair : tempList) {
            trainingData.put(trainingPair.getKey(), trainingPair.getValue());
        }

        System.out.println("Data shuffled!");

        return trainingData;
    }

    public void train() {
        FileHandler fileHandler = new FileHandler();
        try {
            fileHandler.loadFiles();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        LinkedHashMap<double[], Character> trainingData = fileHandler.getTrainingData();
        System.out.println("Training network.");
        int totalPairs = 0;
        int successfulPairs = 0;
        double successRate;

        //looping through epochs
        for (int i = 0; i < 10; i++) {
            LinkedHashMap<double[], Character> shuffledTrainingData = shuffleTrainingData(trainingData);
            //looping through shuffled training data
            for (Map.Entry<double[], Character> trainingPair : shuffledTrainingData.entrySet()) {
                //System.out.println("Evaluating image of character '" + trainingPair.getValue() + "'");
                //computing output for each pair
                compute(trainingPair.getKey(), trainingPair.getValue());

                //finding the highest value in output layer
                Neuron maxValue = Arrays.stream(LAYERS[LAYERS.length - 1]).max(Comparator.comparing(Neuron::getActivation)).orElseThrow();

                //track success
                totalPairs++;
                if (maxValue.getNodeLabel() == trainingPair.getValue()) {
                    successfulPairs++;
                }

                //adjusting parameters after every batch
                if (totalPairs % BATCH_SIZE == 0) {
                    System.out.println("Adjusting parameters for batch " + totalPairs / BATCH_SIZE + " in epoch " + (i + 1));
                    for (Neuron[] layer : LAYERS) {
                        for (Neuron neuron : layer) {
                            //nudge parameters
                            neuron.nudgeWeights(LEARNING_RATE);
                            neuron.nudgeBias(LEARNING_RATE);
                            neuron.nudgeBeta(LEARNING_RATE);
                            neuron.nudgeGamma(LEARNING_RATE);

                            //clear gradients for next batch
                            neuron.clearGradients();
                        }
                    }

                    //calculating success rate
                    successRate = (double) successfulPairs / (double) totalPairs;
                    System.out.println("Success rate: " + successRate * 100 + "%");
                }
            }
        }
    }

    public void compute(double[] image, char label) {
        //setting values for all nodes in input layer
        Neuron[] layer = LAYERS[0];
        for (int i = 0; i < layer.length; i++) {
            layer[i].setActivation(image[i]);
        }

        //computing the values for subsequent layers
        for (int i = 1; i < LAYERS.length; i++) {
            layer = LAYERS[i];
            //compute values for all nodes
            for (Neuron neuron : layer) {
                neuron.compute();
            }

            //apply softmax to output layer
            if (i == LAYERS.length - 1) {
                double sumPowers = Arrays.stream(layer).mapToDouble(neuron -> Math.exp(neuron.getActivation())).sum();
                for (Neuron neuron : layer) {
                    neuron.setActivation(Math.exp(neuron.getActivation()) / sumPowers);
                }
            }
        }

        //propagate gradients backwards once all nodes have computed their values
        backPropagate(label);
    }

    public void backPropagate(char label) {
        //calculate gradients for output layer
        for (Neuron neuron : LAYERS[LAYERS.length - 1]) {
            double expectedActivation = neuron.getNodeLabel() == label ? 1 : 0;
            neuron.addWeightGradients(expectedActivation);
            neuron.addBiasGradient(expectedActivation);
        }

        //calculate gradients for hidden layers
        for (int i = LAYERS.length - 2; i > 0; i--) {
            Neuron[] layer = LAYERS[i];
            for (Neuron neuron : layer) {
                neuron.addWeightGradients(-1);
                neuron.addBiasGradient(-1);
                neuron.addBetaGradient(-1);
                neuron.addGammaGradient(-1);
            }
        }
    }
}