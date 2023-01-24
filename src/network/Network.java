package network;

import java.io.*;
import java.util.*;

import static main.Main.FILE_HANDLER;

public class Network implements Serializable {
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
                Arrays.setAll(inputLayerNeurons, node -> new Neuron(Neuron.NodeType.INPUT, null));
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
                    outputLayerNeurons[nodeIndex] = new Neuron(Neuron.NodeType.OUTPUT, (char) i);
                    nodeIndex++;
                }

                //adding nodes for uppercase letters
                for (int i = 65; i <= 90; i++) {
                    outputLayerNeurons[nodeIndex] = new Neuron(Neuron.NodeType.OUTPUT, (char) i);
                    nodeIndex++;
                }

                //adding nodes for lowercase letters
                for (int i = 97; i <= 122; i++) {
                    outputLayerNeurons[nodeIndex] = new Neuron(Neuron.NodeType.OUTPUT, (char) i);
                    nodeIndex++;
                }

                LAYERS[LAYERS.length - 1] = outputLayerNeurons;
                System.out.println("Output layer created with " + outputLayerSize + " nodes.");
                continue;
            }

            //creating hidden layers
            Neuron[] hiddenLayerNeurons = new Neuron[hiddenLayerSizes[l - 1]];
            Arrays.setAll(hiddenLayerNeurons, node -> new Neuron(Neuron.NodeType.HIDDEN, null));
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

    /**
     * @param trainingData The data to be shuffled.
     * @return The shuffled data.
     */
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

    /**
     * Train the network.
     */
    public void train() {
        //load files
        try {
            FILE_HANDLER.loadFiles(new File("C:\\Users\\Friiiz\\Documents\\NIST Handwritten Forms and Characters Database Training Images"));
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        LinkedHashMap<double[], Character> trainingData = FILE_HANDLER.getTrainingData();

        //training network
        System.out.println("Training network.");
        int totalPairs = 0;
        int successfulPairs = 0;
        double successRate;
        double highestSuccessRate = 0;

        //looping through epochs
        for (int i = 0; i < 10; i++) {
            LinkedHashMap<double[], Character> shuffledTrainingData = shuffleTrainingData(trainingData);
            //looping through shuffled training data
            for (Map.Entry<double[], Character> trainingPair : shuffledTrainingData.entrySet()) {
                //System.out.println("Evaluating image of character '" + trainingPair.getValue() + "'");
                //computing output for each pair
                compute(trainingPair.getKey());
                backPropagate(trainingPair.getValue());

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
                            //neuron.nudgeBeta(LEARNING_RATE);
                            //neuron.nudgeGamma(LEARNING_RATE);

                            //clear gradients for next batch
                            neuron.clearGradients();
                        }
                    }

                    //calculating success rate
                    successRate = (double) successfulPairs / (double) totalPairs;
                    if(successRate > highestSuccessRate) {
                        highestSuccessRate = successRate;
                    }

                    //save prematurely to avoid  unlearning
                    if (successRate > 0.5) {
                        save(successRate, totalPairs / BATCH_SIZE, i + 1);
                        return;
                    }

                    System.out.println("Success rate: " + successRate * 100 + "%");
                }
            }
        }
    }

    /**
     * Saves the network to a file.
     * @param successRate The success rate at the time of saving.
     * @param batches The number of batch evaluated at the time of saving.
     * @param epochs The number of epochs gone through at the time of saving.
     */
    private void save(double successRate, int batches, int epochs) {
        try (FileOutputStream fileOutputStream = new FileOutputStream("network.ser")) {
            ObjectOutputStream outputStream = new ObjectOutputStream(fileOutputStream);
            outputStream.writeObject(this);
            outputStream.close();
            File networkInfo = new File("Network Info.txt");
            FileWriter writer = new FileWriter(networkInfo);
            writer.write("Network Info:\nAverage success rate (fluctuates a lot for individual characters): " + successRate * 100 + "%\nTrained for " + batches + " batches in " + epochs + " epochs.");
            writer.close();
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Computes the activations for the whole network for the given input.
     * @param image The input image.
     */
    public void compute(double[] image) {
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
    }

    /**
     * Propagate parameters backwards through network.
     * @param label The correct label of the current input image.
     */
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
                //neuron.addBetaGradient(-1);
                //neuron.addGammaGradient(-1);
            }
        }
    }

    /**
     * @param image The image to be evaluated.
     * @return The label and the certainty the network computed for the input image.
     */
    public Map.Entry<Character, Double> evaluate(double[] image) {
        compute(image);
        Neuron maxValue = Arrays.stream(LAYERS[LAYERS.length - 1]).max(Comparator.comparing(Neuron::getActivation)).orElseThrow();
        return new AbstractMap.SimpleEntry<>(maxValue.getNodeLabel(), maxValue.getActivation());
    }

    @Deprecated
    public void test() {
        try {
            FILE_HANDLER.loadFiles(new File("C:\\Users\\Friiiz\\Documents\\NIST Handwritten Forms and Characters Database Test Images"));
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        LinkedHashMap<double[], Character> testData = shuffleTrainingData(FILE_HANDLER.getTrainingData());
        //training network
        System.out.println("Training network.");
        int totalPairs = 0;
        int successfulPairs = 0;
        double successRate;
        double highestSuccessRate = 0;
        int i = 0;
        for (Map.Entry<double[], Character> testPair : testData.entrySet()) {
            //computing output
            compute(testPair.getKey());
            //finding the highest value in output layer
            Neuron maxValue = Arrays.stream(LAYERS[LAYERS.length - 1]).max(Comparator.comparing(Neuron::getActivation)).orElseThrow();
            //track success
            totalPairs++;
            if (maxValue.getNodeLabel() == testPair.getValue()) {
                successfulPairs++;
            }
            //calculating success rate
            successRate = (double) successfulPairs / (double) totalPairs;
            if(successRate > highestSuccessRate) {
                highestSuccessRate = successRate;
            }
            if (successRate > 0.5) {
                save(successRate, totalPairs / BATCH_SIZE, i + 1);
                return;
            }
            i ++;
            System.out.println("Success rate: " + successRate * 100 + "%");
        }
    }
}