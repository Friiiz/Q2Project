package network;

import filehandling.FileHandler;

import java.util.*;

public class Network {
    private final double LEARNING_RATE;

    private final Node[][] LAYERS;

    public Network(double learningRate, int inputLayerSize, int outputLayerSize, int... hiddenLayerSizes) {
        LEARNING_RATE = learningRate;
        LAYERS = new Node[1 + hiddenLayerSizes.length + 1][];
        for (int l = 0; l < LAYERS.length; l++) {
            if (l == 0) {
                //creating input layer
                Node[] inputLayerNodes = new Node[inputLayerSize];
                Arrays.setAll(inputLayerNodes, node -> new Node(Node.NodeType.INPUT, null));
                LAYERS[l] = inputLayerNodes;
                System.out.println("Input layer created with " + inputLayerSize + " nodes.");
                continue;
            }
            if (l == LAYERS.length - 1) {
                //creating output layer
                Node[] outputLayerNodes = new Node[outputLayerSize];
                int nodeIndex = 0;
                //adding nodes for numbers
                for (int i = 48; i <= 57; i++) {
                    outputLayerNodes[nodeIndex] = new Node(Node.NodeType.OUTPUT, (char) i);
                    nodeIndex++;
                }

                //adding nodes for uppercase letters
                for (int i = 65; i <= 90; i++) {
                    outputLayerNodes[nodeIndex] = new Node(Node.NodeType.OUTPUT, (char) i);
                    nodeIndex++;
                }

                //adding nodes for lowercase letters
                for (int i = 97; i <= 122; i++) {
                    outputLayerNodes[nodeIndex] = new Node(Node.NodeType.OUTPUT, (char) i);
                    nodeIndex++;
                }

                LAYERS[LAYERS.length - 1] = outputLayerNodes;
                System.out.println("Output layer created with " + outputLayerSize + " nodes.");
                continue;
            }

            //creating hidden layers
            Node[] hiddenLayerNodes = new Node[hiddenLayerSizes[l - 1]];
            Arrays.setAll(hiddenLayerNodes, node -> new Node(Node.NodeType.HIDDEN, null));
            LAYERS[l] = hiddenLayerNodes;
            System.out.println("Hidden layer created with " + hiddenLayerSizes[l - 1] + " nodes.");
        }

        //connecting nodes
        Random random = new Random();
        for (int l = 1; l < LAYERS.length; l++) {
            Node[] layer = LAYERS[l];
            for (Node node : layer) {
                Node[] previousLayer = LAYERS[l - 1];
                for (Node previousNode : previousLayer) {
                    node.addInputNode(previousNode, random.nextGaussian(0, Math.sqrt(1.0d / previousLayer.length)));
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
        int batchSize = 32;
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
                Node maxValue = Arrays.stream(LAYERS[LAYERS.length - 1]).max(Comparator.comparing(Node::getActivation)).orElseThrow();

                //track success
                totalPairs++;
                if (maxValue.getNodeLabel() == trainingPair.getValue()) {
                    successfulPairs++;
                }

                //adjusting parameters after every batch
                if (totalPairs % batchSize == 0) {
                    System.out.println("Adjusting parameters for batch " + totalPairs / batchSize + " in epoch " + (i + 1));
                    for (Node[] layer : LAYERS) {
                        for (Node node : layer) {
                            //nudge parameters
                            node.nudgeWeights(LEARNING_RATE);
                            node.nudgeBias(LEARNING_RATE);

                            //clear gradients for next batch
                            node.clearGradients();
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
        Node[] layer = LAYERS[0];
        for (int i = 0; i < layer.length; i++) {
            layer[i].setActivation(image[i]);
        }

        //computing the values for subsequent layers
        for (int i = 1; i < LAYERS.length; i++) {
            layer = LAYERS[i];
            //compute values for all nodes
            for (Node node : layer) {
                node.compute();
            }

            //apply softmax to output layer
            if (i == LAYERS.length - 1) {
                double sumPowers = Arrays.stream(layer).mapToDouble(node -> Math.pow(Math.E, node.getActivation())).sum();
                for (Node node : layer) {
                    node.setActivation(Math.pow(Math.E, node.getActivation()) / sumPowers);
                }
            }
        }

        //propagate gradients backwards once all nodes have computed their values
        backPropagate(label);
    }

    public void backPropagate(char label) {
        //calculate gradients for output layer
        for (Node node : LAYERS[LAYERS.length - 1]) {
            node.addWeightGradients(node.getNodeLabel() == label ? 1 : 0);
            node.addBiasGradient(node.getNodeLabel() == label ? 1 : 0);
        }

        //calculate gradients for hidden layers
        for (int i = LAYERS.length - 2; i > 0; i--) {
            Node[] layer = LAYERS[i];
            for (Node node : layer) {
                node.addWeightGradients(-1);
                node.addBiasGradient(-1);
            }
        }
    }
}