package network;

import java.util.*;

public class Node {
    //private final Layer LAYER;

    protected enum NodeType {
        INPUT,
        OUTPUT,
        HIDDEN
    }

    private final NodeType NODE_TYPE;
    private final Character NODE_LABEL;
    private final LinkedHashMap<Node, Double> INPUTS;
    private final LinkedList<Node> OUTPUTS;
    private final LinkedList<double[]> WEIGHT_GRADIENTS;
    private final LinkedList<Double> BIAS_GRADIENTS;
    private double gradientFactor;
    private double activation;
    private double weightedInputs;
    private double bias;
    private final Function ACTIVATION_FUNCTION;

    public Node(NodeType nodeType, Character nodeLabel) {
        NODE_TYPE = nodeType;
        NODE_LABEL = nodeLabel;
        INPUTS = new LinkedHashMap<>();
        OUTPUTS = new LinkedList<>();
        WEIGHT_GRADIENTS = new LinkedList<>();
        BIAS_GRADIENTS = new LinkedList<>();
        bias = 0;
        ACTIVATION_FUNCTION = new Function(Function.FunctionTag.RELU);
    }

    public void compute() {
        if(NODE_TYPE == NodeType.INPUT) throw new IllegalStateException("A node in an input layer cannot update its value.");
        //sum values from all input nodes multiplied by their respective weight and add bias
        weightedInputs = INPUTS.entrySet().stream().mapToDouble(n -> n.getKey().activation * n.getValue()).sum() + bias;

        if(NODE_TYPE != NodeType.OUTPUT) {
            activation = ACTIVATION_FUNCTION.compute(weightedInputs);
        } else {
            activation = weightedInputs;
        }
    }

    public double meanSquaredError(double expectedValue) {
        return (activation - expectedValue) * (activation - expectedValue);
    }

    public double meanSquaredErrorDerivative(double expectedValue) {
        return 2 * (activation - expectedValue);
    }

    public double crossEntropyCost(double expectedValue) {
        return -(expectedValue * Math.log(activation) + (1 - expectedValue) * Math.log(1 - activation));
    }

    public double crossEntropyCostDerivative(double expectedValue) {
        //return -(expectedValue * (1 / value) + Math.log(value) + (1 - expectedValue) * (1 / (1 - value)) - Math.log(1 - value));
        return (-activation + expectedValue) / (activation * (activation - 1));
    }

    /**
     * Derivative of the mean squared error with respect to the unrectified activation (∂C/∂z).
     *
     * @param expectedValue Activation of this neuron that is expected for the input image.
     */
    public void setGradientFactor(double expectedValue) {
        if (NODE_TYPE == NodeType.OUTPUT) {
            //TODO: find better way of representing "no expected value"
            if (expectedValue == -1) {
                throw new IllegalStateException("An output layer needs to have a proper expected value.");
            }
            gradientFactor = crossEntropyCostDerivative(expectedValue) * ACTIVATION_FUNCTION.computeDerivative(weightedInputs);
            return;
        }

        gradientFactor = 0;
        gradientFactor = OUTPUTS.stream().mapToDouble(node -> node.gradientFactor * node.INPUTS.get(this)).sum() * ACTIVATION_FUNCTION.computeDerivative(weightedInputs);
    }

    /**
     * Derivative of the cross entropy cost with respect to input weights for output layer (∂C/∂w).
     *
     * @param expectedValue Activation of this neuron that is expected for the input image.
     */
    public void addWeightGradients(double expectedValue) {
        setGradientFactor(expectedValue);
        double[] subWeightGradients = new double[INPUTS.size()];
        int i = 0;
        for (Map.Entry<Node, Double> input : INPUTS.entrySet()) {
            subWeightGradients[i] = gradientFactor * input.getKey().getActivation();
            i++;
        }
        this.WEIGHT_GRADIENTS.add(subWeightGradients);
    }

    /**
     * Derivative of the mean squared error with respect to bias for output layer (∂C/∂b).
     *
     * @param expectedValue Activation of this neuron that is expected for the input image.
     */
    public void addBiasGradient(double expectedValue) {
        setGradientFactor(expectedValue);
        this.BIAS_GRADIENTS.add(activation - expectedValue);
    }

    public void nudgeWeights(double learningRate) {
        //create array to store average gradient for each input weight
        double[] averageWeightGradients = new double[INPUTS.size()];

        //iterate through arrays of gradients
        for (int i = 0; i < averageWeightGradients.length; i++) {
            double[] gradientsForSingleNode = new double[WEIGHT_GRADIENTS.size()];

            //iterate through all ith positions of arrays in list
            for (int j = 0; j < WEIGHT_GRADIENTS.size(); j++) {

                //take ith value from each array in list and store in array
                gradientsForSingleNode[j] = WEIGHT_GRADIENTS.get(j)[i];
            }
            //average gradients for current node and store in array
            averageWeightGradients[i] = Arrays.stream(gradientsForSingleNode).sum() / gradientsForSingleNode.length;
        }

        //subtract averaged gradient multiplied by learning rate from each weight to nudge it towards local minimum
        int i = 0;
        for (Map.Entry<Node, Double> input : INPUTS.entrySet()) {
            INPUTS.replace(input.getKey(), input.getValue() - averageWeightGradients[i] * learningRate);
            i++;
        }
    }

    public void nudgeBias(double learningRate) {
        //subtract averaged gradient multiplied by learning rate from bias to nudge it towards local minimum
        bias -= (BIAS_GRADIENTS.stream().mapToDouble(Double::doubleValue).sum() / BIAS_GRADIENTS.size()) * learningRate;
    }

    public void addInputNode(Node input, double weight) {
        INPUTS.put(input, weight);
        input.OUTPUTS.add(this);
    }

    public void clearGradients() {
        WEIGHT_GRADIENTS.clear();
        BIAS_GRADIENTS.clear();
    }

    public double getActivation() {
        return activation;
    }

    public void setActivation(double activation) {
        this.activation = activation;
    }

    public char getNodeLabel() {
        if(NODE_LABEL == null) throw new IllegalStateException("This node does not have label.");
        return NODE_LABEL;
    }
}
