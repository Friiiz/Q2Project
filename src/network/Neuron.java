package network;

import java.io.Serializable;
import java.util.*;

public class Neuron implements Serializable {

    protected enum NodeType implements Serializable {
        INPUT,
        OUTPUT,
        HIDDEN
    }

    private final NodeType NODE_TYPE;
    private final Character NODE_LABEL;
    private final LinkedHashMap<Neuron, Double> INPUTS;
    private final LinkedList<Neuron> OUTPUTS;
    private final LinkedList<double[]> WEIGHT_GRADIENTS;
    private final LinkedList<Double> BIAS_GRADIENTS;
    private final LinkedList<Double> BETA_GRADIENTS;
    private final LinkedList<Double> GAMMA_GRADIENTS;
    private final LinkedList<Double> BATCH_ACTIVATIONS;
    private double gradientFactor;
    private double activation;
    private double normalizedActivation;
    private double scaledShiftedNormalizedActivation;
    private double weightedInputSum;
    private double bias;
    private double beta;
    private double gamma;
    private double mean;
    private double standardDeviation;
    private final Function ACTIVATION_FUNCTION;

    public Neuron(NodeType nodeType, Character nodeLabel) {
        NODE_TYPE = nodeType;
        NODE_LABEL = nodeLabel;
        INPUTS = new LinkedHashMap<>();
        OUTPUTS = new LinkedList<>();
        WEIGHT_GRADIENTS = new LinkedList<>();
        BIAS_GRADIENTS = new LinkedList<>();
        BETA_GRADIENTS = new LinkedList<>();
        GAMMA_GRADIENTS = new LinkedList<>();
        BATCH_ACTIVATIONS = new LinkedList<>();
        bias = 0;
        ACTIVATION_FUNCTION = new Function(Function.FunctionTag.RELU);
    }

    /**
     * Computes {@link network.Neuron#activation} using the weighted sum of {@link network.Neuron#INPUTS} and {@link network.Neuron#ACTIVATION_FUNCTION}.
     *
     */
    public void compute() {
        if (NODE_TYPE == NodeType.INPUT) {
            throw new IllegalStateException("A node in an input layer cannot update its value.");
        }

        //check if previous layer is hidden
        if(INPUTS.keySet().stream().findAny().orElseThrow().NODE_TYPE == NodeType.HIDDEN) {
            //sum batch normalized activations from all input nodes multiplied by their respective weight and add bias
            weightedInputSum = INPUTS.entrySet().stream().mapToDouble(n -> n.getKey().scaledShiftedNormalizedActivation * n.getValue()).sum() + bias;
        } else {
            //sum activations from all input nodes multiplied by their respective weight and add bias
            weightedInputSum = INPUTS.entrySet().stream().mapToDouble(n -> n.getKey().activation * n.getValue()).sum() + bias;
        }

        if (NODE_TYPE != NodeType.OUTPUT) {
            activation = ACTIVATION_FUNCTION.compute(weightedInputSum);
        } else {
            activation = weightedInputSum;
        }

        BATCH_ACTIVATIONS.add(activation);
    }

    @Deprecated(since = "Implementation of the cross entropy cost function")
    public double meanSquaredError(double expectedActivation) {
        return (activation - expectedActivation) * (activation - expectedActivation);
    }

    @Deprecated(since = "Implementation of the cross entropy cost function")
    public double meanSquaredErrorDerivative(double expectedActivation) {
        return 2 * (activation - expectedActivation);
    }

    /**
     *
     * @param expectedActivation The {@link network.Neuron#activation} of this neuron that is expected for the given input.
     * @return The "cost" of the given training example.
     */
    public double crossEntropyCost(double expectedActivation) {
        return -(expectedActivation * Math.log(activation) + (1 - expectedActivation) * Math.log(1 - activation));
    }

    /**
     *
     * @param expectedActivation The {@link network.Neuron#activation} of this neuron that is expected for the given input.
     * @return The derivative of the "cost" of the given training example.
     */
    public double crossEntropyCostDerivative(double expectedActivation) {
        //return -(expectedActivation * (1 / value) + Math.log(value) + (1 - expectedActivation) * (1 / (1 - value)) - Math.log(1 - value));
        return (-activation + expectedActivation) / (activation * (activation - 1));
    }

    /**
     * Derivative of the {@link network.Neuron#crossEntropyCost(double)} with respect to {@link network.Neuron#weightedInputSum} (∂C/∂z).
     *
     * @param expectedActivation The {@link network.Neuron#activation} of this neuron that is expected for the given input.
     */
    public void setGradientFactor(double expectedActivation) {
        if (expectedActivation != -1) {
            gradientFactor = crossEntropyCostDerivative(expectedActivation) * ACTIVATION_FUNCTION.computeDerivative(weightedInputSum);
            return;
        }

        gradientFactor = 0; //TODO: redundant?
        gradientFactor = OUTPUTS.stream().mapToDouble(neuron -> neuron.gradientFactor * neuron.INPUTS.get(this)).sum() * ACTIVATION_FUNCTION.computeDerivative(weightedInputSum);
    }

    /**
     * Derivative of {@link network.Neuron#crossEntropyCost(double)} with respect to {@link network.Neuron#INPUTS} (∂C/∂w).
     *
     * @param expectedActivation Activation of this neuron that is expected for the input image.
     */
    public void addWeightGradients(double expectedActivation) {
        setGradientFactor(expectedActivation);
        double[] subWeightGradients = new double[INPUTS.size()];
        int i = 0;
        for (Map.Entry<Neuron, Double> input : INPUTS.entrySet()) {
            subWeightGradients[i] = gradientFactor * input.getKey().getActivation();
            i++;
        }
        WEIGHT_GRADIENTS.add(subWeightGradients);
    }

    /**
     * Derivative of {@link network.Neuron#crossEntropyCost(double)} with respect to {@link network.Neuron#bias} (∂C/∂b).
     * @param expectedActivation Activation of this neuron that is expected for the input image.
     */
    public void addBiasGradient(double expectedActivation) {
        setGradientFactor(expectedActivation);
        BIAS_GRADIENTS.add(gradientFactor);
    }

    @Deprecated()
    private void setMean() {
        mean = BATCH_ACTIVATIONS.stream().mapToDouble(Double::doubleValue).sum() / BATCH_ACTIVATIONS.size();
    }

    @Deprecated()
    private void setStandardDeviation() {
        standardDeviation = Math.sqrt(BATCH_ACTIVATIONS.stream().mapToDouble(a -> (a - mean) * (a - mean)).sum() / BATCH_ACTIVATIONS.size());
    }

    @Deprecated()
    private void setNormalizedActivation() {
        setMean();
        setStandardDeviation();
        normalizedActivation = (activation * mean) / standardDeviation;
        //TODO: potentially add noise to standardDeviation if it becomes 0?
    }

    @Deprecated()
    private void setScaledShiftedNormalizedActivation() {
        setNormalizedActivation();
        scaledShiftedNormalizedActivation = gamma * normalizedActivation + beta;
    }

    @Deprecated
    public void addBetaGradient(double expectedActivation) {
        setScaledShiftedNormalizedActivation();
        if (expectedActivation == -1) {
            expectedActivation = OUTPUTS.stream().mapToDouble(neuron -> neuron.INPUTS.entrySet().stream().filter(n -> n.getKey() == this).findAny().orElseThrow().getKey().weightedInputSum /
                    neuron.INPUTS.entrySet().stream().filter(n -> n.getKey() == this).findAny().orElseThrow().getValue()).sum() / OUTPUTS.size();
        }
        BETA_GRADIENTS.add(expectedActivation - scaledShiftedNormalizedActivation);
    }

    @Deprecated
    public void addGammaGradient(double expectedActivation) {
        setScaledShiftedNormalizedActivation();
        if (expectedActivation == -1) {
            expectedActivation = OUTPUTS.stream().mapToDouble(neuron -> neuron.INPUTS.entrySet().stream().filter(n -> n.getKey() == this).findAny().orElseThrow().getKey().weightedInputSum /
                    neuron.INPUTS.entrySet().stream().filter(n -> n.getKey() == this).findAny().orElseThrow().getValue()).sum() / OUTPUTS.size();
        }
        GAMMA_GRADIENTS.add((expectedActivation - scaledShiftedNormalizedActivation) * normalizedActivation);
    }

    /**
     * Nudges the weights based on the averages of the gradients per batch.
     * @param learningRate The learning rate of the network used to scale the gradients.
     */
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
        for (Map.Entry<Neuron, Double> input : INPUTS.entrySet()) {
            INPUTS.replace(input.getKey(), input.getValue() - averageWeightGradients[i] * learningRate);
            i++;
        }
    }

    /**
     * Nudges the biases based on the averages of the gradients per batch.
     * @param learningRate The learning rate of the network used to scale the gradients.
     */
    public void nudgeBias(double learningRate) {
        //subtract averaged gradient multiplied by learning rate from bias to nudge it towards local minimum
        bias -= (BIAS_GRADIENTS.stream().mapToDouble(Double::doubleValue).sum() / BIAS_GRADIENTS.size()) * learningRate;
    }

    @Deprecated
    public void nudgeBeta(double learningRate) {
        //subtract averaged gradient multiplied by learning rate from bias to nudge it towards local minimum
        beta -= -(BETA_GRADIENTS.stream().mapToDouble(Double::doubleValue).sum() / BETA_GRADIENTS.size()) * learningRate;
    }

    @Deprecated
    public void nudgeGamma(double learningRate) {
        //subtract averaged gradient multiplied by learning rate from bias to nudge it towards local minimum
        gamma -= -(GAMMA_GRADIENTS.stream().mapToDouble(Double::doubleValue).sum() / GAMMA_GRADIENTS.size()) * learningRate;
    }

    /**
     * Adds an input to this neuron and this node as an output to the new input.
     * @param input The neuron to be added as an input.
     * @param weight The weight for this connection.
     */
    public void addInputNode(Neuron input, double weight) {
        INPUTS.put(input, weight);
        input.OUTPUTS.add(this);
    }

    /**
     * Clears all gradients and activations accumulated for respective batch.
     */
    public void clearGradients() {
        WEIGHT_GRADIENTS.clear();
        BIAS_GRADIENTS.clear();
        //BETA_GRADIENTS.clear();
        //GAMMA_GRADIENTS.clear();
        BATCH_ACTIVATIONS.clear();
    }

    public double getActivation() {
        return activation;
    }

    @Deprecated
    public double getScaledShiftedNormalizedActivation() {
        return scaledShiftedNormalizedActivation;
    }

    public void setActivation(double activation) {
        this.activation = activation;
    }

    public char getNodeLabel() {
        if (NODE_LABEL == null) throw new IllegalStateException("This node does not have label.");
        return NODE_LABEL;
    }
}
