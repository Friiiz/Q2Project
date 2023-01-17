package network;

public class Function {
    public enum FunctionTag {
        SIGMOID,
        RELU,
        LEAKY_RELU,
        SILU,
        ELU,
        SHIFTED_RELU,
        SCALED_RELU,
        SOFT_PLUS,
        MISH,
        METALLIC_MEAN,
        SHIFTED_LEAKY_RELU
    }

    private final double[] PARAMETERS;

    private final FunctionTag FUNCTION_TAG;

    public Function(FunctionTag functionTag, double... parameters) {
        this.FUNCTION_TAG = functionTag;
        this.PARAMETERS = parameters;

        switch (FUNCTION_TAG) {
            case SIGMOID -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The sigmoid function needs exactly 1 parameter.");
                if(PARAMETERS[0] < 0) throw new IllegalStateException("The parameter for the sigmoid function has to be bigger than 0.");
            }

            case RELU -> {
                if(PARAMETERS.length != 0) throw new IllegalStateException("The ReLU function does not have any parameters.");
            }

            case LEAKY_RELU -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The leaky ReLU function needs exactly 1 parameter.");
                if(PARAMETERS[0] < 0 || PARAMETERS[0] > 1) throw new IllegalStateException("The parameter for the leaky ReLU function has to be between 0 & 1.");
            }

            case SILU -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The SiLU function needs exactly 1 parameter.");
                if(PARAMETERS[0] < 1) throw new IllegalStateException("The parameter for the SiLU function has to be bigger than 1.");
            }

            case ELU -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The ELU function needs exactly 1 parameter.");
                if(PARAMETERS[0] < 0 || PARAMETERS[0] > 1) throw new IllegalStateException("The parameter for the ELU function has to be between 0 & 1.");
            }

            case SHIFTED_RELU -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The shifted ReLU function needs exactly 1 parameter.");
                if(PARAMETERS[0] < 0) throw new IllegalStateException("The parameter for the shifted ReLU function has to be bigger than 0.");
            }

            case SCALED_RELU -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The scaled ReLU function needs exactly 1 parameter.");
                if(PARAMETERS[0] < 0) throw new IllegalStateException("The parameter for the scaled ReLU function has to be bigger than 0.");
            }

            case SOFT_PLUS -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The Softplus function needs exactly 1 parameter.");
                if(PARAMETERS[0] < 1) throw new IllegalStateException("The parameter for the Softplus function has to be bigger than 1.");
            }

            case MISH -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The Mish function needs exactly 1 parameter");
                if(PARAMETERS[0] < 0) throw new IllegalStateException("The parameter for the Mish function has to be bigger than 0.");
            }

            case METALLIC_MEAN -> {
                if(PARAMETERS.length != 1) throw new IllegalStateException("The metallic mean function needs exactly 1 parameter.");
                if(PARAMETERS[0] < 0) throw new IllegalStateException("The parameter for the metallic mean function has to be bigger than 0.");
            }

            case SHIFTED_LEAKY_RELU -> {
                if(PARAMETERS.length != 2) throw new IllegalStateException("The shifted leaky ReLU function needs exactly 2 parameters.");
                if(PARAMETERS[0] < 0) throw new IllegalStateException("The first parameter for the shifted leaky ReLU function has to be bigger than 0.");
                if(PARAMETERS[1] < 0 || PARAMETERS[1] > 1) throw new IllegalStateException("The second parameter for the shifted leaky ReLU function has to be between 0 & 1.");
            }
        }
    }

    public double compute(double x) {
        return switch (FUNCTION_TAG) {
            case SIGMOID -> sigmoid(x);
            case RELU -> reLU(x);
            case LEAKY_RELU -> leakyReLU(x);
            case SILU -> siLU(x);
            case ELU -> eLU(x);
            case SHIFTED_RELU -> shiftedReLU(x);
            case SCALED_RELU -> scaledReLU(x);
            case SOFT_PLUS -> softPlus(x);
            case MISH -> mish(x);
            case METALLIC_MEAN -> metallicMean(x);
            case SHIFTED_LEAKY_RELU -> shiftedLeakyReLU(x);
        };
    }

    public double computeDerivative(double x) {
        return switch (FUNCTION_TAG) {
            case SIGMOID -> sigmoidDerivative(x);
            case RELU -> reLUDerivative(x);
            case LEAKY_RELU -> leakyReLUDerivative(x);
            case SILU -> siLUDerivative(x);
            case ELU -> eLUDerivative(x);
            case SHIFTED_RELU -> shiftedReLUDerivative(x);
            case SCALED_RELU -> scaledReLUDerivative(x);
            case SOFT_PLUS -> softPlusDerivative(x);
            case MISH -> mishDerivative(x);
            case METALLIC_MEAN -> metallicMeanDerivative(x);
            case SHIFTED_LEAKY_RELU -> shiftedLeakyReLUDerivative(x);
        };
    }

    private double sigmoid(double x) {
        double a = PARAMETERS[0]; //0 < a
        return 1 / (1 + Math.exp(-a * x));
    }

    private double sigmoidDerivative(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    private double reLU(double x) {
        return x > 0 ? x : 0;
    }

    private double reLUDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private double leakyReLU(double x) {
        double a = PARAMETERS[0]; //0 < a < 1
        return x > 0 ? x : a * x;
    }

    private double leakyReLUDerivative(double x) {
        double a = PARAMETERS[0]; //0 < a < 1
        return x > 0 ? 1 : a;
    }

    private double siLU(double x) {
        double a = PARAMETERS[0]; //1 < a
        return x / (1 + Math.exp(-a * x));
    }

    private double siLUDerivative(double x) {
        double a = PARAMETERS[0]; //1 < a
        return (Math.exp(a * x) * (a * x + Math.exp(a * x) + 1)) / Math.pow(Math.exp(a * x) + 1, 2);
    }

    private double eLU(double x) {
        double a = PARAMETERS[0]; //0 < a < 1
        return x > 0 ? x : a * (Math.exp(x) - 1);
    }

    private double eLUDerivative(double x) {
        double a = PARAMETERS[0]; //0 < a < 1
        return x > 0 ? 1 : a * Math.exp(x);
    }

    private double shiftedReLU(double x) {
        double a = PARAMETERS[0]; //0 < a
        return x > 0 ? x : -a;
    }

    private double shiftedReLUDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private double scaledReLU(double x) {
        double a = PARAMETERS[0]; //0 < a
        return x > 0 ? a * x : 0;
    }

    private double scaledReLUDerivative(double x) {
        double a = PARAMETERS[0]; //0 < a
        return x > 0 ? a : 0;
    }

    private double softPlus(double x) {
        double a = PARAMETERS[0]; //1 < a
        return Math.log(1 + Math.exp(a * x)) / a;
    }

    private double softPlusDerivative(double x) {
        double a = PARAMETERS[0]; //1 < a
        return 1 / (1 + Math.exp(-a * x));
    }

    private double mish(double x) {
        double a = PARAMETERS[0]; //0 < a
        return x * Math.tanh(Math.log(1 + Math.exp(a * x)) / a);
    }

    private double mishDerivative(double x) {
        double a = PARAMETERS[0]; //0 < a
        return Math.tanh(Math.log(1 + Math.exp(a * x)) / a);
    }

    private double metallicMean(double x) {
        double a = PARAMETERS[0]; //0 < a
        return (x + Math.sqrt(x * x + a)) / 2;
    }

    private double metallicMeanDerivative(double x) {
        double a = PARAMETERS[0]; //0 < a
        return (x / Math.sqrt(x * x + a) + 1) / 2;
    }

    private double shiftedLeakyReLU(double x) {
        double a = PARAMETERS[0]; //0 < a
        double b = PARAMETERS[1]; //0 < b < 1
        return x > 0 ? x : b * x - a;
    }

    private double shiftedLeakyReLUDerivative(double x) {
        double b = PARAMETERS[1]; //0 < b < 1
        return x > 0 ? 1 : b;
    }
}
