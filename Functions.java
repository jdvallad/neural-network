
public class Functions {
    public static double activate(double input, String str, boolean b) throws Exception {
        switch (str) {
            case "swish":
                return swish(input, b);
            case "sigmoid":
                return sigmoid(input, b);
            case "leakyRelu":
                return leakyRELU(input, b);
            case "relu":
                return RELU(input, b);
        }
        throw new Exception("No match found for \"" + str + "\"");
    }


    public static double cost(double output, double expected, String str, boolean b) throws Exception {
        switch (str) {
            case "logLoss":
                return logLoss(output, expected, b);
            case "meanSquaredError":
                return meanSquaredError(output, expected, b);
        }
        throw new Exception("No match found for \"" + str + "\"");
    }

    public static double weightInitialize() {
        return 0.1;
    }

    public static double biasInitialize() {
        return 0.1;
    }

    public static double sigmoid(double input, boolean b) {
        if (b) {
            return 1. / (1. + Math.exp(-input));
        } else {
            double temp = sigmoid(input, true);
            return temp * (1. - temp);
        }
    }

    public static double leakyRELU(double input, boolean b) {
        if (b) {
            return input >= 0 ? input : 0.01 * input;
        } else {
            return input >= 0 ? 1 : 0.01;
        }
    }

    public static double RELU(double input, boolean b) {
        if (b) {
            return Math.max(0., input);
        } else {
            return input >= 0 ? 1 : 0.;
        }
    }

    public static double swish(double input, boolean b) {
        if (b) {
            return input * sigmoid(input, true);
        } else {
            double temp = swish(input, true);
            return temp + sigmoid(input, true) * (1. - temp);
        }
    }

    public static double logLoss(double output, double expected, boolean b) {
        double shiftOutput = Math.max(Math.min(output, 1 - Math.pow(10, -15)), Math.pow(10, -15));
        double shiftExpected = Math.max(Math.min(expected, 1 - Math.pow(10, -15)), Math.pow(10, -15));
        if (b) {
            return -shiftExpected * Math.log(shiftOutput) - (1. - shiftExpected) * Math.log(1. - shiftOutput);
        } else {
            return -shiftExpected / shiftOutput + (1 - shiftExpected) / (1 - shiftOutput);
        }
    }

    public static double meanSquaredError(double output, double expected, boolean b) {
        if (b) {
            return Math.pow(output - expected, 2);
        } else {
            return 2.0 * (output - expected);
        }
    }

    public static double[] randomInput(double[] input) {
        for (int i = 0; i < input.length; i++)
            input[i] = Math.random();
        return input;
    }

    public static int collapse(double[] arr) {
        int max = 0;
        for (int i = 0; i < arr.length; i++)
            if (arr[i] > arr[max])
                max = i;
        return max;
    }
}
