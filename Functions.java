import java.util.Random;

public class Functions {
    static Random rand = new Random(0);

    static double activate(double input, String str, int deriv) throws Exception {
        switch (str) {
            case "swish":
                return swish(input, deriv);
            case "sigmoid":
                return sigmoid(input, deriv);
            case "leakyrelu":
                return leakyrelu(input, deriv);
            case "relu":
                return relu(input, deriv);
            case "tanh":
                return tanh(input, deriv);
            case "softmax":
                throw new Exception("Only use softmax when combined with logLoss!");
        }
        throw new Exception("No match found for \"" + str + "\"");
    }

    static double cost(double output, double expected, String str, int deriv) throws Exception {
        switch (str) {
            case "logLoss":
                return logLoss(output, expected, deriv);
            case "meanSquaredError":
                return meanSquaredError(output, expected, deriv);
        }
        throw new Exception("No match found for \"" + str + "\"");
    }

    static double[] clone(double[] input) {
        return scale(input, 1.);
    }

    static double[][] clone(double[][] input) {
        return scale(input, 1.);
    }

    static double[][][] clone(double[][][] input) {
        return scale(input, 1.);
    }

    static double heParameterInitialize(int previousLayer) {
        return rand.nextGaussian() * Math.sqrt(2. / ((double) previousLayer));
    }

    static void print(double[] input) {
        System.out.print("\r\n[");
        for (int i = 0; i < input.length - 1; i++)
            System.out.print("" + input[i] + ", ");
        System.out.print(input[input.length - 1] + "]\r\n");
    }

    static double softmax(double[] inputLayer, double input) {
        double total = 0.;
        for (int i = 0; i < inputLayer.length; i++)
            total += Math.exp(inputLayer[i]);
        return Math.exp(input) / total;
    }

    static double max(double[] input){
        double output = input[0];
        for(int i = 0; i < input.length; i ++){
            if(input[i] > output){
                output = input[i];
            }
        }
        return output;
    }
    static double sigmoid(double input, int deriv) {
        switch (deriv) {
            case 0:
                return 1. / (1. + Math.exp(-input));
            case 1:
                double temp = sigmoid(input, 0);
                return temp * (1. - temp);
            default:
                return Double.MAX_VALUE;
        }
    }

    static double leakyrelu(double input, int deriv) {
        switch (deriv) {
            case 0:
                return input >= 0 ? input : 0.01 * input;
            case 1:
                return input >= 0 ? 1 : 0.01;
            default:
                return 0;
        }
    }

    static double relu(double input, int deriv) {
        switch (deriv) {
            case 0:
                return Math.max(0., input);
            case 1:
                return input >= 0 ? 1 : 0.;
            default:
                return Double.MAX_VALUE;
        }
    }

    static double swish(double input, int deriv) {
        switch (deriv) {
            case 0:
                return input * sigmoid(input, 0);
            case 1:
                double temp = swish(input, 0);
                return temp + sigmoid(input, 0) * (1. - temp);
            default:
                return Double.MAX_VALUE;
        }
    }

    static double tanh(double input, int deriv) {
        switch (deriv) {
            case 0:
                return 1. - 2. * sigmoid(-2. * input, 0);
            case 1:
                return 4. * sigmoid(-2. * input, 1);
            default:
                return Double.MAX_VALUE;
        }
    }

    static double logLoss(double output, double expected, int deriv) {
        double shiftOutput = Math.max(Math.min(output, 1 - Math.pow(10, -15)), Math.pow(10, -15));
        double shiftExpected = Math.max(Math.min(expected, 1 - Math.pow(10, -15)), Math.pow(10, -15));
        switch (deriv) {
            case 0:
                return -shiftExpected * Math.log(shiftOutput) - (1. - shiftExpected) * Math.log(1. - shiftOutput);
            case 1:
                return -shiftExpected / shiftOutput + (1 - shiftExpected) / (1 - shiftOutput);
            default:
                return Double.MAX_VALUE;
        }
    }

    static double meanSquaredError(double output, double expected, int deriv) {
        switch (deriv) {
            case 0:
                return Math.pow(output - expected, 2);
            case 1:
                return 2.0 * (output - expected);
            default:
                return Double.MAX_VALUE;
        }
    }

    static double[] randomInput(double[] input) {
        for (int i = 0; i < input.length; i++)
            input[i] = Math.random();
        return input;
    }

    static int collapse(double[] arr) {
        int max = 0;
        for (int i = 0; i < arr.length; i++)
            if (arr[i] > arr[max])
                max = i;
        return max;
    }

    static double[] flatten(double[][] input) {
        double[] output = new double[input.length * input[0].length];
        for (int r = 0, index = 0; r < input.length; r++)
            for (int c = 0; c < input[0].length; c++)
                output[index++] = input[r][c];
        return output;
    }

    static double[] flatten(int[][] input) {
        double[] output = new double[input.length * input[0].length];
        for (int r = 0, index = 0; r < input.length; r++)
            for (int c = 0; c < input[0].length; c++)
                output[index++] = input[r][c];
        return output;
    }

    static double[] flatten(double[][][] input) {
        double[] output = new double[input.length * input[0].length * input[0][0].length];
        for (int r = 0, index = 0; r < input.length; r++)
            for (int c = 0; c < input[0].length; c++)
                for (int k = 0; k < input[0][0].length; k++)
                    output[index++] = input[r][c][k];
        return output;
    }

    static double[] flatten(int[][][] input) {
        double[] output = new double[input.length * input[0].length * input[0][0].length];
        for (int r = 0, index = 0; r < input.length; r++)
            for (int c = 0; c < input[0].length; c++)
                for (int k = 0; k < input[0][0].length; k++)
                    output[index++] = input[r][c][k];
        return output;
    }

    static double[][] shape(double[] input, int width, int height) {
        double[][] output = new double[width][height];
        for (int r = 0, index = 0; r < width; r++)
            for (int c = 0; c < height; c++)
                output[r][c] = input[index++];
        return output;
    }

    static double[][][] shape(double[] input, int width, int height, int depth) {
        double[][][] output = new double[width][height][depth];
        for (int r = 0, index = 0; r < width; r++)
            for (int c = 0; c < height; c++)
                for (int k = 0; k < depth; k++)
                    output[r][c][k] = input[index++];
        return output;
    }

    static double[] scale(double[] input, double scale) {
        double[] res = new double[input.length];
        for (int i = 0; i < res.length; i++) {
            res[i] = scale * input[i];
        }
        return res;
    }

    static double[][] scale(double[][] input, double scale) {
        double[][] res = new double[input.length][input[0].length];
        for (int i = 0; i < res.length; i++) {
            for (int r = 0; r < res[0].length; r++)
                res[i][r] = scale * input[i][r];
        }
        return res;
    }

    static double[][][] scale(double[][][] input, double scale) {
        double[][][] res = new double[input.length][input[0].length][input[0][0].length];
        for (int i = 0; i < res.length; i++) {
            for (int r = 0; r < res[0].length; r++)
                for (int c = 0; c < res[0][0].length; c++)
                    res[i][r][c] = scale * input[i][r][c];
        }
        return res;
    }

    public static void set(double[] x, double[] y) {
        for (int i = 0; i < Math.min(x.length, y.length); i++) {
            x[i] = y[i];
        }
        return;
    }

    public static void set(double[] x, double y) {
        for (int i = 0; i < x.length; i++) {
            x[i] = y;
        }
        return;
    }

    public static void set(double[][] x, double[][] y) {
        for (int r = 0; r < Math.min(x.length, y.length); r++) {
            for (int c = 0; c < Math.min(x[0].length, y[0].length); c++) {
                x[r][c] = y[r][c];
            }
        }
        return;
    }

    public static void set(double[][] x, double y) {
        for (int r = 0; r < x.length; r++) {
            for (int c = 0; c < x[0].length; c++) {
                x[r][c] = y;
            }
        }
        return;
    }

    public static void set(double[][][] x, double[][][] y) {
        for (int r = 0; r < x.length; r++) {
            for (int c = 0; c < x[0].length; c++) {
                for (int k = 0; k < x[0][0].length; k++) {
                    x[r][c][k] = y[r][c][k];
                }
            }
        }
        return;
    }

    public static void set(double[][][] x, double y) {
        for (int r = 0; r < x.length; r++) {
            for (int c = 0; c < x[0].length; c++) {
                for (int k = 0; k < x[0][0].length; k++) {
                    x[r][c][k] = y;
                }
            }
        }
        return;
    }

    public static double[] product(double[] x, double y) {
        double[] res = x.clone();
        for (int i = 0; i < x.length; i++)
            res[i] = y * x[i];
        return res;
    }

    public static double[][] product(double[][] x, double y) {
        double[][] res = x.clone();
        for (int r = 0; r < x.length; r++)
            for (int c = 0; c < x[0].length; c++)
                res[r][c] = y * x[r][c];
        return res;
    }

    public static double[][][] product(double[][][] x, double y) {
        double[][][] res = x.clone();
        for (int r = 0; r < x.length; r++)
            for (int c = 0; c < x[r].length; c++)
                for (int k = 0; k < x[r][c].length; k++)
                    res[r][c][k] = y * x[r][c][k];
        return res;
    }

    public static void increase(double[] x, double[] y) {
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] + y[i];
        }
        return;
    }

    public static void increase(double[] x, double y) {
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] + y;
        }
        return;
    }

    public static void increase(double[][] x, double[][] y) {
        for (int r = 0; r < x.length; r++) {
            for (int c = 0; c < x[r].length; c++) {
                x[r][c] = x[r][c] + y[r][c];
            }
        }
        return;
    }

    public static void increase(double[][] x, double y) {
        for (int r = 0; r < x.length; r++) {
            for (int c = 0; c < x[r].length; c++) {
                x[r][c] = x[r][c] + y;
            }
        }
        return;
    }

    public static void increase(double[][][] x, double[][][] y) {
        for (int r = 0; r < x.length; r++) {
            for (int c = 0; c < x[r].length; c++) {
                for (int k = 0; k < x[r][c].length; k++) {
                    x[r][c][k] = x[r][c][k] + y[r][c][k];
                }
            }
        }
        return;
    }

    public static void increase(double[][][] x, double y) {
        for (int r = 0; r < x.length; r++) {
            for (int c = 0; c < x[r].length; c++) {
                for (int k = 0; k < x[r][c].length; k++) {
                    x[r][c][k] = x[r][c][k] + y;
                }
            }
        }
        return;
    }

    public static double[][] transposeMatrix(double[][] m) {
        double[][] temp = new double[m[0].length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;
    }

    static double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix) {
        double[][] result = new double[firstMatrix.length][secondMatrix[0].length];

        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[row].length; col++) {
                result[row][col] = multiplyMatricesCell(firstMatrix, secondMatrix, row, col);
            }
        }

        return result;
    }

    static double multiplyMatricesCell(double[][] firstMatrix, double[][] secondMatrix, int row, int col) {
        double cell = 0;
        for (int i = 0; i < secondMatrix.length; i++) {
            cell += firstMatrix[row][i] * secondMatrix[i][col];
        }
        return cell;
    }
}
