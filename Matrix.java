import java.util.Random;

public class Matrix {
    private double[] cells;
    private int width, height;

    private Matrix(int width, int height) {
        this.width = width;
        this.height = height;
        this.cells = new double[width * height];
    }

    private Matrix(double[] cells) {
        this.cells = cells.clone();
        this.width = cells.length;
        this.height = 1;
    }

    private Matrix(double[] cells, int width, int height) {
        this.cells = cells.clone();
        this.width = width;
        this.height = height;
    }

    private Matrix(int width, int height, double[] cells) {
        this.cells = cells.clone();
        this.width = width;
        this.height = height;
    }

    public static Matrix create(int width, int height) {
        return new Matrix(width, height);
    }

    public static Matrix create(double[] cells) {
        return new Matrix(cells);
    }

    public static Matrix create(double[] cells, int width, int height) {
        return new Matrix(cells, width, height);
    }

    public static Matrix create(int width, int height, double[] cells) {
        return new Matrix(width, height, cells);
    }

    public Matrix clone() {
        return new Matrix(this.getCells(), this.getWidth(), this.getHeight());
    }

    public void set(int row, int col, double input) {
        this.cells[this.width * row + col] = input;
    }

    public void set(String wildcard, int col, Matrix input) {
        if (wildcard.equals("*")) {
            for (int i = 0; i < this.width; i++) {
                this.set(i, col, input.get(i, 1));
            }
        }
    }

    public void set(int row, String wildcard, Matrix input) {
        if (wildcard.equals("*")) {
            for (int i = 0; i < this.height; i++) {
                this.set(row, i, input.get(1, i));
            }
        }
    }

    public void set(String wildcard, String otherWildcard, Matrix input) {
        if (wildcard.equals("*") && otherWildcard.equals("*")) {
            for (int i = 0; i < this.cells.length; i++) {
                this.cells[i] = input.cells[i];
            }
        }
    }

    public double get(int row, int col) {
        return this.cells[this.width * row + col];
    }

    public Matrix get(int row, String wildcard) {
        if (wildcard.equals("*")) {
            double[] output = new double[this.width];
            for (int i = 0; i < output.length; i++) {
                output[i] = this.get(row, i);
            }
            return new Matrix(output, this.width, 1);
        }
        return null;
    }

    public Matrix get(String wildcard, int col) {
        if (wildcard.equals("*")) {
            double[] output = new double[this.height];
            for (int i = 0; i < output.length; i++) {
                output[i] = this.get(i, col);
            }
            return new Matrix(output, 1, this.height);
        }
        return null;
    }

    public Matrix get(String wildcard, String otherWildcard) {
        if (wildcard.equals("*") && otherWildcard.equals("*")) {
            return this.clone();
        }
        return null;
    }

    public double[] getCells() {
        return this.cells.clone();
    }

    public int getWidth() {
        return this.width;
    }

    public int getHeight() {
        return this.height;
    }

    public double getSum() {
        double output = 0;
        for (double cell : this.cells) {
            output += cell;
        }
        return output;
    }

    public double getProduct() {
        double output = 1;
        for (double cell : this.cells) {
            output *= cell;
        }
        return output;
    }

    public boolean isRowVector() {
        return this.height == 1;
    }

    public boolean isColumnVector() {
        return this.width == 1;
    }

    public boolean isVector() {
        return isColumnVector() || isRowVector();
    }

    public Matrix setTranspose() {

        int temp = this.width;
        this.width = this.height;
        this.height = temp;
        return this;
    }

    public Matrix transpose() {
        return this.clone().setTranspose();
    }

    public Matrix setShape(int width, int height) {
        this.width = width;
        this.height = height;
        return this;
    }

    public Matrix shape(int width, int height) {
        return this.clone().setShape(width, height);
    }

    public Matrix setComponentProduct(Matrix input) {
        for (int i = 0; i < this.cells.length; i++) {
            this.cells[i] *= input.cells[i];
        }
        return this;
    }

    public Matrix componentProduct(Matrix input) {
        return this.clone().setComponentProduct(input);
    }

    public Matrix setAdd(Matrix input) {
        for (int i = 0; i < this.cells.length; i++) {
            this.cells[i] += input.cells[i];
        }
        return this;
    }

    public Matrix add(Matrix input) {
        return this.clone().setAdd(input);
    }

    public Matrix multiply(Matrix input) {
        Matrix output = new Matrix(this.width, input.height);
        for (int i = 0; i < this.width; i++) {
            for (int j = 0; j < input.height; j++) {
                for (int k = 0; k < this.height; k++) { // this.height = input.width (hopefully)
                    // output[i,j] += this[i,k] * input[k,j]
                    output.set(i, j, output.get(i, j) + this.get(i, k) * input.get(k, j));
                }
            }
        }
        return output;
    }

    public Matrix innerProduct(Matrix input) {
        this.transpose();
        Matrix output = this.multiply(input);
        this.transpose();
        return output;
    }

    public Matrix outerProduct(Matrix input) {
        input.transpose();
        Matrix output = this.multiply(input);
        input.transpose();
        return output;
    }

    public Matrix setScalarProduct(double input) {
        for (int i = 0; i < this.cells.length; i++) {
            this.cells[i] *= input;
        }
        return this;
    }

    public Matrix scalarProduct(double input) {
        return this.clone().setScalarProduct(input);
    }

    public Matrix setScalarAdd(double input) {
        for (int i = 0; i < this.cells.length; i++) {
            this.cells[i] += input;
        }
        return this;
    }

    public Matrix scalarAdd(double input) {
        return this.clone().setScalarAdd(input);
    }

    public Matrix setRandomize(int seed) {
        Random rand = new Random(seed);
        for (int i = 0; i < this.cells.length; i++) {
            this.cells[i] = rand.nextGaussian();
        }
        return this;
    }

    public Matrix setZero() {
        for (int i = 0; i < this.cells.length; i++) {
            this.cells[i] = 0;
        }
        return this;
    }
}