import java.util.Random;

public class Matrix {
    private double[] cells;
    private int rows, columns;

    private Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.cells = new double[this.rows * this.columns];
    }

    private Matrix(double[] cells) {
        this.cells = cells.clone();
        this.rows = 1;
        this.columns = cells.length;
    }

    private Matrix(double[] cells, int rows, int columns) {
        this.cells = cells.clone();
        this.rows = rows;
        this.columns = columns;
    }

    private Matrix(int rows, int columns, double[] cells) {
        this.cells = cells.clone();
        this.rows = rows;
        this.columns = columns;
    }

    public static Matrix create(int rows, int columns) {
        return new Matrix(rows, columns);
    }

    public static Matrix create(double[] cells) {
        return new Matrix(cells);
    }

    public static Matrix create(double[] cells, int rows, int columns) {
        return new Matrix(cells, rows, columns);
    }

    public static Matrix create(int rows, int columns, double[] cells) {
        return new Matrix(rows, columns, cells);
    }

    public Matrix clone() {
        return Matrix.create(this.getCells(), this.getRows(), this.getColumns());
    }

    public void set(int row, int col, double input) {
        this.cells[this.columns * row + col] = input;
    }

    public void set(String wildcard, int col, Matrix input) {
        if (wildcard.equals("*")) {
            for (int i = 0; i < this.rows; i++) {
                this.set(i, col, input.get(i, 1));
            }
        }
    }

    public void set(int row, String wildcard, Matrix input) {
        if (wildcard.equals("*")) {
            for (int i = 0; i < this.columns; i++) {
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
        return this.cells[this.columns * row + col];
    }

    public Matrix get(int row, String wildcard) {
        if (wildcard.equals("*")) {
            double[] output = new double[this.columns];
            for (int i = 0; i < output.length; i++) {
                output[i] = this.get(row, i);
            }
            return new Matrix(output, 1, this.columns);
        }
        return null;
    }

    public Matrix get(String wildcard, int col) {
        if (wildcard.equals("*")) {
            double[] output = new double[this.rows];
            for (int i = 0; i < output.length; i++) {
                output[i] = this.get(i, col);
            }
            return new Matrix(output, this.rows, 1);
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

    public int getRows() {
        return this.rows;
    }

    public int getColumns() {
        return this.columns;
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
        return this.rows == 1;
    }

    public boolean isColumnVector() {
        return this.columns == 1;
    }

    public boolean isVector() {
        return isColumnVector() || isRowVector();
    }

    public Matrix setTranspose() {
        int temp = this.rows;
        this.rows = this.columns;
        this.columns = temp;
        return this;
    }

    public Matrix transpose() {
        return this.clone().setTranspose();
    }

    public Matrix setShape(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        return this;
    }

    public Matrix shape(int rows, int columns) {
        return this.clone().setShape(rows, columns);
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
        if (this == input) {
            return this.multiply(this.clone());
        }
        Matrix output = new Matrix(this.rows, input.columns);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < input.columns; j++) {
                for (int k = 0; k < this.columns; k++) { // this.columns = input.rows (hopefully)
                    // output[i,j] += this[i,k] * input[k,j]
                    output.set(i, j, output.get(i, j) + this.get(i, k) * input.get(k, j));
                }
            }
        }
        return output;
    }

    public Matrix outerProduct(Matrix input) {
        if (this == input) {
            return this.outerProduct(this.clone());
        }
        this.setTranspose();
        Matrix output = this.multiply(input);
        this.setTranspose();
        return output;
    }

    public Matrix innerProduct(Matrix input) {
        if (this == input) {
            return this.innerProduct(this.clone());
        }
        input.setTranspose();
        Matrix output = this.multiply(input);
        input.setTranspose();
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

    public void print() {
        for (int r = 0; r < this.rows - 1; r++) {
            System.out.print("       [");
            for (int c = 0; c < this.columns - 1; c++) {
                System.out.print("" + this.get(r, c) + ",");
            }
            System.out.print("" + this.get(r, this.columns - 1) + "]\r\n");
        }
        System.out.print("Matrix([");
        for (int c = 0; c < this.columns - 1; c++) {
            System.out.print("" + this.get(this.rows - 1, c) + ",");
        }
        System.out.print("" + this.get(this.rows - 1, this.columns - 1) + "])\r\n\r\n");
    }
}