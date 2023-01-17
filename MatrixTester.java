
public class MatrixTester {
    public static void main(String[] args) throws Exception {
        Matrix square = Matrix.create(new double[] { 1, 2, 3, 4,5,6 }, 2, 3);
        square.print();
        square.transpose();
        square.print();
    }
}