
public class MatrixTester {
    public static void main(String[] args) throws Exception {
        Matrix square = Matrix.create(new double[] { 1, 2, 3, 4 }, 2, 2);
        square.print();
        Matrix result = Matrix.identity(2);
        result.print();
        for (int i = 0; i < 10; i++) {
            result.product(square).print();
        }
    }
}