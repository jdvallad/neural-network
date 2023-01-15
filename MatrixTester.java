import java.text.DecimalFormat;

public class MatrixTester {
    public static void main(String[] args) throws Exception {
        double[] temp = new double[] { 1, 2, 3, 4, 5, 6 };
        Matrix a = Matrix.create(temp, 1, 6);
        Matrix b = Matrix.create(temp, 6, 1);
        Matrix c = Matrix.create(temp, 2, 3);
        Matrix d = Matrix.create(temp, 3, 2);

        Matrix square = Matrix.create(new double[] { 1, 2, 3, 4 }, 2, 2);
        Matrix result = Matrix.identity(2);
        result.print();
        for (int i = 0; i < 10; i++) {
            result.product(square).print();
        }
    }

}
