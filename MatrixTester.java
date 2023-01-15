public class MatrixTester {
    public static void main(String[] args) throws Exception {
        double[] temp = new double[] { 1, 2, 3, 4, 5, 6 };
        Matrix A = Matrix.create(temp, 1, 6);
        Matrix B = Matrix.create(temp, 6, 1);
        Matrix C = Matrix.create(temp, 2, 3);
        Matrix D = Matrix.create(temp, 3, 2);
        A.setShape(2,3).print();
    }

}
